from __future__ import annotations

import json
import threading
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from v2v import server
from v2v.config import Settings
from v2v.engine import Session, _GenerationCancel
from v2v.workers import BusyError, ModelGate


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, models, emit, session_id=None):
        self.m = models
        self.emit = emit
        self.session_id = session_id or "sess"
        self.closed = False
        self.history: list[dict[str, str]] = []
        self.listening_enabled = False
        self.pcm: list[bytes] = []
        self.interrupted: list[str] = []
        self.gen = 0
        FakeSession.instances.append(self)

    def close(self) -> None:
        self.closed = True

    def push_pcm16(self, raw: bytes) -> None:
        self.pcm.append(raw)
        self.emit(("audio", b"\x01\x00\x00\x00" + raw[:4]))

    def set_listen(self, enabled: bool, reason: str = "") -> None:
        self.listening_enabled = bool(enabled)

    def interrupt(self, reason: str = "") -> None:
        self.interrupted.append(reason)
        self.gen += 1

    def playback_done(self) -> None:
        return

    def reset_history(self) -> None:
        self.history.clear()


@contextmanager
def running_app():
    FakeSession.instances = []
    with (
        patch(
            "v2v.server.load_models",
            return_value=SimpleNamespace(
                asr=SimpleNamespace(device="cpu"),
                llm=SimpleNamespace(backend="llama", gpu_offload=False, fallback_reason=""),
                tts=SimpleNamespace(model=object()),
            ),
        ),
        patch("v2v.server.dump_ready", return_value=json.dumps({"ready": True})),
        patch("v2v.server.Session", FakeSession),
    ):
        with TestClient(server.app) as client:
            yield client
    FakeSession.instances = []
    with server._sessions_lock:
        server._sessions.clear()


class ModelGateTests(unittest.TestCase):
    def tearDown(self) -> None:
        gate = getattr(self, "gate", None)
        if gate is not None:
            gate.close()

    def test_fifo_and_busy(self) -> None:
        self.gate = ModelGate("qwen", depth=2)
        release = threading.Event()
        started: list[str] = []

        def hold(name: str):
            def fn():
                started.append(name)
                release.wait(timeout=1.0)

            return fn

        cancel = threading.Event()
        job_a = self.gate.submit("A", hold("A"), cancel)
        job_b = self.gate.submit("B", hold("B"), cancel)
        with self.assertRaises(BusyError) as ctx:
            self.gate.submit("C", hold("C"), cancel)
        self.assertGreaterEqual(ctx.exception.retry_after_ms, 400)
        time.sleep(0.05)
        self.assertEqual(started, ["A"])
        release.set()
        self.assertTrue(job_a.wait(1.0))
        self.assertTrue(job_b.wait(1.0))

    def test_one_session_cannot_monopolize(self) -> None:
        self.gate = ModelGate("qwen", depth=2)
        block = threading.Event()
        ran = []

        def stall():
            ran.append("a1")
            block.wait(timeout=1.0)

        cancel = threading.Event()
        job1 = self.gate.submit("A", stall, cancel)
        job2 = self.gate.submit("A", lambda: ran.append("a2"), cancel)
        job3 = self.gate.submit("A", lambda: ran.append("a3"), cancel)
        time.sleep(0.05)
        self.assertTrue(job2.skip or job2._done.is_set())
        block.set()
        self.assertTrue(job1.wait(1.0))
        self.assertTrue(job3.wait(1.0))
        self.assertIn("a3", ran)
        self.assertNotIn("a2", ran)

    def test_close_releases_pending_waiters(self) -> None:
        self.gate = ModelGate("shutdown", depth=2)
        block = threading.Event()
        running = self.gate.submit("A", lambda: block.wait(timeout=2.0), threading.Event())
        time.sleep(0.02)
        pending = self.gate.submit("B", lambda: None, threading.Event())
        self.gate.close()
        self.assertTrue(pending.wait(0.2))
        self.assertIsInstance(pending.error, RuntimeError)
        block.set()
        running.wait(1.0)

    def test_cancel_a_does_not_cancel_b(self) -> None:
        self.gate = ModelGate("qwen", depth=4)
        release = threading.Event()
        ran = []

        def hold(tag: str):
            def fn():
                ran.append(tag)
                if tag == "A":
                    release.wait(timeout=1.0)

            return fn

        cancel_a = threading.Event()
        cancel_b = threading.Event()
        job_a = self.gate.submit("A", hold("A"), cancel_a)
        job_b = self.gate.submit("B", hold("B"), cancel_b)
        time.sleep(0.05)
        dropped = self.gate.cancel_session("A")
        self.assertGreaterEqual(dropped, 0)
        cancel_a.set()
        release.set()
        self.assertTrue(job_a.wait(1.0))
        self.assertTrue(job_b.wait(1.0))
        self.assertIn("B", ran)

    def test_stale_skip_never_runs(self) -> None:
        self.gate = ModelGate("qwen", depth=4)
        ran = []
        cancel = threading.Event()
        job = self.gate.submit("A", lambda: ran.append("nope"), cancel)
        job.skip = True
        time.sleep(0.1)
        self.assertEqual(ran, [])

    def test_cancel_queued_stream_releases_iterator(self) -> None:
        self.gate = ModelGate("qwen", depth=2)
        release = threading.Event()
        self.gate.submit("A", lambda: release.wait(timeout=1.0), threading.Event())
        time.sleep(0.03)
        stream, job = self.gate.iterate("B", lambda: iter(("never",)), threading.Event())
        self.assertEqual(self.gate.cancel_session("B"), 1)
        consumed: list[str] = []
        reader = threading.Thread(target=lambda: consumed.extend(stream), daemon=True)
        reader.start()
        reader.join(timeout=0.3)
        release.set()
        self.assertFalse(reader.is_alive())
        self.assertTrue(job.wait(0.1))
        self.assertEqual(consumed, [])


class SessionIsolationTests(unittest.TestCase):
    def test_stt_gate_preserves_original_payload(self) -> None:
        gate = ModelGate("stt", depth=1)
        try:
            session = Session.__new__(Session)
            session.m = SimpleNamespace(
                cfg=SimpleNamespace(log_sensitive=False),
                asr=SimpleNamespace(decode=lambda _pcm: ("hello", {"audio_s": 1.0, "rtf": 0.1, "peak": 0.5})),
                stt_gate=gate,
            )
            session.session_id = "stt-session"
            session.request_id = None
            session.gen = 0
            session.cancel = threading.Event()
            session._stop = threading.Event()
            session.busy_partial = True
            session.last_stt_wait_s = 0.0
            events = []
            session.emit = events.append
            session._asr_one({"pcm": __import__("numpy").zeros(16000), "final": False, "t": time.perf_counter()})
            self.assertFalse(session.busy_partial)
            self.assertTrue(any(event.get("type") == "partial" for event in events))
        finally:
            gate.close()

    def test_histories_are_independent(self) -> None:
        a = Session.__new__(Session)
        b = Session.__new__(Session)
        a.history = []
        b.history = []
        a._remember("hello from a", "reply a")
        self.assertEqual(a.history[0]["content"], "hello from a")
        self.assertEqual(b.history, [])
        b._remember("hello from b", "reply b")
        self.assertEqual(a.history[1]["content"], "reply a")
        self.assertEqual(b.history[0]["content"], "hello from b")

    def test_generation_cancel_is_per_session(self) -> None:
        a = Session.__new__(Session)
        b = Session.__new__(Session)
        a.gen = 1
        b.gen = 1
        a.cancel = threading.Event()
        b.cancel = threading.Event()
        a._stop = threading.Event()
        b._stop = threading.Event()
        tok_a = _GenerationCancel(a, 1)
        tok_b = _GenerationCancel(b, 1)
        a.gen = 2
        self.assertTrue(tok_a.is_set())
        self.assertFalse(tok_b.is_set())

    def test_stale_generation_audio_rejected(self) -> None:
        session = Session.__new__(Session)
        session.gen = 2
        session.session_id = "A"
        session.request_id = None
        emitted = []
        session.emit = emitted.append
        session._emit_audio(1, __import__("numpy").zeros(8, dtype="float32"))
        self.assertEqual(emitted, [])


class WebSocketRegistryTests(unittest.TestCase):
    def test_lan_http_and_websocket_require_token_and_trusted_origin(self) -> None:
        cfg = Settings(
            host="0.0.0.0",
            allow_lan=True,
            auth_token="test-secret",
            allowed_origins=["http://10.0.0.5:8088"],
        )
        with patch.object(server, "settings", cfg):
            with running_app() as client:
                self.assertEqual(client.get("/").status_code, 401)
                self.assertEqual(client.get("/?token=test-secret").status_code, 200)
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect("/ws", headers={"origin": "http://10.0.0.5:8088"}):
                        pass
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect(
                        "/ws?token=test-secret", headers={"origin": "https://evil.example"}
                    ):
                        pass
                with client.websocket_connect(
                    "/ws?token=test-secret", headers={"origin": "http://10.0.0.5:8088"}
                ) as ws:
                    self.assertTrue(json.loads(ws.receive_text())["ready"])

    def test_oversized_pcm_is_rejected(self) -> None:
        cfg = Settings(max_pcm_chunk_bytes=8)
        with patch.object(server, "settings", cfg):
            with running_app() as client:
                with client.websocket_connect("/ws") as ws:
                    json.loads(ws.receive_text())
                    ws.send_bytes(b"0123456789")
                    error = json.loads(ws.receive_text())
                    self.assertEqual(error["type"], "error")
                    self.assertIn("too large", error["text"])

    def test_two_sockets_stay_connected_and_second_does_not_close_first(self) -> None:
        with running_app() as client:
            with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
                r1 = json.loads(ws1.receive_text())
                r2 = json.loads(ws2.receive_text())
                self.assertTrue(r1.get("ready") or r1.get("session_id"))
                self.assertNotEqual(r1["session_id"], r2["session_id"])
                self.assertEqual(len(FakeSession.instances), 2)
                self.assertFalse(FakeSession.instances[0].closed)
                self.assertFalse(FakeSession.instances[1].closed)
                with server._sessions_lock:
                    self.assertEqual(len(server._sessions), 2)

    def test_disconnect_cleans_only_that_session(self) -> None:
        with running_app() as client:
            with client.websocket_connect("/ws") as ws1:
                json.loads(ws1.receive_text())
                with client.websocket_connect("/ws") as ws2:
                    json.loads(ws2.receive_text())
                    sid1, sid2 = FakeSession.instances[0].session_id, FakeSession.instances[1].session_id
                time.sleep(0.05)
                self.assertTrue(FakeSession.instances[1].closed)
                self.assertFalse(FakeSession.instances[0].closed)
                with server._sessions_lock:
                    self.assertIn(sid1, server._sessions)
                    self.assertNotIn(sid2, server._sessions)

    def test_audio_from_a_never_reaches_b(self) -> None:
        with running_app() as client:
            with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
                json.loads(ws1.receive_text())
                json.loads(ws2.receive_text())
                a, b = FakeSession.instances
                ws1.send_bytes(b"\x00\x01" * 8)
                got = ws1.receive_bytes()
                self.assertTrue(got.startswith(b"\x01\x00\x00\x00"))
                self.assertEqual(len(a.pcm), 1)
                self.assertEqual(b.pcm, [])
                self.assertIsNot(a.emit, b.emit)

    def test_reset_and_interrupt_are_session_local(self) -> None:
        with running_app() as client:
            with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
                json.loads(ws1.receive_text())
                json.loads(ws2.receive_text())
                a, b = FakeSession.instances
                a.history = [{"role": "user", "content": "keep me"}]
                b.history = [{"role": "user", "content": "also keep"}]
                ws1.send_json({"type": "reset"})
                time.sleep(0.05)
                self.assertEqual(a.history, [])
                self.assertEqual(b.history, [{"role": "user", "content": "also keep"}])
                ws1.send_json({"type": "listen", "enabled": True})
                ws1.send_json({"type": "barge", "source": "button"})
                time.sleep(0.05)
                self.assertEqual(b.interrupted, [])
                self.assertTrue(a.interrupted)

    def test_tool_and_rag_events_carry_session_id(self) -> None:
        session = Session.__new__(Session)
        session.session_id = "sess-a"
        session.request_id = "req-a"
        session.gen = 3
        events = []
        session.emit = events.append
        session._ev("rag", sources=["handbook.md"], utterance_id=9)
        session._ev("bot", text="PTO is 20 days", utterance_id=9)
        self.assertTrue(all(ev["session_id"] == "sess-a" for ev in events))
        self.assertTrue(all(ev["generation_id"] == 3 for ev in events))


class QueueBusyTests(unittest.TestCase):
    def test_queue_full_raises_busy(self) -> None:
        gate = ModelGate("qwen", depth=1)
        try:
            block = threading.Event()
            gate.submit("A", lambda: block.wait(timeout=1.0), threading.Event())
            time.sleep(0.03)
            with self.assertRaises(BusyError):
                gate.submit("B", lambda: None, threading.Event())
            block.set()
        finally:
            gate.close()


class NeedleGateTests(unittest.TestCase):
    def test_needle_does_not_take_qwen_model_lock(self) -> None:
        from v2v.llm import TinyLLM
        from v2v.tools import ToolDecision

        llm = TinyLLM.__new__(TinyLLM)
        llm.cfg = SimpleNamespace(log_sensitive=False)
        llm._model_lock = threading.RLock()
        llm._needle_router = SimpleNamespace(route=lambda user, cancel=None: ToolDecision(tool=None, arguments={}))
        held = threading.Event()
        release = threading.Event()

        def hog() -> None:
            with llm._model_lock:
                held.set()
                release.wait(timeout=2.0)

        hogger = threading.Thread(target=hog, daemon=True)
        hogger.start()
        self.assertTrue(held.wait(1.0))
        t0 = time.perf_counter()
        decision = llm.route_needle("tell me a joke", threading.Event(), session_id="A")
        elapsed = time.perf_counter() - t0
        release.set()
        hogger.join(timeout=1.0)
        self.assertLess(elapsed, 0.4)
        self.assertIsNone(decision.tool)

    def test_decide_tool_fast_skips_needle(self) -> None:
        from v2v.llm import TinyLLM

        llm = TinyLLM.__new__(TinyLLM)
        llm.cfg = SimpleNamespace(log_sensitive=False)
        llm._needle_router = SimpleNamespace(route=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("needle")))
        fast = llm.decide_tool_fast("What is 45 times 19?", threading.Event(), session_id="A")
        self.assertIsNotNone(fast)
        self.assertEqual(fast.tool, "calculator")
        ambiguous = llm.decide_tool_fast("please compute some arithmetic", threading.Event(), session_id="A")
        self.assertIsNone(ambiguous)

    def test_engine_needle_uses_gate_and_busy(self) -> None:
        from v2v.tools import ToolDecision

        session = Session.__new__(Session)
        session.m = SimpleNamespace(cfg=SimpleNamespace(log_sensitive=False), knowledge=None)
        session.events = []
        session.emit = session.events.append
        session.cancel = threading.Event()
        session.session_id = "sess-a"
        session.request_id = "req-a"
        session.gen = 1
        session._stop = threading.Event()
        session._lock = threading.Lock()
        session._last_successful_tool = None
        session.in_flight = True
        session.playing = False
        session.listening_enabled = True
        session.history = []
        session._tool_turns = {}
        needle_calls: list[str] = []

        class LLM:
            def decide_tool_fast(self, *_a, **_k):
                return None

            def route_needle(self, _user, _cancel, session_id=None):
                needle_calls.append(session_id or "")
                return ToolDecision(tool=None, arguments={})

        session.m.llm = LLM()
        session.m.needle_gate = ModelGate("needle", depth=4)
        session._stream_llm_answer = lambda *_a, **_k: None
        now = time.perf_counter()
        job = {
            "text": "Tell me a joke",
            "t": now,
            "t_stt_done": now,
            "t_llm_queued": now,
            "gen": 1,
            "stt": {},
            "utterance_id": 3,
            "request_id": "req-a",
        }
        try:
            session._llm_one(job)
            self.assertEqual(needle_calls, ["sess-a"])
            gate = ModelGate("needle", depth=1)
            session.m.needle_gate = gate
            block = threading.Event()
            gate.submit("other", lambda: block.wait(timeout=1.0), threading.Event())
            time.sleep(0.05)
            session.in_flight = True
            session._llm_one({**job, "utterance_id": 4})
            block.set()
            busy = [e for e in session.events if e.get("type") == "busy"]
            self.assertTrue(busy)
            self.assertEqual(busy[-1]["reason"], "needle-queue-full")
        finally:
            session.m.needle_gate.close()

    def test_cancel_a_needle_does_not_cancel_b(self) -> None:
        gate = ModelGate("needle", depth=4)
        ran: list[str] = []
        release = threading.Event()

        def hold(tag: str):
            def fn():
                ran.append("start-" + tag)
                if tag == "A":
                    release.wait(timeout=1.0)
                ran.append("end-" + tag)

            return fn

        try:
            cancel_a = threading.Event()
            cancel_b = threading.Event()
            job_a = gate.submit("A", hold("A"), cancel_a)
            job_b = gate.submit("B", hold("B"), cancel_b)
            time.sleep(0.05)
            cancel_a.set()
            gate.cancel_session("A")
            release.set()
            self.assertTrue(job_a.wait(1.0))
            self.assertTrue(job_b.wait(1.0))
            self.assertIn("end-B", ran)
        finally:
            gate.close()


if __name__ == "__main__":
    unittest.main()
