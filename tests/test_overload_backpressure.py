from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from v2v.engine import Session
from v2v.workers import BusyError, ModelGate


class OverloadBackpressureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates: list[ModelGate] = []

    def tearDown(self) -> None:
        for g in self.gates:
            g.close()

    def _gate(self, name: str, depth: int = 4) -> ModelGate:
        g = ModelGate(name, depth=depth)
        self.gates.append(g)
        return g

    def test_gate_occupancy_bounded_at_configured_depth(self) -> None:
        gate = self._gate("qwen", depth=3)
        block = threading.Event()
        holders = []

        # Fill 3 slots
        for i in range(3):
            holders.append(gate.submit(f"sess-{i}", lambda: block.wait(timeout=2.0), threading.Event()))

        time.sleep(0.02)
        self.assertEqual(len(gate.occupancy()), 3)
        self.assertEqual(gate.depth_now(), 3)

        # 4th unique session must be rejected with BusyError
        with self.assertRaises(BusyError) as ctx:
            gate.submit("sess-overflow", lambda: None, threading.Event())

        self.assertIn("qwen-queue-full", ctx.exception.reason)
        self.assertGreaterEqual(ctx.exception.retry_after_ms, 400)

        # Release blockers
        block.set()
        for h in holders:
            h.wait(1.0)

    def test_progressive_retry_after_ms(self) -> None:
        gate = self._gate("stt", depth=2)
        block = threading.Event()

        gate.submit("s1", lambda: block.wait(timeout=2.0), threading.Event())
        gate.submit("s2", lambda: block.wait(timeout=2.0), threading.Event())
        time.sleep(0.02)

        try:
            gate.submit("s3", lambda: None, threading.Event())
        except BusyError as exc:
            # retry = max(400, 400 * (len(occupied) + 1)) -> with 2 occupied, retry >= 1200ms
            self.assertGreaterEqual(exc.retry_after_ms, 800)
        finally:
            block.set()

    def test_single_session_replaces_own_pending_job_without_queue_expansion(self) -> None:
        gate = self._gate("qwen", depth=2)
        block = threading.Event()
        ran = []

        # S1 occupies running slot
        gate.submit("s1", lambda: block.wait(timeout=2.0), threading.Event())
        time.sleep(0.02)

        # S2 queues job 1
        j1 = gate.submit("s2", lambda: ran.append("j1"), threading.Event())
        # S2 queues job 2 (supersedes job 1)
        j2 = gate.submit("s2", lambda: ran.append("j2"), threading.Event())
        # S2 queues job 3 (supersedes job 2)
        j3 = gate.submit("s2", lambda: ran.append("j3"), threading.Event())

        # S2 still only occupies 1 slot
        self.assertEqual(len(gate.occupancy()), 2)

        block.set()
        j1.wait(1.0)
        j2.wait(1.0)
        j3.wait(1.0)

        # Only the latest job for S2 runs
        self.assertEqual(ran, ["j3"])

    def test_recovery_after_load_subsides(self) -> None:
        gate = self._gate("qwen", depth=2)
        block = threading.Event()

        h1 = gate.submit("s1", lambda: block.wait(timeout=2.0), threading.Event())
        h2 = gate.submit("s2", lambda: block.wait(timeout=2.0), threading.Event())
        time.sleep(0.02)

        # Overloaded
        with self.assertRaises(BusyError):
            gate.submit("s3", lambda: None, threading.Event())

        # Unblock running jobs
        block.set()
        h1.wait(1.0)
        h2.wait(1.0)
        time.sleep(0.05)

        # After completion, queue is drained and s3 can now submit and run successfully
        ran = []
        j = gate.submit("s3", lambda: ran.append("ok"), threading.Event())
        j.wait(1.0)
        self.assertEqual(ran, ["ok"])

    def test_engine_llm_busy_emits_event_and_resumes_listening(self) -> None:
        session = Session.__new__(Session)
        session.m = SimpleNamespace(
            cfg=SimpleNamespace(log_sensitive=False),
            knowledge=None,
            qwen_gate=self._gate("qwen", depth=1),
        )
        session.events = []
        session.emit = session.events.append
        session.cancel = threading.Event()
        session.session_id = "busy-session"
        session.request_id = "req-busy"
        session.gen = 1
        session._stop = threading.Event()
        session._lock = threading.Lock()
        session._last_successful_tool = None
        session.in_flight = True
        session.playing = False
        session.listening_enabled = True
        session.history = []
        session._tool_turns = {}

        # Fill the qwen gate with another session
        block = threading.Event()
        session.m.qwen_gate.submit("blocking-sess", lambda: block.wait(timeout=2.0), threading.Event())
        time.sleep(0.02)

        # LLM that triggers streaming through qwen gate
        class FakeLLM:
            def decide_tool_fast(self, *_a, **_k):
                return None
            def route_needle(self, *_a, **_k):
                from v2v.tools import ToolDecision
                return ToolDecision(tool=None, arguments={})
            def stream(self, *_a, **_k):
                yield "hello"

        session.m.llm = FakeLLM()
        now = time.perf_counter()
        job = {
            "text": "General question",
            "t": now,
            "t_stt_done": now,
            "t_llm_queued": now,
            "gen": 1,
            "stt": {},
            "utterance_id": 5,
            "request_id": "req-busy",
        }

        session._llm_one(job)
        block.set()

        # Check busy event was emitted
        busy_ev = [e for e in session.events if e.get("type") == "busy"]
        self.assertTrue(len(busy_ev) >= 1)
        self.assertIn("retry_after_ms", busy_ev[0])
        self.assertFalse(session.in_flight)

        # Check state resumed to listening
        state_ev = [e for e in session.events if e.get("type") == "state"]
        self.assertTrue(any(e.get("state") == "listening" for e in state_ev))


if __name__ == "__main__":
    unittest.main()
