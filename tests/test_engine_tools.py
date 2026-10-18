from __future__ import annotations

import queue
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2v import engine


class _FakeLLM:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []

    def remember(self, user: str, bot: str) -> None:
        self.history.append((user, bot))


def _session(*, ack_delay: float = 0.02, workers: int = 2):
    session = engine.Session.__new__(engine.Session)
    cfg = SimpleNamespace(
        weather_timeout_s=1.0,
        tool_ack_delay_s=ack_delay,
        tool_workers=workers,
        default_location=None,
        log_sensitive=False,
    )
    session.m = SimpleNamespace(cfg=cfg, llm=_FakeLLM())
    session.events = []
    session.emit = session.events.append
    session.cancel = threading.Event()
    session.session_id = "test-session"
    session.request_id = "test-request"
    session.gen = 1
    session._stop = threading.Event()
    session._lock = threading.Lock()
    session._tool_executor = ThreadPoolExecutor(max_workers=workers)
    session._tool_slots = threading.BoundedSemaphore(workers * 2)
    session._tool_turns = {}
    session._llm_q = queue.Queue(maxsize=16)
    session._tts_q = queue.Queue(maxsize=32)
    session.in_flight = True
    session.listening_enabled = True
    session.history = []
    return session


def _job(gen: int = 1, uid: int = 7, text: str = "What's the weather in London?") -> dict:
    now = time.perf_counter()
    return {
        "text": text,
        "t": now,
        "t_stt_done": now,
        "t_llm_queued": now,
        "gen": gen,
        "stt": {},
        "utterance_id": uid,
    }


class EngineToolTests(unittest.TestCase):
    def tearDown(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session._stop.set()
            session._tool_executor.shutdown(wait=True, cancel_futures=True)

    def test_generation_cancel_stays_set_after_event_pulse_clears(self) -> None:
        self.session = _session()
        token = engine._GenerationCancel(self.session, 1)
        self.assertFalse(token.is_set())
        self.session.cancel.set()
        self.assertTrue(token.is_set())
        self.session.cancel.clear()
        self.assertFalse(token.is_set())
        self.session.gen = 2
        self.assertTrue(token.is_set())

    def test_fast_tool_has_no_acknowledgement(self) -> None:
        self.session = _session(ack_delay=0.05)
        with patch.object(engine, "execute_tool", return_value={"ok": True}), patch.object(
            engine, "acknowledgement_for", return_value="Checking the weather."
        ):
            self.session._start_tool(_job(), "get_weather", {"location": "London"}, time.perf_counter())
            continuation = self.session._llm_q.get(timeout=0.5)
            time.sleep(0.07)
        self.assertFalse(continuation["ack_started"])
        self.assertEqual(self.session.events, [])
        self.assertTrue(self.session._tts_q.empty())

    def test_slow_weather_ack_precedes_final_and_has_one_eos(self) -> None:
        self.session = _session(ack_delay=0.01)

        def slow_tool(*_args, **_kwargs):
            time.sleep(0.05)
            return {"ok": True}

        with patch.object(engine, "execute_tool", side_effect=slow_tool), patch.object(
            engine, "acknowledgement_for", return_value="Checking the weather."
        ):
            self.session._start_tool(_job(), "get_weather", {"location": "London"}, time.perf_counter())
            continuation = self.session._llm_q.get(timeout=0.5)
        self.assertTrue(continuation["ack_started"])
        with patch.object(engine, "format_tool_result", return_value="Final weather."), patch.object(
            engine, "requires_synthesis", return_value=False
        ):
            self.session._tool_result_one(continuation)
        queued = []
        while not self.session._tts_q.empty():
            queued.append(self.session._tts_q.get_nowait())
        self.assertEqual([item.get("text") for item in queued], ["Checking the weather.", "Final weather.", None])
        self.assertEqual(sum(bool(item.get("eos")) for item in queued), 1)
        deltas = [event for event in self.session.events if event.get("type") == "bot_delta"]
        self.assertEqual([event["phase"] for event in deltas], ["ack", "final"])
        self.assertTrue(all(event["generation_id"] == 1 for event in deltas))
        self.assertEqual(
            self.session.history,
            [
                {"role": "user", "content": continuation["text"]},
                {"role": "assistant", "content": "Final weather."},
            ],
        )

    def test_stale_tool_result_cannot_emit_or_queue(self) -> None:
        self.session = _session(ack_delay=0.01)
        release = threading.Event()

        def blocked(*_args, **_kwargs):
            release.wait(timeout=0.5)
            return {"location": "Dubai"}

        with patch.object(engine, "execute_tool", side_effect=blocked), patch.object(
            engine, "acknowledgement_for", return_value="Checking the weather."
        ):
            self.session._start_tool(_job(text="Weather in Dubai"), "get_weather", {"location": "Dubai"}, time.perf_counter())
            time.sleep(0.03)
            event_count = len(self.session.events)
            self.session.gen = 2
            release.set()
            time.sleep(0.05)
        self.assertEqual(len(self.session.events), event_count)
        self.assertTrue(self.session._llm_q.empty())
        self.assertEqual(self.session.history, [])

    def test_new_weather_can_finish_while_old_weather_is_blocked(self) -> None:
        self.session = _session(ack_delay=0.2, workers=2)
        release_old = threading.Event()

        def provider(_name, arguments, **_kwargs):
            if arguments["location"] == "Dubai":
                release_old.wait(timeout=0.5)
            return {"location": arguments["location"]}

        with patch.object(engine, "execute_tool", side_effect=provider), patch.object(
            engine, "acknowledgement_for", return_value="Checking the weather."
        ):
            self.session._start_tool(_job(gen=1, uid=1), "get_weather", {"location": "Dubai"}, time.perf_counter())
            self.session.gen = 2
            self.session._start_tool(_job(gen=2, uid=2), "get_weather", {"location": "London"}, time.perf_counter())
            london = self.session._llm_q.get(timeout=0.3)
            release_old.set()
            time.sleep(0.05)
        self.assertEqual(london["gen"], 2)
        self.assertEqual(london["tool_result"], {"location": "London"})
        self.assertTrue(self.session._llm_q.empty())

    def test_browser_adopts_user_generation_and_filters_stale_events(self) -> None:
        source = (Path(__file__).parents[1] / "v2v" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('msg.type === "user" && hasGeneration', source)
        self.assertIn('activeGeneration = taggedGeneration', source)
        self.assertIn('"audio_end"', source)
        self.assertIn('reason=stale_generation', source)

    def test_browser_persists_playback_telemetry_without_audio_or_text(self) -> None:
        source = (Path(__file__).parents[1] / "v2v" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('type: "playback_telemetry"', source)
        self.assertIn('localStorage.setItem("v2v_playback_telemetry"', source)
        self.assertIn('first_audio_packet_received', source)
        self.assertIn('playback_started_estimate', source)
        self.assertIn('queue_starvation', source)

    def test_structured_events_include_session_and_request_ids(self) -> None:
        self.session = _session()
        self.session._ev("state", state="off")
        self.assertEqual(self.session.events[0]["session_id"], "test-session")
        self.assertEqual(self.session.events[0]["request_id"], "test-request")

    def test_tts_queue_overflow_does_not_block(self) -> None:
        self.session = _session()
        self.session._tts_q = queue.Queue(maxsize=1)
        self.session._tts_q.put_nowait({"sentinel": True})
        self.session._enqueue_tts("hello", 1, time.perf_counter(), {}, 1, 7)
        self.assertEqual(self.session._tts_q.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
