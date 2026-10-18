from __future__ import annotations

import struct
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from v2v.engine import Session, _GenerationCancel
from v2v.workers import ModelGate


class CancellationStressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates: list[ModelGate] = []

    def tearDown(self) -> None:
        for g in self.gates:
            g.close()

    def _gate(self, name: str, depth: int = 4) -> ModelGate:
        g = ModelGate(name, depth=depth)
        self.gates.append(g)
        return g

    def test_cancel_session_a_while_queued_leaves_b_unaffected(self) -> None:
        gate = self._gate("qwen", depth=4)
        block = threading.Event()
        order = []

        # Blocker
        gate.submit("blocker", lambda: block.wait(timeout=1.0), threading.Event())
        time.sleep(0.02)

        cancel_a = threading.Event()
        cancel_b = threading.Event()

        job_a = gate.submit("sess-A", lambda: order.append("A"), cancel_a)
        job_b = gate.submit("sess-B", lambda: order.append("B"), cancel_b)

        # Cancel A while queued
        cancel_a.set()
        dropped = gate.cancel_session("sess-A")
        self.assertEqual(dropped, 1)

        block.set()
        job_a.wait(1.0)
        job_b.wait(1.0)

        self.assertIn("B", order)
        self.assertNotIn("A", order)
        self.assertTrue(job_a.skip or job_a._done.is_set())

    def test_cancel_session_a_while_generating_cancels_a_and_unblocks_b(self) -> None:
        gate = self._gate("qwen", depth=4)
        a_started = threading.Event()
        b_ran = threading.Event()
        order = []

        cancel_a = threading.Event()
        cancel_b = threading.Event()

        def work_a():
            order.append("start-A")
            a_started.set()
            # Simulate streaming loop checking cancel
            for _ in range(50):
                if cancel_a.is_set():
                    order.append("cancelled-A")
                    return
                time.sleep(0.02)
            order.append("finished-A")

        def work_b():
            order.append("run-B")
            b_ran.set()

        job_a = gate.submit("sess-A", work_a, cancel_a)
        job_b = gate.submit("sess-B", work_b, cancel_b)

        # Wait until A is actually running
        self.assertTrue(a_started.wait(1.0))
        # Trigger cancellation on A
        cancel_a.set()
        gate.cancel_session("sess-A")

        # B should run promptly (well before 50*20ms = 1s)
        t0 = time.perf_counter()
        self.assertTrue(b_ran.wait(1.5))
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.8)

        job_a.wait(1.0)
        job_b.wait(1.0)

        self.assertIn("cancelled-A", order)
        self.assertNotIn("finished-A", order)
        self.assertIn("run-B", order)

    def test_stale_audio_and_pcm_discarded_on_cancel(self) -> None:
        sess = Session.__new__(Session)
        sess.gen = 1
        sess.session_id = "sess-audio"
        sess.request_id = "req-1"
        emitted_audio = []

        def emit(msg):
            if isinstance(msg, tuple) and msg[0] == "audio":
                emitted_audio.append(msg[1])

        sess.emit = emit

        # Emitting audio with current generation 1 works
        chunk = np.zeros(16, dtype=np.float32)
        sess._emit_audio(1, chunk)
        self.assertEqual(len(emitted_audio), 1)
        gen_unpacked = struct.unpack_from("<I", emitted_audio[0][:4])[0]
        self.assertEqual(gen_unpacked, 1)

        # Bump generation to 2 (simulating cancel/interrupt)
        sess.gen = 2
        # Emitting audio from generation 1 must be silently discarded
        sess._emit_audio(1, chunk)
        self.assertEqual(len(emitted_audio), 1)  # Still 1, not 2

        # Emitting audio from generation 2 works
        sess._emit_audio(2, chunk)
        self.assertEqual(len(emitted_audio), 2)
        gen2 = struct.unpack_from("<I", emitted_audio[1][:4])[0]
        self.assertEqual(gen2, 2)

    def test_cancel_a_does_not_affect_b_audio_or_history(self) -> None:
        a = Session.__new__(Session)
        b = Session.__new__(Session)
        a.gen = 1
        b.gen = 1
        a.session_id = "A"
        b.session_id = "B"
        a.history = []
        b.history = []
        a.cancel = threading.Event()
        b.cancel = threading.Event()
        a._stop = threading.Event()
        b._stop = threading.Event()

        a_events = []
        b_events = []
        a.emit = a_events.append
        b.emit = b_events.append

        # Remember in B
        b._remember("User says hi", "Assistant says hello")
        self.assertEqual(len(b.history), 2)

        # Cancel A
        tok_a = _GenerationCancel(a, 1)
        tok_b = _GenerationCancel(b, 1)
        a.gen = 2
        a.cancel.set()

        # Check B is unaffected
        self.assertTrue(tok_a.is_set())
        self.assertFalse(tok_b.is_set())
        self.assertEqual(len(b.history), 2)
        self.assertEqual(b.gen, 1)
        self.assertFalse(b.cancel.is_set())

    def test_rapid_burst_cancellations_recover_cleanly(self) -> None:
        sess = Session.__new__(Session)
        sess.gen = 1
        sess.session_id = "burst"
        sess.request_id = "req-burst"
        sess.cancel = threading.Event()
        sess._stop = threading.Event()
        sess._lock = threading.Lock()
        sess._playback_done = threading.Event()
        sess._playback_done.set()
        sess.playing = True
        sess.in_flight = True
        sess.barge_armed = True
        sess._tts_q = __import__("queue").Queue()
        sess._llm_q = __import__("queue").Queue()
        sess._tool_turns = {}
        sess.vad = SimpleNamespace(
            set_playing=lambda *a, **k: None,
            reset=lambda: None,
        )
        sess.m = SimpleNamespace(
            qwen_gate=None,
            stt_gate=None,
            tts_gate=None,
            needle_gate=None,
        )
        events = []
        sess.emit = events.append
        sess.active_utt = 1
        sess.listening_enabled = True

        # Perform 10 rapid interrupts
        for i in range(10):
            sess.interrupt(f"rapid-{i}")

        self.assertEqual(sess.gen, 11)
        self.assertFalse(sess.cancel.is_set())  # Must be cleared for next generation
        self.assertFalse(sess.playing)
        self.assertFalse(sess.in_flight)
        # Verify interrupt events emitted
        interrupt_events = [e for e in events if e.get("type") == "interrupt"]
        self.assertEqual(len(interrupt_events), 10)


if __name__ == "__main__":
    unittest.main()
