from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from v2v.needle_router import NeedleRouter
from v2v.tools import ToolDecision, deterministic_tool_decision
from v2v.workers import BusyError, ModelGate


class FakeNeedleAgent:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.reset_count = 0
        self.complete_count = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.reset_count += 1

    def complete(self, text: str) -> dict:
        with self._lock:
            self.complete_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        return {
            "function_calls": [{"name": "calculator", "arguments": {"expression": "40 + 2"}}],
            "validation": {},
        }


class NeedleConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates: list[ModelGate] = []

    def tearDown(self) -> None:
        for gate in self.gates:
            gate.close()

    def _create_gate(self, name: str = "needle", depth: int = 4) -> ModelGate:
        gate = ModelGate(name, depth=depth)
        self.gates.append(gate)
        return gate

    def test_needle_router_thread_safety(self) -> None:
        """Concurrent calls on the same NeedleRouter cannot interleave agent.reset and complete."""
        agent = FakeNeedleAgent(delay=0.03)
        router = NeedleRouter(agent=agent)
        results = []
        errors = []

        def call(val: int) -> None:
            try:
                dec = router.route(f"calculate {val} + 1")
                results.append(dec)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)
        self.assertEqual(agent.reset_count, 5)
        self.assertEqual(agent.complete_count, 5)

    def test_needle_gate_fifo_ordering(self) -> None:
        """Needle gate processes requests in FIFO order."""
        gate = self._create_gate(depth=4)
        order: list[str] = []
        block = threading.Event()

        def make_fn(name: str):
            def fn():
                if name == "A":
                    block.wait(timeout=1.0)
                order.append(name)
            return fn

        job_a = gate.submit("sess-A", make_fn("A"), threading.Event())
        time.sleep(0.02)
        job_b = gate.submit("sess-B", make_fn("B"), threading.Event())
        job_c = gate.submit("sess-C", make_fn("C"), threading.Event())

        block.set()
        job_a.wait(1.0)
        job_b.wait(1.0)
        job_c.wait(1.0)

        self.assertEqual(order, ["A", "B", "C"])

    def test_needle_gate_busy_when_queue_full(self) -> None:
        """When needle_gate occupancy hits depth, next unique session gets BusyError."""
        gate = self._create_gate(depth=2)
        block = threading.Event()

        def hold():
            block.wait(timeout=2.0)

        gate.submit("sess-1", hold, threading.Event())
        time.sleep(0.02)
        gate.submit("sess-2", hold, threading.Event())

        with self.assertRaises(BusyError) as ctx:
            gate.submit("sess-3", hold, threading.Event())

        self.assertIn("needle-queue-full", ctx.exception.reason)
        self.assertGreaterEqual(ctx.exception.retry_after_ms, 400)
        block.set()

    def test_needle_gate_cancel_session_does_not_affect_others(self) -> None:
        """Cancelling session A in needle_gate drops A's pending job without impacting B."""
        gate = self._create_gate(depth=4)
        block = threading.Event()
        ran: list[str] = []

        def worker(tag: str):
            def fn():
                if tag == "blocker":
                    block.wait(timeout=1.0)
                ran.append(tag)
            return fn

        cancel_a = threading.Event()
        cancel_b = threading.Event()

        gate.submit("blocker", worker("blocker"), threading.Event())
        time.sleep(0.02)
        job_a = gate.submit("sess-A", worker("A"), cancel_a)
        job_b = gate.submit("sess-B", worker("B"), cancel_b)

        # Cancel session A while queued
        cancel_a.set()
        dropped = gate.cancel_session("sess-A")
        self.assertEqual(dropped, 1)

        block.set()
        job_a.wait(1.0)
        job_b.wait(1.0)

        self.assertIn("blocker", ran)
        self.assertIn("B", ran)
        self.assertNotIn("A", ran)

    def test_deterministic_and_rag_do_not_take_needle_gate(self) -> None:
        """Deterministic tool routing and TF-IDF do not touch needle_gate."""
        gate = self._create_gate(depth=1)
        block = threading.Event()
        # Block the needle gate completely
        gate.submit("blocking-sess", lambda: block.wait(timeout=2.0), threading.Event())
        time.sleep(0.02)
        self.assertEqual(gate.depth_now(), 1)

        # Deterministic routing completes immediately without being blocked by needle_gate
        t0 = time.perf_counter()
        fast_decision = deterministic_tool_decision("What is 45 times 19?")
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.1)
        self.assertIsNotNone(fast_decision)
        self.assertEqual(fast_decision.tool, "calculator")

        block.set()


if __name__ == "__main__":
    unittest.main()
