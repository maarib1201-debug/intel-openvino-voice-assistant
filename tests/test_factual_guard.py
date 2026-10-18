from __future__ import annotations

import queue
import threading
import time
import unittest
from types import SimpleNamespace

from v2v.engine import Session
from v2v.factual_guard import (
    FALLBACK_FACTS_FIRST,
    FALLBACK_LATER,
    FactualRiskGuard,
)


class FactualRiskGuardTests(unittest.TestCase):
    def test_new_institution_and_date_are_blocked(self) -> None:
        guard = FactualRiskGuard("What is the Nareth particle?")
        decision = guard.inspect(
            "It was discovered by researchers at Northbridge University in 1987."
        )
        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "unsupported-specificity")
        self.assertEqual(decision.text, FALLBACK_FACTS_FIRST)

    def test_dense_causal_chain_is_blocked_after_safe_summary(self) -> None:
        guard = FactualRiskGuard("How does this cooling system work?")
        safe = guard.inspect("It moves heat from inside the compartment to the surrounding room.")
        risky = guard.inspect(
            "It does this by creating a vacuum which then draws warm air through a vent."
        )
        self.assertFalse(safe.triggered)
        self.assertTrue(risky.triggered)
        self.assertEqual(risky.reason, "dense-causal-chain")
        self.assertEqual(risky.text, FALLBACK_LATER)

    def test_uncertainty_is_not_replaced(self) -> None:
        guard = FactualRiskGuard("Who invented the unfamiliar theorem?")
        text = "I'm not sure that theorem is real, so I can't verify an inventor."
        self.assertEqual(guard.inspect(text).text, text)

    def test_fictional_style_historical_and_architecture_claims_are_blocked(self) -> None:
        cases = [
            (
                "Who invented the unfamiliar compression theorem?",
                "The unfamiliar compression theorem was not invented by a single individual.",
            ),
            (
                "What happened in the unfamiliar protocol incident?",
                "A U.S. Navy SEAL team breached a secure facility in Tarsen, California.",
            ),
            (
                "What is the unfamiliar particle?",
                "It is from the Star Trek universe, specifically appearing in a named episode.",
            ),
            (
                "Explain the unfamiliar memory architecture.",
                "Zorvax memory is a specialized high-speed RAM architecture.",
            ),
        ]
        for user, answer in cases:
            with self.subTest(user=user):
                self.assertTrue(FactualRiskGuard(user).inspect(answer).triggered)

    def test_normal_common_answers_are_not_hedged(self) -> None:
        cases = [
            ("What is Python?", "Python is a general-purpose programming language."),
            ("What is Kubernetes?", "Kubernetes manages containerized applications across computers."),
            ("What is an API?", "An API lets software systems exchange requests and responses."),
            ("Why is the sky blue?", "Shorter wavelengths of sunlight scatter more strongly in the atmosphere."),
            ("Give me a productivity tip.", "Choose one important task and work on it without switching."),
        ]
        for user, answer in cases:
            with self.subTest(user=user):
                decision = FactualRiskGuard(user).inspect(answer)
                self.assertFalse(decision.triggered)
                self.assertEqual(decision.text, answer)

    def test_grounded_paths_are_bypassed(self) -> None:
        guard = FactualRiskGuard("What happened?", grounded=True)
        text = "Northbridge University reported this in 1987."
        self.assertFalse(guard.inspect(text).triggered)


class _StreamingLLM:
    llama = None

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def stream(self, *_args, **_kwargs):
        for word in self.answer.split(" "):
            yield word + " "

    def stream_with_tool_result(self, *_args, **_kwargs):
        yield from self.stream()


class GuardPipelineTests(unittest.TestCase):
    def _session(self, answer: str) -> Session:
        session = Session.__new__(Session)
        cfg = SimpleNamespace(
            factual_guard_enabled=True,
            factual_guard_specificity_score=2,
            factual_guard_causal_score=3,
            log_sensitive=False,
        )
        session.m = SimpleNamespace(cfg=cfg, llm=_StreamingLLM(answer), qwen_gate=None)
        session.events = []
        session.emit = session.events.append
        session.cancel = threading.Event()
        session.session_id = "guard-test"
        session.request_id = "request-test"
        session.gen = 1
        session._stop = threading.Event()
        session._lock = threading.Lock()
        session._tts_q = queue.Queue(maxsize=32)
        session.history = []
        session.in_flight = True
        session.listening_enabled = True
        return session

    def test_risky_sentence_is_never_queued_for_tts_or_history(self) -> None:
        answer = (
            "The device receives timing signals. "
            "A remote system calculates your position by triangulating the signal and then determines exact coordinates."
        )
        session = self._session(answer)
        now = time.perf_counter()
        job = {
            "text": "How does the positioning system work?",
            "t": now,
            "t_stt_done": now,
            "gen": 1,
            "stt": {},
            "utterance_id": 1,
        }
        session._stream_llm_answer(job)
        queued = []
        while not session._tts_q.empty():
            queued.append(session._tts_q.get_nowait())
        spoken = " ".join(item.get("text", "") for item in queued if not item.get("eos"))
        self.assertIn("receives timing signals", spoken)
        self.assertIn("not confident", spoken)
        self.assertNotIn("calculates your position", spoken)
        self.assertNotIn("calculates your position", session.history[-1]["content"])

    def test_authoritative_tool_stream_bypasses_guard(self) -> None:
        answer = "The calculator returned the exact result in 1987."
        session = self._session(answer)
        now = time.perf_counter()
        job = {
            "text": "What is the result?",
            "t": now,
            "t_stt_done": now,
            "gen": 1,
            "stt": {},
            "utterance_id": 2,
        }
        session._stream_llm_answer(job, tool_name="calculator", tool_result={"value": "1987"})
        spoken = " ".join(
            item.get("text", "")
            for item in list(session._tts_q.queue)
            if not item.get("eos")
        )
        self.assertIn("1987", spoken)
        self.assertNotIn("not confident", spoken)


if __name__ == "__main__":
    unittest.main()
