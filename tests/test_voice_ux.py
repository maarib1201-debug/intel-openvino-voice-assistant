from __future__ import annotations

import time
import unittest
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v2v.asr import SR, WIN, EnergyVad, audio_quality_report
from v2v.config import Settings
from v2v.engine import Session


class _FakeSilero:
    def __init__(self, _path):
        self.ok = True
        self.window = WIN

    def prob(self, win):
        return 1.0 if float(np.sqrt(np.mean(win * win) + 1e-12)) >= 0.01 else 0.0

    def reset(self):
        pass


def _frame(level: float, n: int = 1) -> np.ndarray:
    return np.full(WIN * n, level, dtype=np.float32)


class VoiceUxTests(unittest.TestCase):
    def _vad(self, **overrides) -> EnergyVad:
        cfg = Settings(
            barge_confirm_s=overrides.get("barge_confirm_s", 0.20),
            barge_rms=overrides.get("barge_rms", 0.026),
            barge_echo_ratio=overrides.get("barge_echo_ratio", 1.7),
            speech_start_s=overrides.get("speech_start_s", 0.32),
            min_silence_s=overrides.get("min_silence_s", 0.20),
            post_tts_ignore_s=overrides.get("post_tts_ignore_s", 0.16),
        )
        with patch("v2v.asr.SileroSpeech", _FakeSilero):
            return EnergyVad(cfg)

    def test_one_noisy_frame_does_not_cancel(self) -> None:
        vad = self._vad()
        vad.set_playing(True)
        first = vad.push(_frame(0.08))
        second = vad.push(_frame(0.0, 2))
        self.assertFalse(first["barge"])
        self.assertFalse(second["barge"])
        self.assertFalse(vad.speech)

    def test_sustained_user_speech_confirms_before_utterance_end(self) -> None:
        vad = self._vad()
        vad.set_playing(True)
        event = vad.push(_frame(0.08, 8))
        self.assertTrue(event["barge"])
        self.assertTrue(vad.speech)
        self.assertEqual(event["segments"], [])
        self.assertIsNotNone(event["barge_candidate_at"])

    def test_interrupting_utterance_and_preroll_survive_cancel(self) -> None:
        vad = self._vad()
        vad.set_playing(True)
        event = vad.push(_frame(0.08, 8))
        self.assertTrue(event["barge"])
        captured_at_cancel = sum(frame.size for frame in vad.utt)
        self.assertGreaterEqual(captured_at_cancel, WIN * 7)

        vad.set_playing(False, ignore=False)
        ended = vad.push(np.concatenate([_frame(0.08, 4), _frame(0.0, 18)]))
        self.assertEqual(len(ended["segments"]), 1)
        self.assertTrue(ended["segment_reports"][0]["barge_utterance"])
        self.assertGreater(ended["segments"][0].size, captured_at_cancel)

    def test_post_tts_guard_retains_samples_in_preroll(self) -> None:
        vad = self._vad(post_tts_ignore_s=0.16)
        vad.set_playing(True)
        vad.set_playing(False, ignore=True)
        event = vad.push(_frame(0.08, 5))
        self.assertFalse(event["speech"])
        self.assertEqual(len(vad.preroll), 5)
        self.assertTrue(all(float(np.max(x)) > 0.05 for x in vad.preroll))

    def test_echo_below_barge_energy_gate_does_not_loop(self) -> None:
        vad = self._vad()
        vad.set_playing(True)
        for _ in range(30):
            event = vad.push(_frame(0.015))
            self.assertFalse(event["barge"])
        self.assertFalse(vad.speech)

    def test_engine_cancels_on_confirmed_speech_before_segment_end(self) -> None:
        session = Session.__new__(Session)
        candidate_at = time.perf_counter() - 0.22
        session.listening_enabled = True
        session.playing = True
        session.playing_since = time.perf_counter() - 1.0
        session.barge_armed = True
        session.gen = 4
        session.m = SimpleNamespace(cfg=SimpleNamespace(barge_holdoff_s=0.08))
        session.vad = SimpleNamespace(
            speech=True,
            utt=[_frame(0.08)],
            push=lambda _pcm: {
                "barge": True,
                "barge_candidate_at": candidate_at,
                "segments": [],
                "segment_reports": [],
            },
        )
        calls = []
        session.interrupt = lambda reason, **details: calls.append((reason, details))
        session.push_f32(_frame(0.08))
        self.assertEqual(calls[0][0], "confirmed-user-speech")
        self.assertTrue(calls[0][1]["barge_utterance_preserved"])
        self.assertGreaterEqual(calls[0][1]["barge_cancel_latency_ms"], 200)

    def test_audio_quality_report_has_capture_diagnostics(self) -> None:
        pcm = np.concatenate(
            [np.zeros(int(0.1 * SR), dtype=np.float32), _frame(0.2, 10), np.zeros(int(0.05 * SR), dtype=np.float32)]
        )
        report = audio_quality_report(pcm)
        self.assertGreaterEqual(report["leading_silence_ms"], 90)
        self.assertGreaterEqual(report["trailing_silence_ms"], 40)
        self.assertAlmostEqual(report["peak"], 0.2, places=3)
        self.assertEqual(report["clipping_ratio"], 0.0)
        self.assertGreater(report["crest_factor"], 1.0)
        self.assertLess(report["noise_floor_rms"], 0.01)

    def test_vad_boundary_report_records_preroll_eos_and_submission(self) -> None:
        vad = self._vad(min_silence_s=0.20)
        event = vad.push(np.concatenate([_frame(0.0, 4), _frame(0.08, 12), _frame(0.0, 18)]))
        self.assertEqual(len(event["segments"]), 1)
        report = event["segment_reports"][0]
        self.assertGreater(report["vad_trigger_samples"], 0)
        self.assertGreater(report["vad_retained_preroll_samples"], 0)
        self.assertGreater(report["vad_retained_trailing_samples"], 0)
        self.assertGreater(report["vad_submitted_samples"], report["vad_eos_samples"])

    def test_internal_pause_shorter_than_eos_does_not_split_utterance(self) -> None:
        vad = self._vad(min_silence_s=0.65)
        first = vad.push(np.concatenate([_frame(0.08, 12), _frame(0.0, 12), _frame(0.08, 8)]))
        self.assertEqual(first["segments"], [])
        finished = vad.push(_frame(0.0, 24))
        self.assertEqual(len(finished["segments"]), 1)

    def test_short_valid_utterance_is_retained(self) -> None:
        vad = self._vad(min_silence_s=0.20)
        event = vad.push(np.concatenate([_frame(0.08, 12), _frame(0.0, 18)]))
        self.assertEqual(len(event["segments"]), 1)
        self.assertGreaterEqual(event["segments"][0].size, int(0.25 * SR))

    def test_empty_audio_quality_report_is_safe(self) -> None:
        report = audio_quality_report(np.array([], dtype=np.float32))
        self.assertEqual(report["noise_floor_rms"], 0.0)
        self.assertEqual(report["crest_factor"], 0.0)

    def test_browser_cancel_and_adaptive_prebuffer_contract(self) -> None:
        source = (Path(__file__).parents[1] / "v2v" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("let adaptivePrebufferSec = 0.20", source)
        self.assertIn("Math.max(0.18, need)", source)
        self.assertIn("adaptive_prebuffer_updated", source)
        self.assertIn("reason=cannot_sustain_speed", source)
        self.assertIn("item.src.stop()", source)
        self.assertIn("pendingChunks = pendingChunks.filter((c) => c.gen !== gid)", source)
        self.assertIn("gen !== activeGeneration", source)
        self.assertIn("MAX_SCHEDULE_AHEAD_S = 0.72", source)
        self.assertIn("ahead >= MAX_SCHEDULE_AHEAD_S", source)

    def test_browser_mic_capture_diagnostics_are_opt_in_and_nonidentifying(self) -> None:
        source = (Path(__file__).parents[1] / "v2v" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('get("diagnostics") === "1"', source)
        self.assertIn('captureTelemetry("mic_capture_settings", captureSettings)', source)
        self.assertIn('captureTelemetry("mic_capture_format"', source)
        self.assertIn("requested_noise_suppression: true", source)
        self.assertIn("downsample_required", source)
        self.assertIn("capResampler.flush(), true", source)
        self.assertIn("capResampler.reset()", source)
        self.assertNotIn('"mic_start listening_enabled=true  " + tr.label', source)

    def test_stateful_browser_resampler_harness(self) -> None:
        root = Path(__file__).parents[1]
        completed = subprocess.run(
            ["node", "tests/capture_resampler_harness.js"], cwd=root, check=True,
            capture_output=True, text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["48000_60s"]["samples"], 960000)
        self.assertEqual(result["44100_60s"]["samples"], 960000)
        self.assertLess(result["processing_ms"]["p95"], 1.0)


if __name__ == "__main__":
    unittest.main()
