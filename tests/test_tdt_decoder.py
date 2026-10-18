from __future__ import annotations

import unittest

import numpy as np

from v2v.asr_ov import BLANK, DURATIONS, MAX_SYMBOLS, greedy_tdt


class _Decoder:
    def __call__(self, inputs):
        h = inputs["h_in"] + 1
        return {"decoder_output": np.zeros((1, 1, 640), np.float32), "h_out": h, "c_out": inputs["c_in"] + 1}


class _Joint:
    def __init__(self, choices): self.choices = iter(choices)
    def __call__(self, _inputs):
        token, duration = next(self.choices)
        logits = np.full(1030, -100.0, np.float32)
        logits[token] = 10.0; logits[1025 + duration] = 10.0
        return {"logits": logits.reshape(1, 1, 1, -1)}


def _enc(frames: int) -> np.ndarray: return np.zeros((1, 1024, frames), np.float32)


class TdtDecoderTests(unittest.TestCase):
    def test_immediate_blank_advances_by_predicted_duration(self):
        trace = []
        ids = greedy_tdt(_enc(3), 3, _Decoder(), _Joint([(BLANK, 1)] * 3), trace)
        self.assertEqual(ids, []); self.assertEqual([row["frame"] for row in trace], [0, 1, 2])

    def test_multiple_nonblanks_share_frame_until_duration_advances(self):
        trace = []
        ids = greedy_tdt(_enc(2), 2, _Decoder(), _Joint([(7, 0), (7, 2)]), trace)
        self.assertEqual(ids, [7, 7])
        self.assertEqual([(row["frame"], row["duration"]) for row in trace], [(0, 0), (0, 2)])

    def test_zero_duration_max_symbol_guard_advances(self):
        trace = []
        choices = [(9, 0)] * (MAX_SYMBOLS * 2)
        ids = greedy_tdt(_enc(2), 2, _Decoder(), _Joint(choices), trace)
        self.assertEqual(len(ids), MAX_SYMBOLS * 2)
        self.assertEqual(trace[MAX_SYMBOLS]["frame"], 1)

    def test_long_duration_skips_encoder_frames(self):
        trace = []
        ids = greedy_tdt(_enc(5), 5, _Decoder(), _Joint([(11, 4), (BLANK, 1)]), trace)
        self.assertEqual(ids, [11]); self.assertEqual([row["frame"] for row in trace], [0, 4])

    def test_final_frame_token_is_retained(self):
        ids = greedy_tdt(_enc(1), 1, _Decoder(), _Joint([(12, 1)]))
        self.assertEqual(ids, [12])

    def test_duration_bins_match_model_metadata(self):
        self.assertEqual(DURATIONS, (0, 1, 2, 3, 4))


if __name__ == "__main__": unittest.main()
