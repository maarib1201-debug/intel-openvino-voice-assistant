"""OpenVINO Parakeet TDT decode (encoder on NPU/GPU when available)."""
from __future__ import annotations

import time

import numpy as np

BLANK = 1024
DURATIONS = (0, 1, 2, 3, 4)
N_DUR = len(DURATIONS)
MAX_SYMBOLS = 10
MEL_T = 1501


def detok(ids: list[int], vocab: dict[int, str]) -> str:
    pieces = [vocab.get(i, "") for i in ids if i != BLANK]
    text = "".join(pieces).replace("▁", " ").strip()
    return " ".join(text.split())


def greedy_tdt(enc: np.ndarray, enc_len: int, dec, joint, trace: list[dict] | None = None) -> list[int]:
    tokens: list[int] = []
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    last = np.array([[BLANK]], dtype=np.int32)
    t = 0
    while t < enc_len:
        start = t
        added = 0
        while added < MAX_SYMBOLS:
            d = dec({"targets": last, "h_in": h, "c_in": c})
            pred = d["decoder_output"]
            h2, c2 = d["h_out"], d["c_out"]
            frame = enc[:, :, t : t + 1].transpose(0, 2, 1)
            logits = joint({"encoder_outputs": frame, "decoder_outputs": pred})["logits"][0, 0, 0]
            token = int(np.argmax(logits[:-N_DUR]))
            duration_idx = int(np.argmax(logits[-N_DUR:]))
            skip = DURATIONS[duration_idx]
            if trace is not None:
                trace.append({"frame": t, "token": token, "duration_index": duration_idx,
                              "duration": skip, "blank": token == BLANK})
            if token != BLANK:
                tokens.append(token)
                last = np.array([[token]], dtype=np.int32)
                h, c = h2, c2
            t += skip
            added += 1
            if skip > 0:
                break
        if t <= start:
            t = start + 1
    return tokens


def transcribe_ov(nets: dict, pcm: np.ndarray, vocab: dict[int, str], trace: list[dict] | None = None) -> tuple[str, dict]:
    pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
    audio_len = np.array([len(pcm)], dtype=np.int32)
    t0 = time.perf_counter()
    mel_out = nets["mel"]({"input_signals": pcm[None, :], "input_length": audio_len})
    mel = list(mel_out.values())[0]
    if mel.ndim == 3 and mel.shape[2] == 128:
        mel = np.transpose(mel, (0, 2, 1))
    t = mel.shape[-1]
    padded = np.zeros((1, 128, MEL_T), dtype=np.float32)
    use = min(t, MEL_T)
    padded[:, :, :use] = mel[:, :, :use]
    mel_len = np.array([use], dtype=np.int32)
    enc_out = nets["encoder"]({"melspectogram": padded, "melspectogram_length": mel_len})
    enc = enc_out["encoder_output"]
    enc_len = int(np.array(enc_out["encoder_output_length"]).reshape(-1)[0])
    enc_len = max(1, min(enc_len, enc.shape[-1]))
    ids = greedy_tdt(enc, enc_len, nets["decoder"], nets["joint"], trace=trace)
    text = detok(ids, vocab)
    return text, {"total_s": time.perf_counter() - t0, "enc_frames": enc_len}
