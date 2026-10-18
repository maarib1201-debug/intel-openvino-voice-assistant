"""Run Parakeet STT against labeled clean/noisy fixtures and report WER/CER."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from v2v.asr import ParakeetASR
from v2v.config import Settings
from v2v.stt_correct import apply_stt_vocab
import numpy as np
import wave


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def distance(left: list[str] | list[str], right: list[str] | list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError(f"Expected mono 16-bit 16 kHz WAV: {path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/stt_ground_truth.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/stt_results.jsonl"))
    args = parser.parse_args()
    asr = ParakeetASR(Settings(asr_device="NPU"))
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        audio = load_audio(Path(case["audio"]))
        started = __import__("time").perf_counter()
        raw_transcript, stats = asr.decode(audio)
        transcript, corrections = apply_stt_vocab(raw_transcript)
        elapsed = __import__("time").perf_counter() - started
        ref_words = words(case["reference"])
        hyp_words = words(transcript)
        ref_chars = list(re.sub(r"[^a-z0-9]", "", case["reference"].lower()))
        hyp_chars = list(re.sub(r"[^a-z0-9]", "", transcript.lower()))
        rows.append(
            {
                **case,
                "raw_transcript": raw_transcript,
                "transcript": transcript,
                "corrections": [
                    {
                        "term": correction.term,
                        "original_span": correction.original_span,
                        "similarity": correction.similarity,
                        "source": correction.source,
                    }
                    for correction in corrections
                ],
                "device": asr.device,
                "wer": distance(ref_words, hyp_words) / max(1, len(ref_words)),
                "cer": distance(ref_chars, hyp_chars) / max(1, len(ref_chars)),
                "decode_s": stats.get("decode_s"),
                "wall_s": elapsed,
            }
        )
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
