# Verified local benchmark evidence

These figures were summarized from historical local JSON/JSONL artifacts after excluding their machine paths, transcripts, and audio references. Raw logs are not included. The exact local run environment and warmup state varied between runs; treat these as historical observations, not reproducibility claims.

## Parakeet TDT 0.6B v2, OpenVINO NPU

- Ten labeled utterances recorded `device=ov-npu`.
- Decoder latency: median **115 ms**, mean **145 ms** (one first-run observation was about 423 ms; the other values were about 104–123 ms).
- Arithmetic mean of per-utterance WER: **0.062**; six of ten utterances had zero WER.
- The user's earlier **0.067 WER** is not the exact value in this result file. It is close but should not be repeated as an independently verified score without rerunning the published manifest and audio.

## Qwen3.5-0.8B GGUF, llama.cpp

A five-run backend comparison after one warmup measured:

| Backend | TTFT p50 | TTFT p95 | Decode p50 |
| --- | ---: | ---: | ---: |
| Intel Arc 140V Vulkan, 25/25 layers verified | 218 ms | 219 ms | 40.9 tokens/s |
| CPU | 858 ms | 903 ms | 47.8 tokens/s |

This does not support the earlier **90 ms LLM TTFT** claim. TTFT here means time until the first nonempty model stream chunk in the backend benchmark, not first spoken audio.

## Full voice path

The saved end-to-end benchmark contains four completed Vulkan + NPU iterations. TTFA (speech endpoint to first generated audio) had a median of **512 ms**; full measured turn duration had a median of **3.18 s**. The original capture files and transcript-bearing rows were excluded from this repository.

## Reproduce

Install the optional audio fixtures locally, provide a manifest to `evaluate_stt.py`, and run the model-specific scripts described in the root README. Benchmark outputs can contain transcripts and local paths; keep generated files out of commits. The historical repository snapshot did not include the full original benchmark corpus or conversion environment, so these aggregates cannot yet be independently reproduced from a clean checkout.
