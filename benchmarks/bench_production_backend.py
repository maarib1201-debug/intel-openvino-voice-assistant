"""Production backend comparison: Verified Arc Vulkan vs CPU fallback.

Measures TTFT, prefill tok/s, decode tok/s, and full generation time on a production prompt.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "benchmarks" / "backend_comparison_results.json"
USER_PROMPT = "Explain why the sky is blue and how Rayleigh scattering works."
RUNS = 5
WARMUP = 1


def count_tokens(llama, text: str) -> int:
    try:
        return len(llama.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return 0


def run_benchmark_for_backend(llm_n_gpu_layers: int, label: str) -> dict:
    from v2v.config import Settings
    from v2v.llm import TinyLLM

    cfg = Settings(
        llm="qwen08",
        llm_n_gpu_layers=llm_n_gpu_layers,
        llm_n_ctx=1024,
        llm_n_batch=256,
        llm_n_ubatch=256,
        llm_threads=4,
        llm_flash_attn=True,
        llm_cpu_fallback=False if llm_n_gpu_layers != 0 else True,
        max_new_tokens=160,
    )

    print(f"\n=== Initializing backend: {label} (n_gpu_layers={llm_n_gpu_layers}) ===", flush=True)
    m = TinyLLM(cfg)
    print(
        f"Backend verified: actual={m.actual_backend} gpu_verified={m.gpu_verified} "
        f"device={m.gpu_device} layers_offloaded={m.layers_offloaded}",
        flush=True,
    )

    # Warmup
    print("Running warmup...", flush=True)
    for _ in range(WARMUP):
        list(m.stream("Hello", threading.Event()))

    ttft_list = []
    pe_list = []
    gen_time_list = []
    decode_tps_list = []
    tokens_list = []

    print(f"Running {RUNS} benchmark iterations...", flush=True)
    for i in range(RUNS):
        cancel = threading.Event()
        t0 = time.perf_counter()
        first_token_time = None
        parts = []

        for piece in m.stream(USER_PROMPT, cancel):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            parts.append(piece)

        t_end = time.perf_counter()
        full_text = "".join(parts)
        num_tokens = count_tokens(m.llama, full_text)
        ttft = (first_token_time - t0) if first_token_time else 0.0
        pe = float(getattr(m, "t_prompt_eval", 0) or 0)
        total_time = t_end - t0
        decode_time = (t_end - first_token_time) if first_token_time else 0.0
        decode_tps = (num_tokens / decode_time) if decode_time > 0 and num_tokens > 0 else 0.0

        ttft_list.append(ttft)
        pe_list.append(pe)
        gen_time_list.append(total_time)
        decode_tps_list.append(decode_tps)
        tokens_list.append(num_tokens)

        print(
            f"  [{i+1}/{RUNS}] TTFT: {ttft*1000:.1f}ms | PromptEval: {pe*1000:.1f}ms | "
            f"Total: {total_time:.2f}s | Decode: {decode_tps:.1f} tok/s | Tokens: {num_tokens}",
            flush=True,
        )

    def stats(xs: list[float]) -> dict:
        s = sorted(xs)
        return {
            "p50": round(s[len(s) // 2], 4),
            "p95": round(s[int(len(s) * 0.95)], 4) if len(s) > 1 else round(s[0], 4),
            "mean": round(sum(xs) / len(xs), 4),
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
        }

    return {
        "label": label,
        "actual_backend": m.actual_backend,
        "gpu_verified": m.gpu_verified,
        "gpu_device": m.gpu_device,
        "layers_offloaded": m.layers_offloaded,
        "ttft_s": stats(ttft_list),
        "prompt_eval_s": stats(pe_list),
        "generation_time_s": stats(gen_time_list),
        "decode_tps": stats(decode_tps_list),
        "tokens": stats([float(t) for t in tokens_list]),
    }


def main() -> None:
    print("=== Production Backend Comparison: Verified Vulkan vs CPU ===")
    vulkan_results = run_benchmark_for_backend(-1, "Verified Arc Vulkan")
    cpu_results = run_benchmark_for_backend(0, "CPU Baseline")

    summary = {
        "vulkan": vulkan_results,
        "cpu": cpu_results,
        "comparison": {
            "ttft_speedup": round(cpu_results["ttft_s"]["p50"] / vulkan_results["ttft_s"]["p50"], 2)
            if vulkan_results["ttft_s"]["p50"] > 0 else None,
            "vulkan_ttft_p50_ms": round(vulkan_results["ttft_s"]["p50"] * 1000, 1),
            "cpu_ttft_p50_ms": round(cpu_results["ttft_s"]["p50"] * 1000, 1),
            "vulkan_decode_tps_p50": vulkan_results["decode_tps"]["p50"],
            "cpu_decode_tps_p50": cpu_results["decode_tps"]["p50"],
        }
    }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults written to {OUT}")
    print("\nSummary Comparison:")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
