from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("V2V_MODEL_DIR", ROOT / "models")).expanduser().resolve()

# Speaking speed vs Pocket TTS output. 1.0 = native. 1.10 stays ahead of
# playback on this machine more reliably than the previous 1.15 default.
TTS_SPEED = 1.10


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

QWEN35_GGUF_Q5 = Path(os.environ.get("V2V_QWEN_GGUF", MODELS / "Qwen3.5-0.8B-Q5_K_M.gguf")).expanduser()

LLM_IDS = {
    # Default: latest small chat model. 350M LFM is too weak for open conversation.
    "qwen08": "Qwen/Qwen3.5-0.8B",
    "lfm350": "LiquidAI/LFM2.5-350M",
    "gemma270": "unsloth/gemma-3-270m-it",
    "qwen08-agent": os.environ.get("V2V_QWEN08", "Qwen/Qwen3.5-0.8B"),
}

SYSTEM_PROMPT = (
    "You are a fast, natural voice assistant. Answer the user's actual question immediately, "
    "without filler or announcing the answer. Use concise spoken prose: one or two sentences for "
    "a simple fact, and about 60 to 90 words for an explanation. Explain the core real mechanism "
    "using only details you are confident are correct. Prefer a short, incomplete but correct answer "
    "to a longer speculative one. If an important fact or mechanism is uncertain, say plainly that "
    "you are not sure; do not guess or invent missing steps. In a mechanism, keep straight what sends, "
    "receives, measures, or calculates, and the direction of information or energy; omit any uncertain "
    "part. Treat unfamiliar names or premises as uncertain instead of supplying dates, people, places, "
    "or events. Avoid unnecessary analogies, absolute claims, equations, and extra detail. Use the "
    "previous conversation to resolve follow-ups. "
    "Write no markdown, lists, headings, emojis, stage directions, or thinking aloud. If speech "
    "recognition is slightly wrong but the intent is obvious, answer the intended question. Never "
    "claim you used a tool, changed something, or know live information unless that actually happened."
)



@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8088
    llm: str = "qwen08"
    llm_revision: str = field(default_factory=lambda: os.environ.get("V2V_QWEN_REVISION", "2fc06364715b967f1860aea9cf38778875588b17"))
    voice: str = "alba"
    cpu_threads: int = 4
    asr_threads: int = 4
    asr_device: str = field(default_factory=lambda: os.environ.get("V2V_ASR_DEVICE", "AUTO").strip().upper() or "AUTO")
    llm_n_gpu_layers: int = field(default_factory=lambda: _env_int("V2V_LLM_N_GPU_LAYERS", -1))
    llm_n_ctx: int = field(default_factory=lambda: _env_int("V2V_LLM_N_CTX", 1024))
    llm_n_batch: int = field(default_factory=lambda: _env_int("V2V_LLM_N_BATCH", 256))
    llm_n_ubatch: int = field(default_factory=lambda: _env_int("V2V_LLM_N_UBATCH", 256))
    llm_threads: int = field(default_factory=lambda: _env_int("V2V_LLM_THREADS", 0))
    llm_flash_attn: bool = field(default_factory=lambda: _env_bool("V2V_LLM_FLASH_ATTN", True))
    llm_cpu_fallback: bool = field(default_factory=lambda: _env_bool("V2V_LLM_CPU_FALLBACK", True))
    llm_temperature: float = field(
        default_factory=lambda: _env_float("V2V_LLM_TEMPERATURE", 0.3, 0.0, 2.0)
    )
    llm_top_p: float = field(default_factory=lambda: _env_float("V2V_LLM_TOP_P", 0.8, 0.05, 1.0))
    llm_top_k: int = field(default_factory=lambda: max(0, min(200, _env_int("V2V_LLM_TOP_K", 20))))
    llm_repeat_penalty: float = field(
        default_factory=lambda: _env_float("V2V_LLM_REPEAT_PENALTY", 1.05, 0.8, 1.5)
    )
    factual_guard_enabled: bool = field(
        default_factory=lambda: _env_bool("V2V_FACTUAL_GUARD_ENABLED", True)
    )
    factual_guard_specificity_score: int = field(
        default_factory=lambda: max(1, min(6, _env_int("V2V_FACTUAL_GUARD_SPECIFICITY_SCORE", 2)))
    )
    factual_guard_causal_score: int = field(
        default_factory=lambda: max(2, min(8, _env_int("V2V_FACTUAL_GUARD_CAUSAL_SCORE", 3)))
    )
    allow_lan: bool = field(default_factory=lambda: _env_bool("V2V_ALLOW_LAN", False))
    auth_token: str | None = field(
        default_factory=lambda: (os.environ.get("V2V_AUTH_TOKEN") or "").strip() or None
    )
    allowed_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in (os.environ.get("V2V_ALLOWED_ORIGINS") or "*").split(",") if o.strip()]
    )
    max_ws_text_bytes: int = field(default_factory=lambda: _env_int("V2V_MAX_WS_TEXT_BYTES", 65536))
    max_pcm_chunk_bytes: int = field(default_factory=lambda: _env_int("V2V_MAX_PCM_CHUNK_BYTES", 262144))
    max_ws_messages_per_second: int = field(
        default_factory=lambda: max(10, _env_int("V2V_MAX_WS_MESSAGES_PER_SECOND", 120))
    )
    log_sensitive: bool = field(default_factory=lambda: _env_bool("V2V_LOG_SENSITIVE", False))
    soak_diagnostics: bool = field(default_factory=lambda: _env_bool("V2V_SOAK_DIAGNOSTICS", False))
    trust_remote_code: bool = field(default_factory=lambda: _env_bool("V2V_TRUST_REMOTE_CODE", False))
    tts_queue_size: int = field(default_factory=lambda: max(4, _env_int("V2V_TTS_QUEUE_SIZE", 32)))
    qwen_queue_depth: int = field(default_factory=lambda: max(1, _env_int("V2V_QWEN_QUEUE_DEPTH", 4)))
    stt_queue_depth: int = field(default_factory=lambda: max(1, _env_int("V2V_STT_QUEUE_DEPTH", 4)))
    tts_gate_depth: int = field(default_factory=lambda: max(1, _env_int("V2V_TTS_GATE_DEPTH", 4)))
    needle_queue_depth: int = field(default_factory=lambda: max(1, _env_int("V2V_NEEDLE_QUEUE_DEPTH", 4)))
    shutdown_timeout_s: float = field(default_factory=lambda: _env_float("V2V_SHUTDOWN_TIMEOUT_S", 2.0, 0.2, 10.0))
    max_new_tokens: int = field(
        default_factory=lambda: max(16, min(512, _env_int("V2V_MAX_NEW_TOKENS", 128)))
    )
    quantize_tts: bool = True
    asr_decoding: str = "greedy_search"
    # Pocket TTS has no native rate control. Client plays at this speed (1.0–1.2).
    tts_speed: float = field(
        default_factory=lambda: _env_float("V2V_TTS_SPEED", TTS_SPEED, 0.9, 1.25)
    )
    knowledge_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("V2V_KNOWLEDGE_DIR") or (ROOT / "company_knowledge"))
    )
    rag_top_k: int = 3
    rag_min_score: float = 0.12

    # Tool calling. Location stays unset by default so weather never guesses.
    default_location: str | None = field(default_factory=lambda: (os.environ.get("V2V_DEFAULT_LOCATION") or "").strip() or None)
    weather_timeout_s: float = field(
        default_factory=lambda: _env_float("V2V_WEATHER_TIMEOUT_S", 6.0, 1.0, 15.0)
    )
    tool_ack_delay_s: float = 0.30
    tool_workers: int = 4
    router_max_tokens: int = 72

    # Listen gate (heuristic fallback only). Speech vs noise is Silero VAD.
    # Previous too-strict: rms=0.032, vad=0.70, hold=0.48s, floor*5.5
    vad_threshold: float = 0.55
    speech_start_rms: float = 0.025
    speech_hold_rms: float = 0.010
    speech_start_s: float = 0.32
    noise_cal_s: float = 1.0
    noise_floor_mult: float = 3.8
    min_silence_s: float = 0.65
    min_speech_s: float = 0.55
    max_speech_s: float = 12.0
    barge_min_speech_s: float = 0.45
    # Frames during this short echo-tail guard are retained in pre-roll rather
    # than discarded, so speech beginning immediately after playback is intact.
    post_tts_ignore_s: float = field(
        default_factory=lambda: _env_float("V2V_POST_TTS_GUARD_S", 0.16, 0.0, 0.5)
    )
    # Barge-in is a SEPARATE, easier detector vs speaker echo (not listen_rms).
    # Old barge used max(listen_on, 0.038, echo*2.1) so it could never fire.
    barge_holdoff_s: float = 0.08
    barge_confirm_s: float = field(
        default_factory=lambda: _env_float("V2V_BARGE_CONFIRM_S", 0.20, 0.12, 0.4)
    )
    barge_rms: float = 0.026
    barge_echo_ratio: float = 1.70
    barge_prob: float = 0.45
    asr_preroll_s: float = 0.45
    asr_barge_preroll_s: float = 0.45
    asr_tail_s: float = 0.22

    partial_every_s: float = 0.55

    parakeet_dir: Path = field(default_factory=lambda: MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
    vad_path: Path = field(default_factory=lambda: MODELS / "silero_vad.onnx")
    gguf_path: Path = field(default_factory=lambda: QWEN35_GGUF_Q5)

    @property
    def llm_id(self) -> str:
        if self.llm in LLM_IDS:
            return LLM_IDS[self.llm]
        return self.llm
