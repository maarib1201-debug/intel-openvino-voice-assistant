from __future__ import annotations

import os
import copy
import threading
import time
from types import MethodType
from typing import Iterator

import numpy as np
import torch

os.environ.setdefault("POCKET_TTS_NO_BEARTYPE", "1")

from pocket_tts import TTSModel  # noqa: E402

from .config import Settings


@torch.no_grad()
def _stable_short_stream(
    model, model_state: dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
):
    """Pocket TTS short-text streamer without per-utterance native thread pools.

    Pocket TTS 1.x creates two new Python/Torch worker threads per phrase. On
    Windows their native Torch pools remain resident after join, causing thread
    growth under session churn. TTS is already serialized by ModelGate, so
    interleaving latent generation and Mimi decoding on that persistent worker
    preserves chunk streaming while bounding process threads.
    """
    from pocket_tts.modules.stateful_module import increment_steps, init_states

    if copy_state:
        model_state = copy.deepcopy(model_state)
    prepared = model.flow_lm.conditioner.prepare(text_to_generate)
    token_count = prepared.tokens.shape[1]
    max_gen_len = model._estimate_max_gen_len(token_count)
    mimi_steps = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
    mimi_state = init_states(model.mimi, batch_size=1, sequence_length=max_gen_len * mimi_steps)

    current_end = model._flow_lm_current_end(model_state)
    model._expand_kv_cache(model_state, sequence_length=current_end + token_count + max_gen_len)
    model._run_flow_lm_and_increment_step(model_state=model_state, text_tokens=prepared.tokens)

    backbone = torch.full(
        (1, 1, model.flow_lm.ldim),
        fill_value=float("NaN"),
        device=next(iter(model.flow_lm.parameters())).device,
        dtype=model.flow_lm.dtype,
    )
    eos_step = None
    for generation_step in range(max_gen_len):
        latent, is_eos = model._run_flow_lm_and_increment_step(
            model_state=model_state, backbone_input_latents=backbone
        )
        if is_eos.item() and eos_step is None:
            eos_step = generation_step
        if eos_step is not None and generation_step >= eos_step + frames_after_eos:
            break
        mimi_input = latent * model.flow_lm.emb_std + model.flow_lm.emb_mean
        audio = model.mimi.decode_from_latent(mimi_input, mimi_state)
        increment_steps(model.mimi, mimi_state, increment=mimi_steps)
        backbone = latent
        yield audio[0, 0]


def _install_stable_streamer(model) -> bool:
    required = (
        "_generate_audio_stream_short_text",
        "_estimate_max_gen_len",
        "_flow_lm_current_end",
        "_expand_kv_cache",
        "_run_flow_lm_and_increment_step",
        "flow_lm",
        "mimi",
    )
    if not all(hasattr(model, name) for name in required):
        return False
    model._generate_audio_stream_short_text = MethodType(_stable_short_stream, model)
    return True


class PocketTTS:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        t0 = time.perf_counter()
        kwargs = dict(language="english", sampler_decode_steps=1, temp=0.35)
        self.model = None
        if cfg.quantize_tts:
            try:
                self.model = TTSModel.load_model(quantize=True, **kwargs)
            except Exception as exc:
                print(f"  TTS quantize=True failed ({exc}); using fp32", flush=True)
        if self.model is None:
            self.model = TTSModel.load_model(quantize=False, **kwargs)
        self.model.eval()
        self.stable_streaming = _install_stable_streamer(self.model)
        print(
            f"  TTS stable_streaming={str(self.stable_streaming).lower()} "
            f"(persistent gate worker; no per-phrase Torch workers)",
            flush=True,
        )
        self.load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            self.voice = self.model.get_state_for_audio_prompt(cfg.voice)
        except Exception:
            self.voice = self.model.get_state_for_audio_prompt(
                "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
            )
        self.voice_s = time.perf_counter() - t1
        self.sr = int(self.model.sample_rate)
        self.lock = threading.Lock()
        print(f"  TTS_SPEED={cfg.tts_speed:.2f} (client playback; Pocket TTS has no native rate)", flush=True)

    def warmup(self) -> None:
        list(self.stream("Ready.", threading.Event()))

    def stream(self, text: str, cancel: threading.Event) -> Iterator[tuple[np.ndarray, dict]]:
        text = (text or "").strip()
        if not text:
            return
        words = len(text.split())
        logged_text = text if self.cfg.log_sensitive else "<redacted>"
        print(f"TTS send chars={len(text)} words={words} text={logged_text!r}", flush=True)
        t0 = time.perf_counter()
        first = None
        n = 0
        samples = 0
        cancelled = False
        with self.lock:
            # Pocket TTS writes KV-cache in-place; InferenceMode forbids that.
            try:
                gen = self.model.generate_audio_stream(self.voice, text, copy_state=True)
            except TypeError:
                gen = self.model.generate_audio_stream(self.voice, text)
            for chunk in gen:
                if cancel.is_set():
                    cancelled = True
                    print(
                        f"TTS cancel during gen after {n} chunks "
                        f"audio_s={samples / self.sr:.2f} reason=cancel-event",
                        flush=True,
                    )
                    break
                wav = _to_f32(chunk)
                if wav.size == 0:
                    continue
                n += 1
                samples += int(wav.size)
                if first is None:
                    first = time.perf_counter() - t0
                yield wav, {"chunk": n, "ttfa_s": first, "done": False}
        dt = time.perf_counter() - t0
        audio_s = samples / self.sr if self.sr else 0.0
        print(
            f"TTS generated audio_s={audio_s:.2f} chunks={n} gen_s={dt:.2f} cancelled={cancelled}",
            flush=True,
        )
        yield np.zeros(0, dtype=np.float32), {
            "chunk": n,
            "ttfa_s": first or dt,
            "done": True,
            "gen_s": dt,
            "audio_s": audio_s,
            "rtf": (dt / audio_s) if audio_s else 0.0,
            "cancelled": cancelled,
        }


def _to_f32(chunk) -> np.ndarray:
    if hasattr(chunk, "detach"):
        wav = chunk.detach().cpu().numpy()
    else:
        wav = np.asarray(chunk)
    wav = np.squeeze(wav).astype(np.float32)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    np.clip(wav, -1.0, 1.0, out=wav)
    return wav


def f32_to_i16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
