from __future__ import annotations

import json
import queue
import struct
import threading
import time
import traceback
import uuid
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .asr import SR, EnergyVad, ParakeetASR, is_junk
from .config import Settings
from .factual_guard import FactualRiskGuard
from .llm import TinyLLM, _clean_reply
from .rag import CompanyKnowledge, Hit, format_knowledge_block
from .sentences import SentenceBuffer
from .stt_correct import apply_stt_vocab, normalize_application_text
from .tools import (
    ToolError,
    acknowledgement_for,
    execute_tool,
    format_tool_result,
    requires_synthesis,
    validate_tool_arguments,
)
from .tts import PocketTTS, f32_to_i16
from .workers import BusyError, ModelGate


Emit = Callable[[dict[str, Any] | tuple[str, bytes]], None]


def _safe_text(cfg: Settings, text: str) -> str:
    return text if cfg.log_sensitive else "<redacted>"


class _GenerationCancel:
    """Cancellation view that cannot miss the shared Event's brief pulse."""

    def __init__(self, session: "Session", generation_id: int):
        self.session = session
        self.generation_id = generation_id

    def is_set(self) -> bool:
        return (
            self.session._stop.is_set()
            or self.session.cancel.is_set()
            or self.generation_id != self.session.gen
        )


@dataclass
class Models:
    asr: ParakeetASR
    llm: TinyLLM
    tts: PocketTTS
    cfg: Settings
    load: dict[str, float]
    knowledge: CompanyKnowledge | None = None
    qwen_gate: ModelGate | None = None
    stt_gate: ModelGate | None = None
    tts_gate: ModelGate | None = None
    needle_gate: ModelGate | None = None


def health_snapshot(models: Models | None) -> dict[str, Any]:
    names = ("stt", "npu", "qwen", "vulkan", "needle", "tts", "rag", "tool_providers")
    if models is None:
        return {"status": "starting", "components": {name: {"status": "starting"} for name in names}}
    asr_device = str(models.asr.device)
    llm = models.llm
    cfg = getattr(models, "cfg", None)
    knowledge = getattr(models, "knowledge", None)
    needle_available = _module_available("needle")
    gpu_verified = bool(getattr(llm, "gpu_verified", False))
    gpu_offload = bool(getattr(llm, "gpu_offload", False))
    actual_backend = getattr(llm, "actual_backend", llm.backend)
    components = {
        "stt": {"status": "ok", "device": asr_device},
        "npu": {
            "status": "ok" if asr_device == "ov-npu" else "degraded",
            "available": asr_device == "ov-npu",
            "device": asr_device,
            "requested": getattr(cfg, "asr_device", "AUTO"),
        },
        "qwen": {
            "status": "ok" if llm.backend in {"llama", "hf"} else "degraded",
            "backend": llm.backend,
            "requested_backend": getattr(llm, "backend_requested", "cpu"),
            "actual_backend": actual_backend,
            "gpu_verified": gpu_verified,
        },
        "vulkan": {
            "status": "ok" if gpu_verified else (
                "degraded"
                if getattr(llm, "backend_requested", "cpu") == "vulkan" or bool(getattr(llm, "fallback_reason", ""))
                else "disabled"
            ),
            "available": gpu_offload,
            "gpu_verified": gpu_verified,
            "gpu_device": getattr(llm, "gpu_device", None),
            "layers_offloaded": getattr(llm, "layers_offloaded", 0),
            "fallback_reason": llm.fallback_reason or None,
        },
        "needle": {"status": "ok" if needle_available else "disabled", "available": needle_available},
        "tts": {"status": "ok" if models.tts.model is not None else "degraded", "sample_rate": models.tts.sr},
        "rag": {
            "status": "ok" if knowledge is not None else "disabled",
            "chunk_count": len(knowledge.chunks) if knowledge else 0,
            "directory": str(getattr(cfg, "knowledge_dir", "")),
        },
        "tool_providers": {"status": "ok", "weather": "configured", "calculator": "local", "time": "local"},
    }
    queues = {}
    for key in ("qwen_gate", "stt_gate", "tts_gate", "needle_gate"):
        gate = getattr(models, key, None)
        if gate is not None:
            queues[gate.name] = {"depth": gate.depth, "occupancy": len(gate.occupancy()), "queued": gate.depth_now()}
    return {
        "status": "degraded" if any(item["status"] not in {"ok", "disabled"} for item in components.values()) else "ok",
        "components": components,
        "queues": queues,
    }


def _module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def load_models(cfg: Settings) -> Models:
    n = max(1, cfg.cpu_threads)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print("loading Parakeet...", flush=True)
    asr = ParakeetASR(cfg)
    print(f"  ASR {asr.load_s:.1f}s device={asr.device}", flush=True)
    print(f"loading LLM {cfg.llm} -> {cfg.llm_id} ...", flush=True)
    llm = TinyLLM(cfg)
    print(f"  LLM {llm.load_s:.1f}s backend={llm.backend}", flush=True)
    print("loading Pocket TTS...", flush=True)
    tts = PocketTTS(cfg)
    print(f"  TTS {tts.load_s:.1f}s  voice {tts.voice_s:.1f}s  sr={tts.sr}", flush=True)
    print("warming up (dummy STT/LLM/TTS)...", flush=True)
    t0 = time.perf_counter()
    asr.warmup()
    tts.warmup()
    llm.warmup()
    print(f"  warmup {time.perf_counter()-t0:.1f}s", flush=True)
    knowledge = CompanyKnowledge.from_dir(cfg.knowledge_dir)
    print(
        f"  RAG chunks={len(knowledge.chunks)} dir={cfg.knowledge_dir} load_s={knowledge.load_s:.3f}",
        flush=True,
    )
    return Models(
        asr=asr,
        llm=llm,
        tts=tts,
        cfg=cfg,
        load={"asr": asr.load_s, "llm": llm.load_s, "tts": tts.load_s, "voice": tts.voice_s, "rag": knowledge.load_s},
        knowledge=knowledge,
        qwen_gate=ModelGate("qwen", depth=cfg.qwen_queue_depth),
        stt_gate=ModelGate("stt", depth=cfg.stt_queue_depth),
        tts_gate=ModelGate("tts", depth=cfg.tts_gate_depth),
        needle_gate=ModelGate("needle", depth=cfg.needle_queue_depth),
    )


class Session:
    """One duplex conversation. Models are shared; this holds VAD / history / cancel."""

    def __init__(self, models: Models, emit: Emit, session_id: str | None = None):
        self.m = models
        self.emit = emit
        self.vad = EnergyVad(models.cfg)
        self.cancel = threading.Event()
        self.gen = 0
        self.playing = False
        self.in_flight = False
        self.busy_partial = False
        self.last_user = ""
        self._utt_seq = 0
        self.n_valid_transcripts = 0
        self.n_llm_requests = 0
        self.active_utt = 0
        self._asr_q: queue.Queue = queue.Queue(maxsize=16)
        self._llm_q: queue.Queue = queue.Queue(maxsize=16)
        self._tts_q: queue.Queue = queue.Queue(maxsize=models.cfg.tts_queue_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._playback_done = threading.Event()
        self._playback_done.set()
        tool_workers = max(1, int(models.cfg.tool_workers))
        self._tool_executor = ThreadPoolExecutor(
            max_workers=tool_workers,
            thread_name_prefix="v2v-tool",
        )
        self._tool_slots = threading.BoundedSemaphore(tool_workers * 2)
        self._tool_turns: dict[int, dict[str, Any]] = {}
        self._last_successful_tool: str | None = None
        self.session_id = session_id or uuid.uuid4().hex
        self.request_id: str | None = None
        self.history: list[dict[str, str]] = []
        self.last_qwen_wait_s = 0.0
        self.last_qwen_infer_s = 0.0
        self.last_stt_wait_s = 0.0
        self.last_tts_wait_s = 0.0
        self.closed = False
        self.playing_since = 0.0
        self.barge_armed = False
        self.listening_enabled = False
        self._soak_records: dict[int, dict[str, Any]] = {}
        self._soak_lock = threading.Lock()
        self._soak_path: Path | None = None
        if self.m.cfg.soak_diagnostics:
            soak_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "soak"
            soak_dir.mkdir(parents=True, exist_ok=True)
            self._soak_path = soak_dir / f"soak_{time.strftime('%Y%m%d_%H%M%S')}_{self.session_id[:8]}.jsonl"
        self._threads = [
            threading.Thread(target=self._asr_loop, daemon=True, name="v2v-asr"),
            threading.Thread(target=self._llm_loop, daemon=True, name="v2v-llm"),
            threading.Thread(target=self._tts_loop, daemon=True, name="v2v-tts"),
        ]
        for t in self._threads:
            t.start()
        print("listening_enabled=false", flush=True)
        self._ev("state", state="off")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        self.cancel.set()
        self._playback_done.set()
        with self._lock:
            turns = list(self._tool_turns.values())
            self._tool_turns.clear()
        for turn in turns:
            timer = turn.get("timer")
            if timer is not None:
                timer.cancel()
        self._cancel_session_gates()
        self._tool_executor.shutdown(wait=False, cancel_futures=True)
        for q in (self._asr_q, self._llm_q, self._tts_q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        deadline = time.perf_counter() + self.m.cfg.shutdown_timeout_s
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.perf_counter()))

    def set_listen(self, enabled: bool, reason: str = "") -> None:
        reason = reason or ("user-start" if enabled else "user-stop")
        self.listening_enabled = bool(enabled)
        print(f"listening_enabled={str(self.listening_enabled).lower()} reason={reason}", flush=True)
        if not self.listening_enabled:
            self.vad.reset()
            self._ev("state", state="off")
        else:
            self.vad.recalibrate()
            if not self.playing and not self.in_flight:
                self._ev("state", state="listening")

    def _resume_listen_ui(self, why: str) -> None:
        print(f"automatic resume attempt why={why} listening_enabled={str(self.listening_enabled).lower()}", flush=True)
        if self.listening_enabled:
            self._ev("state", state="listening")
        else:
            self._ev("state", state="off")

    def push_pcm16(self, raw: bytes) -> None:
        if not raw or not self.listening_enabled:
            return
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self.push_f32(pcm)

    def push_f32(self, pcm: np.ndarray) -> None:
        if not self.listening_enabled:
            return
        ev = self.vad.push(pcm)
        if self.playing:
            held = time.perf_counter() - self.playing_since
            if ev["barge"] and held >= self.m.cfg.barge_holdoff_s:
                candidate_at = ev.get("barge_candidate_at")
                cancel_latency_ms = (
                    (time.perf_counter() - float(candidate_at)) * 1000.0
                    if candidate_at is not None
                    else 0.0
                )
                preserved = bool(self.vad.speech and self.vad.utt)
                print(
                    f"barge_confirmed generation_id={self.gen} held={held:.2f}s "
                    f"barge_in_armed={self.barge_armed} "
                    f"barge_cancel_latency_ms={cancel_latency_ms:.1f} "
                    f"barge_utterance_preserved={str(preserved).lower()}",
                    flush=True,
                )
                self.interrupt(
                    "confirmed-user-speech",
                    barge_cancel_latency_ms=round(cancel_latency_ms, 1),
                    barge_utterance_preserved=preserved,
                )
                # playing is now False; fall through so a just-closed
                # utterance in this same frame can go to STT.
            else:
                # Echo stays out of STT until playback is cancelled; the
                # in-progress VAD utterance (with pre-roll) is kept.
                return
        reports = ev.get("segment_reports") or []
        for i, seg in enumerate(ev["segments"]):
            audio_debug = reports[i] if i < len(reports) else {}
            self._metric("speech_end")
            print(
                f"sending_to_stt samples={seg.size} audio_bytes={seg.nbytes} "
                f"duration={seg.size / SR:.2f}s sample_rate={SR} audio_format=pcm_f32 "
                f"barge_utterance_preserved={str(bool(audio_debug.get('barge_utterance'))).lower()}",
                flush=True,
            )
            self._submit_asr(seg, final=True, audio_debug=audio_debug)

    def playback_done(self, generation_id: Any = None) -> None:
        try:
            tagged = int(generation_id)
        except (TypeError, ValueError):
            tagged = self.gen
        if tagged != self.gen:
            print(
                f"playback_generation_complete ignored stale_generation={tagged} active={self.gen}",
                flush=True,
            )
            return
        print(f"playback_generation_complete generation_id={tagged} source=browser-last-source", flush=True)
        self.vad.set_audible_playback(False)
        self._playback_done.set()
        self._soak_finish(tagged, playback_complete=True, playback_completion_reason="last-source-ended")

    def playback_telemetry(self, event: str, generation_id: Any = None, data: dict | None = None) -> None:
        """Feed browser-audible state into echo handling without trusting text/audio content."""
        try:
            tagged = int(generation_id)
        except (TypeError, ValueError):
            tagged = self.gen
        if event == "playback_started_estimate" and tagged == self.gen and self.playing:
            self.vad.set_audible_playback(True)
            print(
                f"browser_playback_started generation_id={tagged} barge_echo_calibration=true",
                flush=True,
            )
        if event == "playback_summary" and tagged == self.gen:
            summary = data or {}
            self._soak_note(tagged, **{key: summary.get(key) for key in (
                "max_gap_ms", "starvation_count", "max_pending_buffer_ms", "max_scheduled_ahead_ms",
                "stale_chunk_discard_count", "pcm_boundary_count", "max_raw_boundary_delta",
                "max_smoothed_boundary_delta", "worst_boundary_chunk", "playback_completion_reason",
            ) if key in summary})

    def interrupt(self, reason: str = "", **details: Any) -> None:
        reason = reason or "unspecified"
        with self._lock:
            old = self._bump_generation_locked(reason)
            killed = self.active_utt
        self._ev(
            "interrupt",
            reason=reason,
            utterance_id=killed,
            generation_id=self.gen,
            cancelled_generation=old,
            **details,
        )
        self._soak_finish(old, barge_in=reason in {"confirmed-user-speech", "user-button"},
                          barge_cancel_latency_ms=details.get("barge_cancel_latency_ms"),
                          playback_complete=False, cancellation_reason=reason)
        self._resume_listen_ui("interrupt")

    def _bump_generation_locked(self, reason: str) -> int:
        """Invalidate generation N. N+1 is not cancelled. Caller holds self._lock."""
        old = self.gen
        self.gen += 1
        self.cancel.set()
        self.playing = False
        self.in_flight = False
        self.barge_armed = False
        # Do not ignore mic frames — the interrupting utterance must survive.
        self.vad.set_playing(False, ignore=False)
        # Do NOT zero loud_n / preroll here. A rapid second interruption may
        # already be in start-hold; wiping it is the no-STT overlap miss.
        n_tts = self._drain(self._tts_q)
        n_llm = self._drain(self._llm_q)
        self._cancel_session_gates()
        turn = self._tool_turns.pop(old, None)
        if turn is not None and turn.get("timer") is not None:
            turn["timer"].cancel()
        self._playback_done.set()
        self.cancel.clear()
        print(
            f"cancel_generation={old} new_generation={self.gen} reason={reason} "
            f"tts_queue_cleared generation={old} drained_llm={n_llm} drained_tts={n_tts} "
            f"mic_capture_continues=true barge_in_armed=false "
            f"(re-arms when next TTS starts) playing=false in_flight=false",
            flush=True,
        )
        return old

    def reset_history(self) -> None:
        self.history.clear()
        self.last_user = ""

    def _remember(self, user: str, bot: str) -> None:
        bot = _clean_reply(bot)
        if not bot:
            return
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": bot})
        del self.history[:-2]

    def _cancel_session_gates(self) -> None:
        for name in ("qwen_gate", "stt_gate", "tts_gate", "needle_gate"):
            gate = getattr(self.m, name, None)
            if gate is not None:
                gate.cancel_session(self.session_id)

    def _submit_asr(self, pcm: np.ndarray, final: bool, audio_debug: dict | None = None) -> None:
        job = {
            "pcm": pcm,
            "final": final,
            "t": time.perf_counter(),
            "audio_debug": audio_debug or {},
        }
        try:
            self._asr_q.put_nowait(job)
            print(f"stt_queued samples={pcm.size} final={final} qlen={self._asr_q.qsize()}", flush=True)
            if not final:
                self.busy_partial = True
        except queue.Full:
            print(f"stt_queue_full DROPPED samples={pcm.size} final={final}", flush=True)
            try:
                self._asr_q.put(job, timeout=1.0)
                print("stt_queued after-wait", flush=True)
            except Exception as exc:
                print(f"stt_queue_failed {exc}", flush=True)

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._asr_q.get()
            except Exception:
                traceback.print_exc()
                continue
            if job is None:
                return
            try:
                self._asr_one(job)
            except Exception:
                print("ASR loop crashed on job; recovering", flush=True)
                traceback.print_exc()
                self._resume_listen_ui("asr-loop-crash")

    def _asr_one(self, job: dict) -> None:
            pcm = job["pcm"]
            print(
                f"stt_request_started samples={pcm.size} duration={pcm.size / SR:.2f}s final={job['final']}",
                flush=True,
            )
            try:
                gate = getattr(self.m, "stt_gate", None)
                if gate is not None:
                    (text, st), gate_job = gate.run(
                        self.session_id,
                        lambda: self.m.asr.decode(pcm),
                        _GenerationCancel(self, self.gen),
                    )
                    self.last_stt_wait_s = gate_job.wait_s
                    print(
                        f"stt_queue_wait_ms={gate_job.wait_s * 1000.0:.1f} "
                        f"session_id={self.session_id}",
                        flush=True,
                    )
                else:
                    text, st = self.m.asr.decode(pcm)
                    self.last_stt_wait_s = 0.0
            except BusyError as exc:
                print(
                    f"stt_busy session_id={self.session_id} retry_after_ms={exc.retry_after_ms}",
                    flush=True,
                )
                self._ev("busy", retry_after_ms=exc.retry_after_ms, reason=exc.reason)
                self._resume_listen_ui("stt-busy")
                return
            except Exception as exc:
                print(f"stt_request_finished error={exc!r}", flush=True)
                traceback.print_exc()
                self._resume_listen_ui("stt-error")
                return
            print(
                f"stt_request_finished transcript={_safe_text(self.m.cfg, text)!r} audio_s={st.get('audio_s', 0):.2f} "
                f"rtf={st.get('rtf', 0):.3f} peak={st.get('peak', 0):.2f}",
                flush=True,
            )
            capture = {**(job.get("audio_debug") or {}), **st}
            print(
                "STT_CAPTURE_REPORT "
                + json.dumps(
                    {
                        "utterance_duration_s": round(float(capture.get("duration_s") or st.get("audio_s") or 0), 3),
                        "leading_silence_ms": round(float(capture.get("leading_silence_ms") or 0), 1),
                        "trailing_silence_ms": round(float(capture.get("trailing_silence_ms") or 0), 1),
                        "peak": round(float(capture.get("peak") or 0), 4),
                        "rms": round(float(capture.get("rms") or 0), 4),
                        "clipping_ratio": round(float(capture.get("clipping_ratio") or 0), 6),
                        "noise_floor_rms": round(float(capture.get("noise_floor_rms") or 0), 4),
                        "crest_factor": round(float(capture.get("crest_factor") or 0), 3),
                        "vad_speech_start_latency_ms": round(
                            float(capture.get("vad_speech_start_latency_ms") or 0), 1
                        ),
                        "barge_utterance": bool(capture.get("barge_utterance")),
                        "transcript": _safe_text(self.m.cfg, text),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not job["final"]:
                self.busy_partial = False
                if text:
                    self._ev("partial", text=text, rtf=round(st["rtf"], 3))
                return
            text = (text or "").strip()
            original_text = text
            corrected_text, vocab_hits = apply_stt_vocab(text)
            if vocab_hits:
                print(
                    f"stt_vocab_correction applied=true term={','.join(hit.term for hit in vocab_hits)} "
                    f"similarity={min(hit.similarity for hit in vocab_hits):.3f} "
                    f"original={_safe_text(self.m.cfg, original_text)!r} "
                    f"corrected={_safe_text(self.m.cfg, corrected_text)!r}",
                    flush=True,
                )
                text = corrected_text
            application_text = normalize_application_text(text)
            if application_text != text:
                print(f"stt_application_normalization applied=true raw={_safe_text(self.m.cfg, text)!r} normalized={_safe_text(self.m.cfg, application_text)!r}", flush=True)
                text = application_text
            self._utt_seq += 1
            uid = self._utt_seq
            self.request_id = uuid.uuid4().hex
            print(f"utterance_id={uid} transcript={_safe_text(self.m.cfg, text)!r}", flush=True)
            if not text:
                print(f'utterance_id={uid} dropped reason=empty-transcript transcript=""', flush=True)
                self._resume_listen_ui("empty-transcript")
                return
            # Non-empty Parakeet text is a valid utterance. Never silently
            # discard it because of stale playing/in_flight flags, short audio
            # padding, de-dup, or a full queue.
            if is_junk(text, float(st.get("audio_s") or 0)):
                print(
                    f"utterance_id={uid} note=is_junk-would-have-dropped "
                    f"audio_s={st.get('audio_s', 0):.2f} STILL sending_to_llm",
                    flush=True,
                )
            if text.lower() == self.last_user.lower():
                print(
                    f"utterance_id={uid} note=same-text-as-previous STILL sending_to_llm",
                    flush=True,
                )
            self.n_valid_transcripts += 1
            print(
                f"utterance_id={uid} transcript_accepted "
                f"valid_transcript_count={self.n_valid_transcripts} "
                f"playing={self.playing} in_flight={self.in_flight} gen={self.gen}",
                flush=True,
            )
            print(
                f"YOU  {_safe_text(self.m.cfg, text)!r}  audio={st.get('audio_s', 0):.2f}s  rtf={st.get('rtf', 0):.3f}  "
                f"peak={st.get('peak', 0):.2f}  asr={self.m.asr.device} utterance_id={uid}",
                flush=True,
            )
            t_stt_done = time.perf_counter()
            self._metric("stt_ready", utterance_id=uid)
            print(f"utterance_id={uid} stt_finished", flush=True)
            with self._lock:
                # Newest transcript always owns a new generation_id. Cancelling
                # N must not leave cancel set for N+1.
                if self.playing or self.in_flight:
                    print(
                        f"utterance_id={uid} replacing previous work "
                        f"playing={self.playing} in_flight={self.in_flight} "
                        f"prev_utterance_id={self.active_utt} prev_generation={self.gen}",
                        flush=True,
                    )
                    old = self._bump_generation_locked("new-user-utterance")
                    self._ev(
                        "interrupt",
                        reason="new-user-utterance",
                        utterance_id=uid,
                        generation_id=self.gen,
                        cancelled_generation=old,
                    )
                else:
                    self.gen += 1
                    self.cancel.clear()
                self.in_flight = True
                self.active_utt = uid
                gen = self.gen
                llm_job = {
                    "text": text,
                    "t": job["t"],
                    "t_stt_done": t_stt_done,
                    "t_llm_queued": time.perf_counter(),
                    "gen": gen,
                    "stt": st,
                    "utterance_id": uid,
                    "request_id": self.request_id,
                }
                queued = False
                try:
                    self._llm_q.put_nowait(llm_job)
                    queued = True
                except queue.Full:
                    try:
                        evicted = self._llm_q.get_nowait()
                        print(
                            f"dropped reason=llm-queue-full-evicted "
                            f"utterance_id={evicted.get('utterance_id')}",
                            flush=True,
                        )
                    except queue.Empty:
                        pass
                    try:
                        self._llm_q.put_nowait(llm_job)
                        queued = True
                    except queue.Full:
                        queued = False
            if not queued:
                print(
                    f"utterance_id={uid} dropped reason=llm-queue-full "
                    f"valid_transcript_count={self.n_valid_transcripts} "
                    f"llm_request_count={self.n_llm_requests}",
                    flush=True,
                )
                self.in_flight = False
                self._resume_listen_ui("llm-queue-full")
                return
            self.last_user = text
            self.n_llm_requests += 1
            print(
                f"utterance_id={uid} sending_to_llm llm_queued generation_id={gen} "
                f"llm_request_count={self.n_llm_requests} "
                f"valid_transcript_count={self.n_valid_transcripts}",
                flush=True,
            )
            self._ev("user", text=text, utterance_id=uid, generation_id=gen, **_round(st))
            self._soak_note(gen, utterance_id=uid, transcript=text, stt_latency_ms=round(float(st.get("decode_s") or 0) * 1000, 1),
                            stt_rtf=st.get("rtf"), listening_enabled=self.listening_enabled)
            self._ev("state", state="thinking", utterance_id=uid, generation_id=gen)

    def _llm_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._llm_q.get()
            except Exception:
                traceback.print_exc()
                continue
            if job is None:
                return
            try:
                if job.get("kind") == "tool_result":
                    self._tool_result_one(job)
                else:
                    self._llm_one(job)
            except Exception:
                print("LLM loop crashed on job; recovering", flush=True)
                traceback.print_exc()
                try:
                    self.in_flight = False
                    self._resume_listen_ui("llm-loop-crash")
                except Exception:
                    traceback.print_exc()

    def _llm_one(self, job: dict) -> None:
        uid = job.get("utterance_id")
        gen = job["gen"]
        t_deq = time.perf_counter()
        t_stt_done = float(job.get("t_stt_done") or job["t"])
        t_queued = float(job.get("t_llm_queued") or t_deq)
        print(f"utterance_id={uid} llm_dequeued generation_id={gen}", flush=True)
        print(
            f"utterance_id={uid} llm_request_started generation_id={gen} current_gen={self.gen} "
            f"cancel={self.cancel.is_set()} playing={self.playing} in_flight={self.in_flight} "
            f"stt_to_llm_start={t_deq - t_stt_done:.3f}s queue_wait={t_deq - t_queued:.3f}s",
            flush=True,
        )
        if gen != self.gen:
            print(
                f"utterance_id={uid} dropped reason=stale-gen job_gen={gen} current_gen={self.gen} "
                f"(cancel of N must not drop N+1)",
                flush=True,
            )
            return
        cancel = _GenerationCancel(self, gen)
        route_started = time.perf_counter()
        try:
            decision = self.m.llm.decide_tool_fast(
                job["text"],
                cancel,
                previous_tool=self._last_successful_tool,
                session_id=self.session_id,
            )
            if decision is None:
                needle_gate = getattr(self.m, "needle_gate", None)
                if needle_gate is not None:
                    decision, needle_job = needle_gate.run(
                        self.session_id,
                        lambda: self.m.llm.route_needle(
                            job["text"], cancel, session_id=self.session_id
                        ),
                        cancel,
                    )
                    print(
                        f"utterance_id={uid} generation_id={gen} session_id={self.session_id} "
                        f"needle_queue_wait_ms={needle_job.wait_s * 1000.0:.1f} "
                        f"needle_inference_ms={needle_job.infer_s * 1000.0:.1f}",
                        flush=True,
                    )
                    self._metric(
                        "needle_timing",
                        utterance_id=uid,
                        needle_queue_wait_ms=round(needle_job.wait_s * 1000.0, 1),
                        needle_inference_ms=round(needle_job.infer_s * 1000.0, 1),
                    )
                else:
                    decision = self.m.llm.route_needle(
                        job["text"], cancel, session_id=self.session_id
                    )
        except BusyError as exc:
            print(
                f"utterance_id={uid} generation_id={gen} session_id={self.session_id} "
                f"needle_busy retry_after_ms={exc.retry_after_ms}",
                flush=True,
            )
            self._ev(
                "busy",
                retry_after_ms=exc.retry_after_ms,
                reason=exc.reason,
                utterance_id=uid,
                generation_id=gen,
            )
            self.in_flight = False
            self._resume_listen_ui("needle-busy")
            return
        route_finished = time.perf_counter()
        route_ms = (route_finished - t_stt_done) * 1000.0
        decision_data = decision.model_dump()
        print(
            f"utterance_id={uid} generation_id={gen} tool_decision={json.dumps(decision_data, sort_keys=True)} "
            f"STT_end_to_tool_decision_ms={route_ms:.1f}",
            flush=True,
        )
        self._metric("routing_complete", utterance_id=uid, tool=decision.tool)
        if cancel.is_set():
            print(
                f"tool_result_discarded reason=stale_generation generation_id={gen} active={self.gen}",
                flush=True,
            )
            return
        if decision.tool is not None:
            arguments = dict(decision.arguments)
            if decision.tool == "get_weather" and not str(arguments.get("location") or "").strip():
                if self.m.cfg.default_location:
                    arguments["location"] = self.m.cfg.default_location
                else:
                    self._deliver_direct(job, "Which city should I check?")
                    return
            try:
                arguments = validate_tool_arguments(decision.tool, arguments)
            except ToolError as exc:
                print(
                    f"utterance_id={uid} generation_id={gen} tool_arguments_invalid "
                    f"tool_name={decision.tool} code={exc.code}",
                    flush=True,
                )
                # A semantically invalid router call is not executable. Preserve
                # the ordinary assistant path rather than crashing or guessing.
                self._stream_llm_answer(job)
                return
            self._start_tool(job, decision.tool, arguments, route_finished)
            return
        self._last_successful_tool = None
        knowledge = self.m.knowledge
        if knowledge is not None and knowledge.should_retrieve(job["text"]):
            t_rag = time.perf_counter()
            hits = knowledge.retrieve(
                job["text"],
                top_k=int(self.m.cfg.rag_top_k),
                min_score=float(self.m.cfg.rag_min_score),
            )
            rag_ms = (time.perf_counter() - t_rag) * 1000.0
            sources = [hit.chunk.source for hit in hits]
            scores = [round(hit.score, 3) for hit in hits]
            print(
                f"utterance_id={uid} generation_id={gen} rag_triggered=true "
                f"hits={len(hits)} sources={sources} scores={scores} latency_ms={rag_ms:.1f}",
                flush=True,
            )
            self._ev(
                "rag",
                utterance_id=uid,
                generation_id=gen,
                sources=sources,
                chunk_ids=[hit.chunk.chunk_id for hit in hits],
                scores=scores,
            )
            if not hits:
                self._deliver_direct(job, "I couldn't find that in the available company knowledge.")
                return
            self._stream_llm_answer(job, knowledge_hits=hits)
            return
        self._stream_llm_answer(job)

    def _start_tool(self, job: dict, tool_name: str, arguments: dict, route_finished: float) -> None:
        uid = job.get("utterance_id")
        gen = job["gen"]
        started = time.perf_counter()
        print(
            f"utterance_id={uid} generation_id={gen} tool_name={tool_name} "
            f"tool_arguments={json.dumps(arguments, sort_keys=True)} tool_started=true",
            flush=True,
        )
        if not self._tool_slots.acquire(blocking=False):
            print(
                f"utterance_id={uid} generation_id={gen} tool_submit_error=backpressure",
                flush=True,
            )
            self._deliver_direct(job, "I'm handling too many requests right now. Please try again.")
            return
        try:
            future = self._tool_executor.submit(
                execute_tool,
                tool_name,
                arguments,
                timeout_s=float(self.m.cfg.weather_timeout_s),
            )
        except Exception as exc:
            self._tool_slots.release()
            print(
                f"utterance_id={uid} generation_id={gen} tool_submit_error={type(exc).__name__}",
                flush=True,
            )
            self._deliver_direct(job, "I couldn't run that tool right now.")
            return

        turn_lock = threading.Lock()
        turn: dict[str, Any] = {
            "lock": turn_lock,
            "future": future,
            "timer": None,
            "ack_started": False,
            "tool_name": tool_name,
        }
        with self._lock:
            if gen != self.gen or self._stop.is_set():
                future.add_done_callback(lambda _done: self._tool_slots.release())
                future.cancel()
                print(
                    f"tool_result_discarded reason=stale_generation generation_id={gen} active={self.gen}",
                    flush=True,
                )
                return
            self._tool_turns[gen] = turn

        ack = acknowledgement_for(tool_name)

        def ack_if_slow() -> None:
            if not ack:
                return
            with turn_lock:
                if future.done() or gen != self.gen or self._stop.is_set():
                    return
                turn["ack_started"] = True
                print(
                    f"utterance_id={uid} generation_id={gen} tool_name={tool_name} "
                    f"tool_ack_started=true elapsed_ms={(time.perf_counter() - started) * 1000.0:.1f}",
                    flush=True,
                )
                self._emit_ack(job, ack)

        def completed(done: Future) -> None:
            self._tool_slots.release()
            with turn_lock:
                timer = turn.get("timer")
                if timer is not None:
                    timer.cancel()
                latency_ms = (time.perf_counter() - started) * 1000.0
                result = None
                error_message = None
                error_code = None
                try:
                    result = done.result()
                except ToolError as exc:
                    error_message = exc.user_message
                    error_code = exc.code
                except Exception as exc:
                    error_message = "I couldn't complete that request right now."
                    error_code = type(exc).__name__
                    print(
                        f"utterance_id={uid} generation_id={gen} tool_unexpected_error={type(exc).__name__}",
                        flush=True,
                    )
                    traceback.print_exc()
                if gen != self.gen or self._stop.is_set():
                    print(
                        f"tool_result_discarded reason=stale_generation generation_id={gen} active={self.gen}",
                        flush=True,
                    )
                    return
                if error_message is None:
                    print(
                        f"utterance_id={uid} generation_id={gen} tool_finished=true "
                        f"tool_latency_ms={latency_ms:.1f} tool_result={json.dumps(result, sort_keys=True)}",
                        flush=True,
                    )
                else:
                    print(
                        f"utterance_id={uid} generation_id={gen} tool_finished=true "
                        f"tool_latency_ms={latency_ms:.1f} tool_error={error_code}",
                        flush=True,
                    )
                continuation = {
                    **job,
                    "kind": "tool_result",
                    "tool_name": tool_name,
                    "tool_arguments": arguments,
                    "tool_result": result,
                    "tool_error": error_message,
                    "tool_error_code": error_code,
                    "tool_started_at": started,
                    "tool_route_finished": route_finished,
                    "ack_started": bool(turn["ack_started"]),
                    "request_id": job.get("request_id"),
                }
                try:
                    self._llm_q.put(continuation, timeout=0.5)
                except queue.Full:
                    print(
                        f"utterance_id={uid} generation_id={gen} dropped reason=tool-continuation-queue-full",
                        flush=True,
                    )
                    if gen == self.gen:
                        self._deliver_direct(job, "I couldn't complete that request right now.")
            with self._lock:
                if self._tool_turns.get(gen) is turn:
                    self._tool_turns.pop(gen, None)

        if ack:
            timer = threading.Timer(float(self.m.cfg.tool_ack_delay_s), ack_if_slow)
            timer.daemon = True
            turn["timer"] = timer
            timer.start()
        future.add_done_callback(completed)

    def _emit_ack(self, job: dict, text: str) -> None:
        gen = job["gen"]
        uid = job.get("utterance_id")
        if gen != self.gen or self._stop.is_set():
            return
        self._ev("bot_delta", text=text + " ", utterance_id=uid, generation_id=gen, phase="ack")
        self._enqueue_tts(
            text,
            gen,
            job["t"],
            job["stt"],
            1,
            uid,
            t_stt_done=job.get("t_stt_done"),
            phase="ack",
            phase_first=True,
        )

    def _tool_result_one(self, job: dict) -> None:
        gen = job["gen"]
        uid = job.get("utterance_id")
        if gen != self.gen or self._stop.is_set():
            print(
                f"tool_result_discarded reason=stale_generation generation_id={gen} active={self.gen}",
                flush=True,
            )
            return
        if job.get("tool_error"):
            self._last_successful_tool = None
            self._deliver_direct(job, str(job["tool_error"]))
            return
        tool_name = job["tool_name"]
        arguments = job["tool_arguments"]
        result = job["tool_result"]
        self._last_successful_tool = tool_name
        direct = format_tool_result(tool_name, arguments, result)
        if requires_synthesis(tool_name, job["text"]):
            self._stream_llm_answer(
                job,
                tool_name=tool_name,
                tool_result=result,
                direct_fallback=direct,
            )
        else:
            self._deliver_direct(job, direct)

    def _deliver_direct(self, job: dict, text: str) -> None:
        gen = job["gen"]
        uid = job.get("utterance_id")
        cancel = _GenerationCancel(self, gen)
        text = (text or "").strip()
        if not text or cancel.is_set():
            return
        idx = 2 if job.get("ack_started") else 1
        self._ev("bot_delta", text=text, utterance_id=uid, generation_id=gen, phase="final")
        self._enqueue_tts(
            text,
            gen,
            job["t"],
            job["stt"],
            idx,
            uid,
            t_stt_done=job.get("t_stt_done"),
            phase="final",
            phase_first=True,
        )
        if cancel.is_set():
            return
        with self._lock:
            if cancel.is_set():
                return
            self._remember(job["text"], text)
        if cancel.is_set():
            return
        self._ev("bot", text=text, llm_s=0.0, utterance_id=uid, generation_id=gen)
        self._queue_eos(job, {"s": 0.0, "text": text, "pieces": 1})

    def _stream_llm_answer(
        self,
        job: dict,
        *,
        tool_name: str | None = None,
        tool_result: dict | None = None,
        direct_fallback: str | None = None,
        knowledge_hits: list[Hit] | None = None,
    ) -> None:
        uid = job.get("utterance_id")
        gen = job["gen"]
        cancel = _GenerationCancel(self, gen)
        if cancel.is_set():
            return
        user = job["text"]
        t_end = job["t"]
        t_stt_done = float(job.get("t_stt_done") or t_end)
        t0 = time.perf_counter()
        print(
            f"utterance_id={uid} generation_id={gen} final_llm_started=true tool_name={tool_name}",
            flush=True,
        )
        t_first_tok = None
        t_first_tts = None
        parts: list[str] = []
        emitted_parts: list[str] = []
        buf = SentenceBuffer()
        base_tts = 1 if job.get("ack_started") else 0
        n_tts = base_tts
        first_tok = False
        first_sub = False
        t_first_subtitle = None
        llm_error = None
        guard_reason: str | None = None
        guard_latency_s = 0.0
        for attempt in (1, 2):
            parts = []
            emitted_parts = []
            buf = SentenceBuffer()
            n_tts = base_tts
            first_tok = False
            first_sub = False
            t_first_subtitle = None
            llm_error = None
            guard_reason = None
            guard_latency_s = 0.0
            guard = None
            if (
                tool_name is None
                and not knowledge_hits
                and bool(getattr(self.m.cfg, "factual_guard_enabled", True))
            ):
                prior_context = " ".join(str(item.get("content") or "") for item in self.history)
                guard = FactualRiskGuard(
                    user,
                    prior_context=prior_context,
                    specificity_threshold=int(
                        getattr(self.m.cfg, "factual_guard_specificity_score", 2)
                    ),
                    causal_threshold=int(getattr(self.m.cfg, "factual_guard_causal_score", 3)),
                )
                if guard.needs_extended_prefix:
                    # The normal 14-word first chunk is excellent for TTFA but too
                    # short to judge causal or unfamiliar multi-word claims. Four
                    # extra words keep the median overhead below the 100 ms target.
                    buf = SentenceBuffer(first_max_words=18, later_max_words=28)

            def commit_chunk(text: str, *, emit_delta: bool) -> None:
                nonlocal first_sub, n_tts, t_first_tts, t_first_subtitle
                text = (text or "").strip()
                if not text or cancel.is_set():
                    return
                emitted_parts.append(text)
                if emit_delta:
                    self._ev(
                        "bot_delta",
                        text=text + " ",
                        utterance_id=uid,
                        generation_id=gen,
                        phase="final",
                    )
                if not first_sub:
                    first_sub = True
                    t_first_subtitle = time.perf_counter()
                    print(
                        f"utterance_id={uid} subtitle first +{time.perf_counter() - t_stt_done:.3f}s",
                        flush=True,
                    )
                n_tts += 1
                if n_tts == base_tts + 1:
                    t_first_tts = time.perf_counter()
                    dt_tok = (t_first_tts - t_first_tok) if t_first_tok else 0.0
                    print(
                        f"utterance_id={uid} first_tts_chunk generation_id={gen} "
                        f"first_token_to_first_tts={dt_tok:.3f}s text={_safe_text(self.m.cfg, text)!r}",
                        flush=True,
                    )
                self._enqueue_tts(
                    text,
                    gen,
                    t_end,
                    job["stt"],
                    n_tts,
                    uid,
                    t_stt_done=t_stt_done,
                    phase="final",
                    phase_first=n_tts == base_tts + 1,
                )
                if n_tts == base_tts + 1 and t_first_subtitle is not None:
                    print(
                        f"utterance_id={uid} subtitle_to_tts_request_ms="
                        f"{(time.perf_counter() - t_first_subtitle) * 1000.0:.1f}",
                        flush=True,
                    )

            def review_and_commit(text: str) -> None:
                nonlocal guard_reason, guard_latency_s
                if guard is None:
                    commit_chunk(text, emit_delta=False)
                    return
                started = time.perf_counter()
                decision = guard.inspect(text)
                guard_latency_s += time.perf_counter() - started
                if decision.text:
                    commit_chunk(decision.text, emit_delta=True)
                if decision.triggered and guard_reason is None:
                    guard_reason = decision.reason or "risk"
            try:
                if tool_name is not None:
                    factory = lambda: self.m.llm.stream_with_tool_result(user, tool_name, tool_result or {}, cancel)
                elif knowledge_hits:
                    factory = lambda: self.m.llm.stream_with_knowledge(
                        user, format_knowledge_block(knowledge_hits), cancel
                    )
                else:
                    factory = lambda: self.m.llm.stream(user, cancel, history=self.history)
                gate = getattr(self.m, "qwen_gate", None)
                qwen_job = None
                if gate is not None:
                    stream, qwen_job = gate.iterate(self.session_id, factory, cancel)
                else:
                    stream = factory()
                for piece in stream:
                    if cancel.is_set():
                        print(f"utterance_id={uid} llm_aborted reason=gen-mismatch-during-stream", flush=True)
                        break
                    if not first_tok:
                        first_tok = True
                        t_first_tok = time.perf_counter()
                        self._metric("first_llm_token", utterance_id=uid)
                        pe = float(getattr(self.m.llm, "t_prompt_eval", 0) or 0)
                        print(
                            f"utterance_id={uid} llm_first_token generation_id={gen} "
                            f"llm_start_to_first_token={t_first_tok - t0:.3f}s "
                            f"stt_to_first_token={t_first_tok - t_stt_done:.3f}s "
                            f"prompt_eval={pe:.3f}s attempt={attempt} final_llm_first_token=true",
                            flush=True,
                        )
                    parts.append(piece)
                    if guard is None:
                        self._ev("bot_delta", text=piece, utterance_id=uid, generation_id=gen, phase="final")
                    for sent in buf.push(piece):
                        review_and_commit(sent)
                tail = buf.flush()
                if tail and not cancel.is_set():
                    review_and_commit(tail)
                if qwen_job is not None:
                    self.last_qwen_wait_s = qwen_job.wait_s
                    self.last_qwen_infer_s = qwen_job.infer_s
                    print(
                        f"utterance_id={uid} generation_id={gen} session_id={self.session_id} "
                        f"qwen_queue_wait_ms={qwen_job.wait_s * 1000.0:.1f} "
                        f"qwen_inference_ms={qwen_job.infer_s * 1000.0:.1f}",
                        flush=True,
                    )
                    self._metric(
                        "qwen_timing",
                        utterance_id=uid,
                        qwen_queue_wait_ms=round(qwen_job.wait_s * 1000.0, 1),
                        qwen_inference_ms=round(qwen_job.infer_s * 1000.0, 1),
                    )
            except BusyError as exc:
                print(
                    f"utterance_id={uid} generation_id={gen} session_id={self.session_id} "
                    f"qwen_busy retry_after_ms={exc.retry_after_ms}",
                    flush=True,
                )
                self._ev(
                    "busy",
                    retry_after_ms=exc.retry_after_ms,
                    reason=exc.reason,
                    utterance_id=uid,
                    generation_id=gen,
                )
                self.in_flight = False
                self._resume_listen_ui("qwen-busy")
                return
            except Exception as exc:
                llm_error = exc
                print(f"utterance_id={uid} llm_error={exc!r} attempt={attempt}", flush=True)
                traceback.print_exc()
                self._ev("error", text="LLM response failed", utterance_id=uid, generation_id=gen)
            if first_tok or parts or llm_error is not None:
                break
            if cancel.is_set():
                break
            print(f"utterance_id={uid} empty-llm-output retrying attempt={attempt}", flush=True)
            try:
                llama = getattr(self.m.llm, "llama", None)
                if llama is not None:
                    self.m.llm.reset_cache()
                    print(f"utterance_id={uid} llama.reset() after empty output", flush=True)
            except Exception:
                traceback.print_exc()
        raw_full = _clean_reply("".join(parts))
        full = _clean_reply(" ".join(emitted_parts)) if guard is not None else raw_full
        llm_s = time.perf_counter() - t0
        generated_tokens = 0
        llama = getattr(self.m.llm, "llama", None)
        if llama is not None and raw_full:
            try:
                generated_tokens = len(llama.tokenize(raw_full.encode("utf-8"), add_bos=False))
            except Exception:
                generated_tokens = 0
        words = len(full.split()) if full else 0
        self._metric(
            "qwen_generation_complete",
            utterance_id=uid,
            generated_tokens=generated_tokens,
            generation_s=llm_s,
        )
        if guard is not None:
            guard_ms = guard_latency_s * 1000.0
            print(
                f"utterance_id={uid} generation_id={gen} "
                f"factual_guard_triggered={'true' if guard_reason else 'false'} "
                f"factual_guard_reason={guard_reason or 'none'} factual_guard_latency_ms={guard_ms:.3f}",
                flush=True,
            )
            self._metric(
                "factual_guard",
                utterance_id=uid,
                triggered=bool(guard_reason),
                reason=guard_reason,
                latency_ms=round(guard_ms, 3),
            )
        print(
            f"utterance_id={uid} llm_finished generation_id={gen} chars={len(full)} words={words} "
            f"s={llm_s:.2f} tts_chunks={n_tts} first_token={first_tok} "
            f"job_gen={gen} current_gen={self.gen} text={_safe_text(self.m.cfg, full)!r}",
            flush=True,
        )
        if llm_error is not None:
            print(f"utterance_id={uid} llm_finished with error (see traceback above)", flush=True)
        elif not first_tok and not full:
            why = "cancel-before-first-token" if cancel.is_set() else "empty-llm-output"
            print(f"utterance_id={uid} dropped reason={why}", flush=True)
            self._ev("error", text=f"LLM produced no tokens ({why})", utterance_id=uid, generation_id=gen)
        if not full and direct_fallback and not cancel.is_set():
            self._deliver_direct(job, direct_fallback)
            return
        if full and not cancel.is_set():
            with self._lock:
                if cancel.is_set():
                    return
                self._remember(user, full)
            if cancel.is_set():
                return
            print(f"BOT  {_safe_text(self.m.cfg, full)!r}  llm={llm_s:.2f}s utterance_id={uid}", flush=True)
            self._ev("bot", text=full, llm_s=round(llm_s, 3), utterance_id=uid, generation_id=gen)
        stats = {"s": llm_s, "text": full, "pieces": len(parts)}
        self._queue_eos(job, stats)

    def _queue_eos(self, job: dict, stats: dict) -> None:
        gen = job["gen"]
        if gen != self.gen or self._stop.is_set():
            return
        try:
            self._tts_q.put_nowait(
                {
                    "eos": True,
                    "gen": gen,
                    "t_end": job["t"],
                    "t_stt_done": job.get("t_stt_done"),
                    "stt": job["stt"],
                    "llm": stats,
                    "utterance_id": job.get("utterance_id"),
                    "request_id": job.get("request_id"),
                }
            )
        except queue.Full:
            print(f"utterance_id={job.get('utterance_id')} tts_queue_full eos_dropped=true", flush=True)
            self.in_flight = False
            self._resume_listen_ui("tts-queue-failed")

    def _enqueue_tts(
        self,
        text: str,
        gen: int,
        t_end: float,
        stt: dict,
        idx: int,
        uid: int | None = None,
        *,
        t_stt_done: float | None = None,
        phase: str = "final",
        phase_first: bool = False,
    ) -> None:
        text = (text or "").strip()
        if not text:
            print(f"utterance_id={uid} dropped reason=empty-tts-chunk idx={idx}", flush=True)
            return
        if gen != self.gen or self.cancel.is_set():
            print(
                f"utterance_id={uid} dropped reason=tts-stale-or-cancel idx={idx} "
                f"job_gen={gen} current_gen={self.gen} cancel={self.cancel.is_set()}",
                flush=True,
            )
            return
        qlen = self._tts_q.qsize()
        nwords = len(text.split())
        print(
            f"utterance_id={uid} generation_id={gen} tts_text_chunk i={idx} phase={phase} "
            f"n_words={nwords} qlen={qlen} text={_safe_text(self.m.cfg, text)!r}",
            flush=True,
        )
        if phase == "final" and phase_first:
            self._metric("first_final_text_to_tts", utterance_id=uid)
            print(
                f"utterance_id={uid} generation_id={gen} final_tts_started=true text={_safe_text(self.m.cfg, text)!r}",
                flush=True,
            )
        if idx == 1:
            print(f"utterance_id={uid} tts_started generation_id={gen} text={_safe_text(self.m.cfg, text)!r} qlen={qlen}", flush=True)
            self._ev("state", state="speaking", utterance_id=uid, generation_id=gen)
        t_tts_request = time.perf_counter()
        try:
            self._tts_q.put_nowait(
                {
                    "text": text,
                    "gen": gen,
                    "t_end": t_end,
                    "idx": idx,
                    "first": idx == 1,
                    "phase": phase,
                    "phase_first": phase_first,
                    "t_stt_done": t_stt_done,
                    "stt": stt,
                    "utterance_id": uid,
                    "t_tts_request": t_tts_request,
                    "request_id": self.request_id,
                }
            )
        except queue.Full:
            print(f"utterance_id={uid} tts_queue_full dropped=true", flush=True)
            self._ev("error", text="audio queue full", utterance_id=uid, generation_id=gen)

    def _tts_loop(self) -> None:
        t_first_audio = None
        total_audio = 0.0
        total_gen = 0.0
        started = False
        while not self._stop.is_set():
            job = self._tts_q.get()
            if job is None:
                return
            uid = job.get("utterance_id")
            if job.get("gen") != self.gen:
                print(
                    f"utterance_id={uid} dropped reason=tts-stale-gen "
                    f"job_gen={job.get('gen')} current_gen={self.gen} eos={bool(job.get('eos'))}",
                    flush=True,
                )
                started = False
                t_first_audio = None
                total_audio = 0.0
                total_gen = 0.0
                continue
            if job.get("eos"):
                print(
                    f"playback_generation_tts_complete generation_id={job['gen']} "
                    f"utterance_id={uid} total_audio_s={total_audio:.3f}",
                    flush=True,
                )
                self._ev(
                    "audio_end",
                    audio_s=round(total_audio, 3),
                    generation_id=job["gen"],
                    utterance_id=uid,
                )
                timeout = max(1.5, total_audio + 1.5) if started else 0.2
                if started and job["gen"] == self.gen and not self.cancel.is_set():
                    deadline = time.perf_counter() + timeout
                    while job["gen"] == self.gen and not self.cancel.is_set() and not self._stop.is_set():
                        if self._playback_done.wait(timeout=0.25):
                            print("TTS playback_end confirmed_by_browser=true", flush=True)
                            break
                        if time.perf_counter() >= deadline:
                            # This is diagnostic only. Do not clear playing or
                            # resume listening until the browser's final source
                            # actually ends (or a real interrupt changes gen).
                            print(
                                f"TTS playback_end waiting_for_browser generation_id={job['gen']} "
                                f"expected_after_s={timeout:.1f}",
                                flush=True,
                            )
                            deadline = time.perf_counter() + max(2.0, total_audio * 0.25)
                if job["gen"] != self.gen:
                    started = False
                    t_first_audio = None
                    total_audio = 0.0
                    total_gen = 0.0
                    continue
                stt = job.get("stt") or {}
                llm = job.get("llm") or {}
                if t_first_audio is not None:
                    self._metric("final_response_complete", utterance_id=uid)
                    self._ev(
                        "metrics",
                        ttfa_s=round(t_first_audio - job["t_end"], 3),
                        e2e_s=round(time.perf_counter() - job["t_end"], 3),
                        stt_s=round(stt.get("decode_s") or 0, 3),
                        stt_rtf=round(stt.get("rtf") or 0, 3),
                        llm_s=round(float(llm.get("s") or 0), 3),
                        tts_s=round(total_gen, 3),
                        tts_audio_s=round(total_audio, 3),
                        tts_rtf=round((total_gen / total_audio) if total_audio else 0, 3),
                        generation_id=job["gen"],
                        utterance_id=uid,
                    )
                self.playing = False
                self.in_flight = False
                # If the user is already barging in, do not eat their first
                # words with the post-TTS ignore window.
                barge_pending = bool(self.vad.speech or self.vad.loud_n > 0)
                self.vad.set_playing(False, ignore=not barge_pending)
                if barge_pending:
                    print(
                        f"tts-finished barge_pending=true ignore=false "
                        f"generation_id={self.gen} mic_capture_continues=true",
                        flush=True,
                    )
                self.barge_armed = False
                self._resume_listen_ui("tts-finished")
                started = False
                t_first_audio = None
                total_audio = 0.0
                total_gen = 0.0
                continue

            text = job.get("text") or ""
            idx = job.get("idx") or 0
            if not started:
                if job["gen"] != self.gen:
                    continue
                self.playing = True
                self.playing_since = time.perf_counter()
                self.vad.set_playing(True)
                self.barge_armed = True
                self._playback_done.clear()
                self._ev(
                    "audio_start",
                    sr=self.m.tts.sr,
                    tts_speed=float(self.m.cfg.tts_speed),
                    generation_id=job["gen"],
                    utterance_id=uid,
                )
                started = True
                print(
                    f"barge_in_armed=true generation_id={job['gen']} utterance_id={uid} "
                    f"mic_capture_continues=true",
                    flush=True,
                )
            self._ev(
                "tts_chunk",
                i=idx,
                text=text,
                qlen=self._tts_q.qsize(),
                utterance_id=uid,
                generation_id=job["gen"],
                phase=job.get("phase") or "final",
            )
            print(
                f"utterance_id={uid} tts_gen_start idx={idx} qlen={self._tts_q.qsize()} "
                f"text={_safe_text(self.m.cfg, text)!r}",
                flush=True,
            )
            t_gen0 = time.perf_counter()
            phrase_audio = 0.0
            held: list[np.ndarray] = []
            held_n = 0
            phase_audio_logged = False
            min_pack = max(1, int(0.16 * self.m.tts.sr))

            def _flush_held() -> None:
                nonlocal held, held_n
                if not held:
                    return
                packed = held[0] if len(held) == 1 else np.concatenate(held)
                self._emit_audio(job["gen"], packed)
                held = []
                held_n = 0

            try:
                cancel = _GenerationCancel(self, job["gen"])
                gate = getattr(self.m, "tts_gate", None)
                tts_job = None
                if gate is not None:
                    stream, tts_job = gate.iterate(
                        self.session_id,
                        lambda: self.m.tts.stream(text, cancel),
                        cancel,
                    )
                else:
                    stream = self.m.tts.stream(text, cancel)
                for wav, meta in stream:
                    if job["gen"] != self.gen:
                        print(
                            f"utterance_id={uid} TTS cancel reason=gen-mismatch during stream",
                            flush=True,
                        )
                        break
                    if meta.get("done"):
                        _flush_held()
                        phrase_audio = float(meta.get("audio_s") or 0)
                        total_gen += float(meta.get("gen_s") or 0)
                        total_audio += phrase_audio
                        continue
                    if wav.size == 0:
                        continue
                    if job.get("phase_first") and not phase_audio_logged:
                        phase_audio_logged = True
                        stt_done = float(job.get("t_stt_done") or job["t_end"])
                        elapsed_ms = (time.perf_counter() - stt_done) * 1000.0
                        if job.get("phase") == "ack":
                            print(
                                f"utterance_id={uid} generation_id={job['gen']} "
                                f"STT_end_to_ack_audio_ms={elapsed_ms:.1f}",
                                flush=True,
                            )
                        else:
                            print(
                                f"utterance_id={uid} generation_id={job['gen']} "
                                f"STT_end_to_final_audio_ms={elapsed_ms:.1f}",
                                flush=True,
                            )
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                        self._metric("first_tts_audio", utterance_id=uid)
                        dt = t_first_audio - job["t_end"]
                        print(
                            f"utterance_id={uid} first_audio generation_id={job['gen']} "
                            f"+{dt:.3f}s idx={idx} "
                            f"tts_request_to_first_pcm_ms="
                            f"{(t_first_audio - float(job.get('t_tts_request') or t_gen0)) * 1000.0:.1f}",
                            flush=True,
                        )
                        self._ev("ttfa", s=round(dt, 3), utterance_id=uid, generation_id=job["gen"])
                        self._emit_audio(job["gen"], wav)
                    else:
                        held.append(wav)
                        held_n += int(wav.size)
                        if held_n >= min_pack:
                            _flush_held()
                _flush_held()
                if tts_job is not None:
                    self.last_tts_wait_s = tts_job.wait_s
                    print(
                        f"utterance_id={uid} tts_queue_wait_ms={tts_job.wait_s * 1000.0:.1f} "
                        f"session_id={self.session_id}",
                        flush=True,
                    )
            except BusyError as exc:
                print(
                    f"utterance_id={uid} tts_busy session_id={self.session_id} "
                    f"retry_after_ms={exc.retry_after_ms}",
                    flush=True,
                )
                self._ev("busy", retry_after_ms=exc.retry_after_ms, reason=exc.reason, utterance_id=uid)
            except Exception as exc:
                print(f"utterance_id={uid} tts_error={exc!r}", flush=True)
                traceback.print_exc()
            gen_s = time.perf_counter() - t_gen0
            rtf = (gen_s / phrase_audio) if phrase_audio > 0.01 else 0.0
            print(
                f"utterance_id={uid} tts_gen_end idx={idx} audio_s={phrase_audio:.2f} "
                f"gen_s={gen_s:.2f} tts_generation_rtf={rtf:.2f} queue_length={self._tts_q.qsize()}",
                flush=True,
            )
            self._ev(
                "tts_rtf",
                rtf=round(rtf, 3),
                audio_s=round(phrase_audio, 3),
                gen_s=round(gen_s, 3),
                utterance_id=uid,
                generation_id=job["gen"],
            )

    def _emit_audio(self, gen: int, wav: np.ndarray) -> None:
        if gen != self.gen:
            print(
                f"tts_chunk_discarded reason=stale_generation generation_id={gen} active={self.gen}",
                flush=True,
            )
            return
        pkt = struct.pack("<I", int(gen) & 0xFFFFFFFF) + f32_to_i16(wav)
        self.emit(("audio", pkt))

    def _ev(self, typ: str, **kw) -> None:
        kw.setdefault("session_id", self.session_id)
        kw.setdefault("request_id", self.request_id)
        kw.setdefault("generation_id", self.gen)
        msg = {"type": typ, **kw}
        try:
            self.emit(msg)
        except Exception:
            print(f"emit failed type={typ}", flush=True)
            traceback.print_exc()

    def _soak_note(self, generation_id: int, **fields: Any) -> None:
        if getattr(self, "_soak_path", None) is None:
            return
        with self._soak_lock:
            row = self._soak_records.setdefault(generation_id, {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "session_id": self.session_id,
                "generation_id": generation_id,
            })
            row.update({key: value for key, value in fields.items() if value is not None})

    def _soak_finish(self, generation_id: int, **fields: Any) -> None:
        if getattr(self, "_soak_path", None) is None:
            return
        with self._soak_lock:
            row = self._soak_records.pop(generation_id, None)
            if row is None:
                return
            row.update({key: value for key, value in fields.items() if value is not None})
            row["listening_enabled"] = self.listening_enabled
            try:
                with self._soak_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            except OSError as exc:
                print(f"soak_diagnostics_write_error={type(exc).__name__}", flush=True)

    def _metric(self, event: str, **kw) -> None:
        kw.setdefault("session_id", self.session_id)
        kw.setdefault("request_id", self.request_id)
        kw.setdefault("generation_id", self.gen)
        print(
            "E2E_EVENT "
            + json.dumps(
                {
                    "event": event,
                    "mono": time.perf_counter(),
                    "wall_ns": time.time_ns(),
                    **kw,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @staticmethod
    def _drain(q: queue.Queue) -> int:
        n = 0
        while True:
            try:
                q.get_nowait()
                n += 1
            except queue.Empty:
                return n


def _round(st: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}


def dump_ready(models: Models) -> str:
    return json.dumps(
        {
            "ready": True,
            "llm": models.cfg.llm,
            "llm_id": models.cfg.llm_id,
            "llm_backend": getattr(models.llm, "backend", "hf"),
            "llm_requested_backend": getattr(models.llm, "backend_requested", "cpu"),
            "llm_actual_backend": getattr(models.llm, "actual_backend", getattr(models.llm, "backend", "hf")),
            "llm_gpu_offload": bool(getattr(models.llm, "gpu_offload", False)),
            "llm_gpu_verified": bool(getattr(models.llm, "gpu_verified", False)),
            "llm_gpu_device": getattr(models.llm, "gpu_device", None),
            "llm_layers_offloaded": int(getattr(models.llm, "layers_offloaded", 0)),
            "llm_fallback_reason": getattr(models.llm, "fallback_reason", "") or None,
            "llm_model_path": str(models.cfg.gguf_path) if models.cfg.gguf_path.exists() else None,
            "llm_n_ctx": models.cfg.llm_n_ctx,
            "llm_n_batch": models.cfg.llm_n_batch,
            "llm_n_ubatch": models.cfg.llm_n_ubatch,
            "llm_threads": models.llm.cfg.llm_threads or (4 if models.cfg.llm_n_gpu_layers else max(4, min(6, models.cfg.cpu_threads + 2))),
            "llm_flash_attn": models.cfg.llm_flash_attn,
            "llm_thinking": getattr(models.llm, "thinking", "unknown"),
            "tool_router_json_grammar": bool(getattr(models.llm, "json_grammar_supported", False)),
            "tools": ["get_current_time", "calculator", "get_weather"],
            "asr": "parakeet-tdt-0.6b-v2",
            "asr_device": models.asr.device,
            "asr_decoding": "greedy_search",
            "tts": "pocket-tts",
            "tts_sr": models.tts.sr,
            "tts_speed": float(models.cfg.tts_speed),
            "qwen_queue_depth": int(models.cfg.qwen_queue_depth),
            "in_sr": SR,
            "load_s": {k: round(v, 2) for k, v in models.load.items()},
        }
    )
