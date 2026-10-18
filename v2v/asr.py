from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

from .config import ROOT, Settings

SR = 16000
WIN = 512  # 32 ms


def speech_score(win: np.ndarray) -> float:
    """Fallback only when Silero is unavailable. Not a real VAD.

    The old ZCR/HF mix scored *fan/hiss at 1.0* and *vowels near 0*, which is
    why the volume bar moved but Parakeet never ran. Broadband noise is now
    down-weighted via spectral flatness.
    """
    x = np.asarray(win, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-5:
        return 0.0
    zc = float(np.count_nonzero(np.diff(np.signbit(x)))) / max(x.size - 1, 1)
    hp = np.diff(x, prepend=x[0])
    hf = float(np.sqrt(np.mean(hp * hp) + 1e-12)) / rms
    zc_s = float(np.clip((zc - 0.035) / 0.13, 0.0, 1.0))
    hf_s = float(np.clip((hf - 0.32) / 0.70, 0.0, 1.0))
    score = 0.45 * zc_s + 0.55 * hf_s
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12
    flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    if flat > 0.40:
        score *= float(np.clip((0.55 - flat) / 0.15, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


class SileroSpeech:
    """Real Silero VAD via sherpa-onnx. 1.0 = speech, 0.0 = not speech."""

    def __init__(self, path: Path):
        self.ok = False
        self.window = WIN
        self._lock = threading.Lock()
        self.model = None
        if not path.exists():
            print(f"VAD backend=heuristic silero missing at {path}", flush=True)
            return
        try:
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(path)
            cfg.silero_vad.threshold = 0.5
            cfg.silero_vad.window_size = WIN
            cfg.sample_rate = SR
            cfg.num_threads = 1
            self.model = sherpa_onnx.VadModel.create(cfg)
            try:
                self.window = int(self.model.window_size())
            except Exception:
                self.window = WIN
            self.ok = True
            print(f"VAD backend=silero window={self.window} model={path.name}", flush=True)
        except Exception as exc:
            print(f"VAD backend=heuristic silero failed: {exc!r}", flush=True)
            self.model = None

    def prob(self, win: np.ndarray) -> float:
        if not self.ok or self.model is None:
            return -1.0
        x = np.ascontiguousarray(win, dtype=np.float32).reshape(-1)
        if x.size < self.window:
            y = np.zeros(self.window, dtype=np.float32)
            y[: x.size] = x
            x = y
        elif x.size > self.window:
            x = x[: self.window]
        try:
            with self._lock:
                return 1.0 if self.model.is_speech(x) else 0.0
        except Exception as exc:
            print(f"silero is_speech error={exc!r}", flush=True)
            return 0.0

    def reset(self) -> None:
        if self.model is None:
            return
        try:
            with self._lock:
                self.model.reset()
        except Exception:
            pass


JUNK = re.compile(
    r"^(uh+|um+|mm+|hmm+|ah+|oh+|mhm|mm-?hmm|huh|yeah|yep|yes|no|okay|ok)[.!,?]*$",
    re.I,
)


def normalize_pcm(pcm: np.ndarray) -> tuple[np.ndarray, float]:
    pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if 0.04 < peak < 0.22:
        pcm = np.clip(pcm * (0.45 / peak), -1.0, 1.0)
    return pcm, peak


def audio_quality_report(pcm: np.ndarray, silence_rms: float = 0.008) -> dict:
    """Return cheap capture/segmentation diagnostics without changing audio."""
    x = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
    if not x.size:
        return {
            "duration_s": 0.0,
            "leading_silence_ms": 0.0,
            "trailing_silence_ms": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "clipping_ratio": 0.0,
            "noise_floor_rms": 0.0,
            "crest_factor": 0.0,
        }
    frame_n = max(1, int(0.01 * SR))
    frame_rms = []
    for i in range(0, x.size, frame_n):
        frame = x[i : i + frame_n]
        frame_rms.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
    active = [i for i, value in enumerate(frame_rms) if value >= silence_rms]
    lead_frames = active[0] if active else len(frame_rms)
    tail_frames = (len(frame_rms) - 1 - active[-1]) if active else len(frame_rms)
    noise_floor = float(np.median(np.partition(np.asarray(frame_rms), max(0, len(frame_rms) // 5 - 1))[: max(1, len(frame_rms) // 5)]))
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(x)))
    return {
        "duration_s": x.size / SR,
        "leading_silence_ms": lead_frames * 10.0,
        "trailing_silence_ms": tail_frames * 10.0,
        "peak": peak,
        "rms": rms,
        "clipping_ratio": float(np.mean(np.abs(x) >= 0.995)),
        "noise_floor_rms": noise_floor,
        "crest_factor": peak / max(rms, 1e-8),
    }


def _utt_dump_dir() -> Path | None:
    raw = (os.environ.get("V2V_SAVE_UTT") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


def pad_utterance(pcm: np.ndarray, head_s: float = 0.20, tail_s: float = 0.20) -> np.ndarray:
    pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [
            np.zeros(int(head_s * SR), dtype=np.float32),
            pcm,
            np.zeros(int(tail_s * SR), dtype=np.float32),
        ]
    )


class ParakeetASR:
    def __init__(self, cfg: Settings):
        self.device = "cpu"
        self.load_s = 0.0
        self._lock = threading.Lock()
        self._ov = None
        self._utt_dump = _utt_dump_dir()
        self._utt_seq = 0
        t0 = time.perf_counter()
        if self._try_openvino(cfg):
            self.load_s = time.perf_counter() - t0
            return
        d = cfg.parakeet_dir
        if not (d / "encoder.int8.onnx").exists():
            raise FileNotFoundError(f"Parakeet INT8 model missing at {d}")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=max(2, cfg.asr_threads),
            sample_rate=SR,
            feature_dim=80,
            dither=0.00003,
            provider="cpu",
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )
        self.device = "cpu-sherpa"
        self.load_s = time.perf_counter() - t0

    def _try_openvino(self, cfg: Settings) -> bool:
        ov_dir = cfg.parakeet_dir.parent / "parakeet-tdt-0.6b-v2-ov"
        if not (ov_dir / "parakeet_encoder.xml").exists():
            return False
        try:
            import openvino as ov
        except ImportError:
            print("  openvino not installed; sherpa CPU ASR", flush=True)
            return False
        core = ov.Core()
        devices = []
        try:
            devices = list(core.available_devices)
        except Exception:
            devices = ["CPU"]
        print(f"  OpenVINO devices: {devices}", flush=True)
        requested = str(getattr(cfg, "asr_device", "AUTO") or "AUTO").upper()
        enc_dev = "CPU"
        candidates = (requested,) if requested != "AUTO" else ("NPU", "GPU", "CPU")
        for cand in candidates:
            if cand in devices:
                enc_dev = cand
                break
        if requested != "AUTO" and enc_dev != requested:
            print(f"  requested OpenVINO device {requested} unavailable; using {enc_dev}", flush=True)
        cache = ROOT / "ov-cache" / "parakeet"
        nets = {}
        plan = [
            ("mel", "parakeet_melspectogram.xml", "CPU"),
            ("encoder", "parakeet_encoder.xml", enc_dev),
            ("decoder", "parakeet_decoder.xml", "CPU"),
            ("joint", "parakeet_joint.xml", "CPU"),
        ]
        try:
            for key, fname, device in plan:
                print(f"  OV compile {key} -> {device}", flush=True)
                model = core.read_model(str(ov_dir / fname))
                (cache / f"{key}-{device}").mkdir(parents=True, exist_ok=True)
                nets[key] = core.compile_model(
                    model,
                    device,
                    {"CACHE_DIR": str(cache / f"{key}-{device}"), "PERFORMANCE_HINT": "LATENCY"},
                )
        except Exception as exc:
            print(f"  OpenVINO ASR failed ({exc}); sherpa CPU", flush=True)
            return False
        vocab_path = ov_dir / "parakeet_vocab.json"
        vocab = {int(k): v for k, v in json.loads(vocab_path.read_text(encoding="utf-8")).items()}
        self._ov = {"nets": nets, "vocab": vocab, "enc_dev": enc_dev}
        self.device = f"ov-{enc_dev.lower()}"
        print(f"  Parakeet encoder on {enc_dev}", flush=True)
        return True

    def decode(self, pcm: np.ndarray) -> tuple[str, dict]:
        if pcm.size < int(0.35 * SR):
            return "", {"audio_s": pcm.size / SR, "decode_s": 0.0, "rtf": 0.0, "peak": 0.0}
        pcm, peak = normalize_pcm(pcm)
        t0 = time.perf_counter()
        with self._lock:
            if self._ov is not None:
                text = self._decode_ov(pcm)
            else:
                stream = self.rec.create_stream()
                stream.accept_waveform(SR, pcm)
                self.rec.decode_stream(stream)
                text = _clean_transcript(stream.result.text or "")
        dt = time.perf_counter() - t0
        audio_s = pcm.size / SR
        quality = audio_quality_report(pcm)
        quality["peak_before_normalize"] = peak
        if self._utt_dump is not None and peak >= 1e-4:
            self._save_utt(pcm, text, quality)
        return text, {
            "audio_s": audio_s,
            "decode_s": dt,
            "rtf": dt / audio_s if audio_s else 0.0,
            "peak": peak,
            **quality,
        }

    def _save_utt(self, pcm: np.ndarray, text: str, quality: dict) -> None:
        try:
            self._utt_dump.mkdir(parents=True, exist_ok=True)
            self._utt_seq += 1
            stem = time.strftime("%Y%m%d-%H%M%S") + f"-{self._utt_seq:03d}"
            wav_path = self._utt_dump / f"{stem}.wav"
            txt_path = self._utt_dump / f"{stem}.txt"
            json_path = self._utt_dump / f"{stem}.json"
            pcm_i16 = np.clip(np.asarray(pcm, dtype=np.float32) * 32767.0, -32767, 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm_i16.tobytes())
            txt_path.write_text(text if text is not None else "", encoding="utf-8")
            json_path.write_text(
                json.dumps({"transcript": text or "", **quality}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"ASR_UTT_SAVED wav={wav_path} transcript={text!r}", flush=True)
        except Exception as exc:
            print(f"ASR_UTT_SAVE_FAILED {exc!r}", flush=True)

    def _decode_ov(self, pcm: np.ndarray) -> str:
        from .asr_ov import transcribe_ov

        text, _ = transcribe_ov(self._ov["nets"], pcm, self._ov["vocab"])
        return _clean_transcript(text)

    def warmup(self) -> None:
        self.decode(np.zeros(int(0.6 * SR), dtype=np.float32))


class EnergyVad:
    """Endpointing: Silero (real VAD) for speech vs noise; RMS only for echo/barge.

    Listen gate and TTS barge-in are independent. After speech_start, every PCM
    frame is kept and end-of-speech is Silero-not-speech for ~min_silence_s,
    not a re-application of the start gate.
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.silero = SileroSpeech(cfg.vad_path)
        self.preroll: list[np.ndarray] = []
        self.utt: list[np.ndarray] = []
        self.speech = False
        self.loud_n = 0
        self.quiet_n = 0
        self.speech_s = 0.0
        self._utt_peak = 0.0
        self.playing = False
        self.audible_playback = False
        self._echo_calibration_frames = 0
        self.ignore_s = 0.0
        self.preroll_n = max(1, int(cfg.asr_preroll_s / (WIN / SR)))
        self.barge_preroll_n = max(1, int(cfg.asr_barge_preroll_s / (WIN / SR)))
        self.noise_floor = 0.006
        self._cal: list[float] = []
        self._cal_need = max(8, int(cfg.noise_cal_s / (WIN / SR)))
        self._calibrated = False
        self._log_t = 0.0
        self._reject_t = 0.0
        self._peak_loud = 0
        self.echo_floor = 0.006
        self._echo_n = 0
        self._candidate_logged = False
        self._silero_skip = 0
        self._utt_frames = 0
        self._utt_speech_frames = 0
        self._candidate_started_at: float | None = None
        self._utterance_start_latency_ms = 0.0
        self._utterance_was_barge = False
        self._trigger_samples = 0
        self._retained_preroll_samples = 0
        print(
            "VAD thresholds "
            f"backend={'silero' if self.silero.ok else 'heuristic'} "
            f"listen_rms={cfg.speech_start_rms:.3f} listen_vad={cfg.vad_threshold:.2f} "
            f"listen_hold={cfg.speech_start_s:.2f}s floor_mult={cfg.noise_floor_mult:.1f} "
            f"min_silence={cfg.min_silence_s:.2f}s "
            f"barge_rms={cfg.barge_rms:.3f} barge_ratio={cfg.barge_echo_ratio:.2f} "
            f"barge_vad={cfg.barge_prob:.2f} barge_hold={cfg.barge_confirm_s:.2f}s",
            flush=True,
        )

    def set_playing(self, playing: bool, ignore: bool = True) -> None:
        was = self.playing
        self.playing = playing
        if playing and not was:
            self.echo_floor = max(self.noise_floor, 0.006)
            self._echo_n = 0
            self._candidate_logged = False
            self.audible_playback = False
            self._echo_calibration_frames = 0
            # Keep an in-progress human start-hold / utterance. Wiping loud_n
            # here is why a rapid second interrupt produced no STT: TTS start
            # erased the hold and then echo-holdoff blocked barge-in.
            if self.speech or self.loud_n > 0:
                print(
                    f"barge_overlap preserve speech={self.speech} loud_n={self.loud_n} "
                    f"preroll={len(self.preroll)}",
                    flush=True,
                )
            elif not self.speech:
                self.utt = []
        if not playing and ignore:
            self.ignore_s = self.cfg.post_tts_ignore_s
        if not playing:
            self.audible_playback = False
            self._echo_calibration_frames = 0

    def set_audible_playback(self, audible: bool) -> None:
        """Mark real browser playback, distinct from server-side TTS generation."""
        self.audible_playback = bool(audible and self.playing)
        self._echo_calibration_frames = 3 if self.audible_playback else 0
        print(
            f"vad_audible_playback={str(self.audible_playback).lower()} "
            f"echo_calibration_frames={self._echo_calibration_frames}",
            flush=True,
        )

    def reset(self) -> None:
        self.utt = []
        self.speech = False
        self.loud_n = 0
        self.quiet_n = 0
        self.speech_s = 0.0
        self._utt_peak = 0.0
        self._peak_loud = 0
        self._candidate_logged = False
        self._utt_frames = 0
        self._utt_speech_frames = 0
        self._silero_skip = 0
        self._candidate_started_at = None
        self._utterance_start_latency_ms = 0.0
        self._utterance_was_barge = False
        self.silero.reset()

    def recalibrate(self) -> None:
        """Re-estimate room noise when the user hits Start listening."""
        self._cal = []
        self._calibrated = False
        self.noise_floor = 0.006
        self.reset()
        print("VAD recalibrate on listen-start", flush=True)

    def _listen_on(self) -> float:
        if not self._calibrated:
            return self.cfg.speech_start_rms
        return max(self.cfg.speech_start_rms, self.noise_floor * self.cfg.noise_floor_mult + 0.008)

    def _barge_on(self) -> float:
        echo = max(self.echo_floor, self.noise_floor)
        # Independent of the strict listen gate so talking-over-TTS can fire.
        return max(self.cfg.barge_rms, echo * self.cfg.barge_echo_ratio + 0.008)

    def _gates(self, rms: float) -> tuple[float, float, float]:
        if self.playing:
            on = self._barge_on()
            echo = max(self.echo_floor, self.noise_floor)
            off = max(echo * 1.12, 0.018)
        else:
            on = self._listen_on()
            off = max(self.cfg.speech_hold_rms, self.noise_floor * 1.6)
        return on, off, 0.0

    def _frame_decision(self, rms: float, win: np.ndarray) -> tuple[float, bool, bool, float]:
        """vad_prob, energy_ok, vad_ok, energy_threshold.

        Listen: Silero is the speech detector. RMS is only a weak floor so we
        do not start on digital silence. Barge-in uses the echo-relative RMS
        gate so speaker playback cannot trigger just because Silero hears TTS.
        """
        on, _, _ = self._gates(rms)
        if self.silero.ok:
            # Exact digital silence poisons Silero's LSTM so the next noisy
            # onset (fan) is falsely classified as speech. Skip those frames
            # and reset after a short stretch of empty audio.
            if rms < 5e-4:
                vad = 0.0
                self._silero_skip += 1
                if self._silero_skip == 8:
                    self.silero.reset()
            else:
                self._silero_skip = 0
                vad = self.silero.prob(win)
            vad_ok = vad >= 0.5
            if self.playing:
                energy_ok = rms >= on
            else:
                on = max(0.008, self.noise_floor * 1.8)
                energy_ok = rms >= on
        else:
            vad = speech_score(win)
            need_prob = self.cfg.barge_prob if self.playing else self.cfg.vad_threshold
            energy_ok = rms >= on
            vad_ok = vad >= need_prob
        return vad, energy_ok, vad_ok, on

    def _update_echo(self, rms: float) -> None:
        """Track speaker-leak baseline while TTS is playing and we are not in a barge candidate."""
        if not self.playing or not self.audible_playback or self.loud_n > 2:
            return
        if self._echo_calibration_frames > 0:
            self.echo_floor = max(self.noise_floor, rms)
            self._echo_n = max(self._echo_n, 1)
            self._echo_calibration_frames -= 1
            return
        if self._echo_n < 12:
            self.echo_floor = (self.echo_floor * self._echo_n + rms) / (self._echo_n + 1)
            self._echo_n += 1
        else:
            self.echo_floor = 0.92 * self.echo_floor + 0.08 * min(rms, self.echo_floor * 1.6)

    def _update_noise(self, rms: float) -> None:
        if self.speech or self.playing:
            return
        self._cal.append(rms)
        if len(self._cal) > self._cal_need * 2:
            self._cal = self._cal[-self._cal_need :]
        if not self._calibrated and len(self._cal) >= self._cal_need:
            arr = np.asarray(self._cal, dtype=np.float32)
            self.noise_floor = float(np.percentile(arr, 60))
            self._calibrated = True
            print(
                f"VAD noise_floor={self.noise_floor:.4f} threshold={self._listen_on():.4f} "
                f"listen_vad={self.cfg.vad_threshold:.2f}",
                flush=True,
            )
        elif self._calibrated:
            # Slow adapt to room changes; never chase speech peaks.
            self.noise_floor = 0.97 * self.noise_floor + 0.03 * min(rms, self.noise_floor * 1.8)

    def push(self, pcm: np.ndarray) -> dict:
        pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        # Suppress starts during the short speaker-tail guard, but retain those
        # samples in pre-roll. The previous slice permanently clipped speech
        # that began immediately after playback.
        guarded_samples = min(pcm.size, int(max(0.0, self.ignore_s) * SR))
        self.ignore_s = max(0.0, self.ignore_s - pcm.size / SR)

        barge = False
        segments: list[np.ndarray] = []
        segment_reports: list[dict] = []
        rms_last = 0.0
        prob_last = 0.0
        i = 0
        hop = WIN / SR
        start_need = (
            max(5, math.ceil(self.cfg.barge_confirm_s / hop))
            if self.playing
            else max(8, math.ceil(self.cfg.speech_start_s / hop))
        )
        hang = max(16, int(self.cfg.min_silence_s / hop))
        failsafe = hang + max(2, int(0.05 / hop))
        while i + WIN <= pcm.size:
            win = pcm[i : i + WIN]
            i += WIN
            start_suppressed = i <= guarded_samples and not self.speech
            rms = float(np.sqrt(np.mean(win * win) + 1e-12))
            rms_last = rms
            if not start_suppressed:
                self._update_noise(rms)
            self._update_echo(rms)
            vad, energy_ok, vad_ok, on = self._frame_decision(rms, win)
            prob_last = vad
            above = energy_ok and vad_ok and not start_suppressed

            if not self.speech:
                if above:
                    self.loud_n += 1
                    if self.playing and self.loud_n == 1:
                        self._candidate_started_at = time.perf_counter()
                    self._peak_loud = max(self._peak_loud, self.loud_n)
                    self.quiet_n = 0
                else:
                    if self._peak_loud >= 4 and 0 < self.loud_n < start_need:
                        now = time.perf_counter()
                        if now - self._reject_t > 0.8:
                            self._reject_t = now
                            reasons = []
                            if not energy_ok:
                                reasons.append("low-rms")
                            if not vad_ok:
                                reasons.append("not-speech")
                            if self.loud_n < start_need:
                                reasons.append("hold-short")
                            kind = "barge_in_rejected" if self.playing else "rejected_noise"
                            print(
                                f"{kind} reason={'+'.join(reasons) or 'below-gate'} "
                                f"rms_ok={energy_ok} speech_score_ok={vad_ok} "
                                f"rms={rms:.4f} noise_floor={self.noise_floor:.4f} vad={vad:.2f} "
                                f"threshold={on:.4f} hold={self.loud_n * hop:.2f}s/{start_need * hop:.2f}s "
                                f"echo={self.echo_floor:.4f}",
                                flush=True,
                            )
                    # One missed frame must not wipe a nearly-complete start hold.
                    self.loud_n = max(0, self.loud_n - 1)
                    if self.loud_n == 0:
                        if self.playing and self._candidate_started_at is not None:
                            age_ms = (time.perf_counter() - self._candidate_started_at) * 1000.0
                            print(
                                f"barge_false_candidate duration_ms={age_ms:.1f} "
                                f"peak_frames={self._peak_loud} reason=confirmation-lost",
                                flush=True,
                            )
                            self._candidate_started_at = None
                        self._peak_loud = 0
                        self._candidate_logged = False
                    self.quiet_n += 1

                self.preroll.append(win)
                keep_n = self.barge_preroll_n if self.playing else self.preroll_n
                if len(self.preroll) > keep_n:
                    self.preroll = self.preroll[-keep_n:]
                if (not self.playing) and self.loud_n == 4 and not self._candidate_logged:
                    self._candidate_logged = True
                    print(
                        f"listen_candidate rms={rms:.4f} noise_floor={self.noise_floor:.4f} vad={vad:.2f} "
                        f"threshold={on:.4f} rms_ok={energy_ok} speech_score_ok={vad_ok} "
                        f"hold_ok=false hold={self.loud_n * hop:.2f}s/{start_need * hop:.2f}s",
                        flush=True,
                    )
                if self.playing and self.loud_n == 2 and not self._candidate_logged:
                    self._candidate_logged = True
                    print(
                        f"barge_candidate rms={rms:.4f} vad={vad:.2f} echo={self.echo_floor:.4f} "
                        f"noise_floor={self.noise_floor:.4f} threshold={on:.4f} "
                        f"barge_in_confirmation_duration={self.loud_n * hop:.2f}s",
                        flush=True,
                    )
                if self.loud_n >= start_need:
                    self.speech = True
                    self.utt = list(self.preroll)
                    n = int(sum(w.size for w in self.utt))
                    self._trigger_samples = n
                    self._retained_preroll_samples = max(0, n - min(n, self.loud_n * WIN))
                    self.preroll = []
                    self.speech_s = n / SR
                    self.quiet_n = 0
                    self._utt_peak = rms
                    self._utterance_was_barge = self.playing
                    if self._candidate_started_at is not None:
                        self._utterance_start_latency_ms = (
                            time.perf_counter() - self._candidate_started_at
                        ) * 1000.0
                    self._peak_loud = 0
                    self._candidate_logged = False
                    print(
                        f"speech_start rms={rms:.4f} vad={vad:.2f} samples={n} "
                        f"preroll_s="
                        f"{(self.cfg.asr_barge_preroll_s if self.playing else self.cfg.asr_preroll_s):.2f} "
                        f"playing={self.playing} "
                        f"backend={'silero' if self.silero.ok else 'heuristic'}",
                        flush=True,
                    )
                    if self.playing:
                        barge = True
                        print(
                            f"barge_confirmed rms={rms:.4f} vad={vad:.2f} echo={self.echo_floor:.4f} "
                            f"confirmation_ms={self._utterance_start_latency_ms:.1f}",
                            flush=True,
                        )
            else:
                # Utterance open: keep every PCM frame. Do not re-apply the
                # start gate. Close on Silero-not-speech (or RMS-near-floor
                # fallback) for min_silence_s, else max-utterance timeout.
                self.utt.append(win)
                self.speech_s += hop
                self._utt_peak = max(self._utt_peak, rms)
                self._utt_frames += 1
                if vad >= 0.5:
                    self._utt_speech_frames += 1
                if self.silero.ok:
                    silent = vad < 0.5
                else:
                    end_off = max(self.noise_floor * 2.5, self.cfg.speech_hold_rms, 0.014)
                    silent = rms < end_off
                if silent:
                    self.quiet_n += 1
                else:
                    self.quiet_n = 0
                timed_out = self.speech_s >= self.cfg.max_speech_s
                if self.quiet_n >= hang or self.quiet_n >= failsafe or timed_out:
                    why = "max-utterance" if timed_out else (
                        "silence-failsafe" if self.quiet_n >= failsafe else "silence"
                    )
                    body = np.concatenate(self.utt) if self.utt else win
                    dur = body.size / SR
                    frac = self._utt_speech_frames / max(self._utt_frames, 1)
                    print(
                        f"speech_end reason={why} utterance_duration={dur:.2f}s "
                        f"audio_samples={body.size} audio_bytes={body.nbytes} "
                        f"sample_rate={SR} audio_format=pcm_f32 "
                        f"quiet_s={self.quiet_n * hop:.2f} speech_frac={frac:.2f}",
                        flush=True,
                    )
                    if body.size < int(0.25 * SR):
                        print(
                            f"speech_end drop empty-or-tiny samples={body.size} duration={dur:.2f}s",
                            flush=True,
                        )
                    elif self.silero.ok and self._utt_speech_frames == 0:
                        print(
                            f"rejected_noise reason=no-speech-frames duration={dur:.2f}s",
                            flush=True,
                        )
                    else:
                        report = audio_quality_report(body)
                        report.update(
                            {
                                "vad_speech_start_latency_ms": self._utterance_start_latency_ms,
                                "barge_utterance": self._utterance_was_barge,
                                "speech_fraction": frac,
                                "vad_trigger_samples": self._trigger_samples,
                                "vad_retained_preroll_samples": self._retained_preroll_samples,
                                "vad_eos_samples": body.size,
                                "vad_retained_trailing_samples": self.quiet_n * WIN,
                                "vad_submitted_samples": body.size + int((0.18 + self.cfg.asr_tail_s) * SR),
                            }
                        )
                        segments.append(pad_utterance(body, 0.18, self.cfg.asr_tail_s))
                        segment_reports.append(report)
                    self.reset()
                    self.preroll = []
        if i < pcm.size and self.speech:
            self.utt.append(pcm[i:])
        now = time.perf_counter()
        interval = 0.45 if self.playing else 2.0
        if now - self._log_t > interval:
            self._log_t = now
            if self.playing:
                on = self._barge_on()
            elif self.silero.ok:
                on = max(0.008, self.noise_floor * 1.8)
            else:
                on = self._listen_on()
            hold = self.loud_n * hop
            if self.playing:
                print(
                    f"noise_floor={self.noise_floor:.4f} rms={rms_last:.4f} vad={prob_last:.2f} "
                    f"threshold={on:.4f} echo={self.echo_floor:.4f} candidate={self.loud_n > 0} "
                    f"hold={hold:.2f}s need={start_need * hop:.2f}s",
                    flush=True,
                )
            elif self.speech:
                print(
                    f"speech_open utterance_duration={self.speech_s:.2f}s quiet_s={self.quiet_n * hop:.2f} "
                    f"samples={int(sum(w.size for w in self.utt))} rms={rms_last:.4f} vad={prob_last:.2f}",
                    flush=True,
                )
            else:
                print(
                    f"listen rms={rms_last:.4f} noise_floor={self.noise_floor:.4f} vad={prob_last:.2f} "
                    f"threshold={on:.4f} rms_ok={rms_last >= on} speech_score_ok={prob_last >= 0.5} "
                    f"hold_ok={self.loud_n >= start_need} hold={hold:.2f}s/{start_need * hop:.2f}s "
                    f"cal={self._calibrated} backend={'silero' if self.silero.ok else 'heuristic'}",
                    flush=True,
                )
        return {
            "barge": barge,
            "segments": segments,
            "segment_reports": segment_reports,
            "rms": rms_last,
            "speech": self.speech,
            "prob": prob_last,
            "floor": self.noise_floor,
            "barge_candidate_at": self._candidate_started_at,
        }


def is_junk(text: str, audio_s: float) -> bool:
    t = (text or "").strip()
    if audio_s < 0.55:
        return True
    if len(t) < 6:
        return True
    if JUNK.match(t):
        return True
    return False


def _clean_transcript(text: str) -> str:
    text = " ".join((text or "").replace("▁", " ").split()).strip()
    return text
