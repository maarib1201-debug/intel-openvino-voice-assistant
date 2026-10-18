from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

# Must land before torch / pocket_tts import.
os.environ.setdefault("POCKET_TTS_NO_BEARTYPE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import TTS_SPEED, Settings
from .engine import Session, dump_ready, health_snapshot, load_models
from .tools import close_tool_clients

STATIC = Path(__file__).resolve().parent / "static"
settings = Settings()
models = None
_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _localhost_only(cfg: Settings) -> bool:
    return cfg.host.strip().lower() in _LOOPBACK_HOSTS


def _presented_token(headers, query_params) -> str:
    auth = (headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (headers.get("x-v2v-token") or query_params.get("token") or "").strip()


def _authorized(headers, query_params) -> bool:
    if _localhost_only(settings):
        return True
    expected = settings.auth_token or ""
    presented = _presented_token(headers, query_params)
    return bool(expected and presented and secrets.compare_digest(expected, presented))


def _origin_allowed(origin: str | None) -> bool:
    # Non-browser clients may omit Origin; bearer authentication still applies on LAN.
    if not origin:
        return True
    allowed = settings.allowed_origins
    if "*" not in allowed:
        return origin.rstrip("/") in {item.rstrip("/") for item in allowed}
    # The default wildcard is loopback-only, preventing a random website from
    # driving the localhost microphone WebSocket.
    if not _localhost_only(settings):
        return False
    try:
        return (urlsplit(origin).hostname or "").lower() in _LOOPBACK_HOSTS
    except ValueError:
        return False


class _MessageRate:
    def __init__(self, limit: int):
        self.limit = max(10, int(limit))
        self._times: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - 1.0
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        if len(self._times) >= self.limit:
            return False
        self._times.append(now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models = load_models(settings)
    print(
        "startup_diagnostics "
        + json.dumps(
            {
                "host": settings.host,
                "port": settings.port,
                "stt_device": models.asr.device,
                "llm_backend": models.llm.backend,
                "llm_requested_backend": getattr(models.llm, "backend_requested", "cpu"),
                "llm_actual_backend": getattr(models.llm, "actual_backend", models.llm.backend),
                "llm_gpu_offload": bool(models.llm.gpu_offload),
                "llm_gpu_verified": bool(getattr(models.llm, "gpu_verified", False)),
                "llm_gpu_device": getattr(models.llm, "gpu_device", None),
                "llm_layers_offloaded": int(getattr(models.llm, "layers_offloaded", 0)),
                "llm_model_path": str(settings.gguf_path) if settings.gguf_path.exists() else None,
                "tts": "ready" if models.tts.model is not None else "degraded",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        yield
    finally:
        with _sessions_lock:
            live_sessions = list(_sessions.values())
            _sessions.clear()
        for session in live_sessions:
            session.close()
        for gate_name in ("qwen_gate", "stt_gate", "tts_gate", "needle_gate"):
            gate = getattr(models, gate_name, None)
            if gate is not None:
                gate.close()
        close_tool_clients()
        models = None


app = FastAPI(title="v2v", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def require_lan_auth(request: Request, call_next):
    if not _authorized(request.headers, request.query_params):
        return JSONResponse(
            {"detail": "authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/ready")
async def ready():
    if models is None:
        return JSONResponse({"ready": False}, status_code=503)
    return JSONResponse(json.loads(dump_ready(models)))


@app.get("/health")
async def health():
    snapshot = health_snapshot(models)
    with _sessions_lock:
        live = [sid for sid, sess in _sessions.items() if not sess.closed]
    snapshot["active_session"] = bool(live)
    snapshot["active_sessions"] = len(live)
    snapshot["bind"] = {
        "host": settings.host,
        "localhost_only": _localhost_only(settings),
        "authentication_required": not _localhost_only(settings),
        "origin_validation": True,
    }
    return JSONResponse(snapshot, status_code=200 if snapshot["status"] in {"ok", "degraded"} else 503)


@app.websocket("/ws")
async def ws_voice(ws: WebSocket):
    if not _authorized(ws.headers, ws.query_params):
        await ws.close(code=4401, reason="authentication required")
        return
    if not _origin_allowed(ws.headers.get("origin")):
        await ws.close(code=4403, reason="origin not allowed")
        return
    await ws.accept()
    if models is None:
        await ws.send_text(json.dumps({"type": "error", "text": "models still loading"}))
        await ws.close()
        return
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue(maxsize=256)
    inbound_rate = _MessageRate(settings.max_ws_messages_per_second)

    def emit(msg) -> None:
        def _put() -> None:
            is_audio = isinstance(msg, tuple) and msg[0] == "audio"
            try:
                out.put_nowait(msg)
                return
            except asyncio.QueueFull:
                pass
            if is_audio:
                print("ws_queue_full DROPPED audio chunk", flush=True)
                return
            # Never drop user/bot/error: evict queued audio first.
            dropped = 0
            kept = []
            while True:
                try:
                    item = out.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, tuple) and item[0] == "audio":
                    dropped += 1
                else:
                    kept.append(item)
            for item in kept:
                try:
                    out.put_nowait(item)
                except asyncio.QueueFull:
                    print("ws_queue_full DROPPED kept-control while compacting", flush=True)
                    traceback.print_exc()
                    break
            try:
                out.put_nowait(msg)
                if dropped:
                    kind = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
                    print(f"ws_queue evicted {dropped} audio to send type={kind}", flush=True)
            except asyncio.QueueFull:
                kind = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
                print(f"ws_queue_full DROPPED type={kind}", flush=True)

        try:
            loop.call_soon_threadsafe(_put)
        except Exception:
            print("emit call_soon_threadsafe failed", flush=True)
            traceback.print_exc()

    session_id = uuid.uuid4().hex
    session = Session(models, emit, session_id=session_id)
    with _sessions_lock:
        _sessions[session_id] = session
    print(f"websocket_connected session_id={session_id} active_sessions={len(_sessions)}", flush=True)
    ready_payload = json.loads(dump_ready(models))
    ready_payload["session_id"] = session_id
    await ws.send_text(json.dumps(ready_payload))

    async def sender():
        while True:
            msg = await out.get()
            if msg is None:
                return
            try:
                if isinstance(msg, tuple) and msg[0] == "audio":
                    print(
                        "E2E_EVENT "
                        + json.dumps(
                            {
                                "event": "audio_sent_to_browser",
                                "mono": time.perf_counter(),
                                "wall_ns": time.time_ns(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    await ws.send_bytes(msg[1])
                else:
                    await ws.send_text(json.dumps(msg, default=str))
            except Exception:
                traceback.print_exc()
                return

    send_task = asyncio.create_task(sender())
    try:
        while True:
            packet = await ws.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            data = packet.get("bytes")
            if data is not None:
                if len(data) > settings.max_pcm_chunk_bytes:
                    await ws.send_text(json.dumps({"type": "error", "text": "audio message too large"}))
                    await ws.close(code=1009, reason="audio message too large")
                    break
                if not inbound_rate.allow():
                    await ws.send_text(
                        json.dumps({"type": "busy", "reason": "message-rate-limit", "retry_after_ms": 1000})
                    )
                    await ws.close(code=1008, reason="message rate limit")
                    break
                session.push_pcm16(data)
                continue
            text = packet.get("text")
            if not text:
                continue
            if len(text.encode("utf-8")) > settings.max_ws_text_bytes:
                await ws.send_text(json.dumps({"type": "error", "text": "text message too large"}))
                await ws.close(code=1009, reason="text message too large")
                break
            if not inbound_rate.allow():
                await ws.send_text(
                    json.dumps({"type": "busy", "reason": "message-rate-limit", "retry_after_ms": 1000})
                )
                await ws.close(code=1008, reason="message rate limit")
                break
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            typ = msg.get("type")
            if typ == "playback_telemetry":
                telemetry = {
                    "event": msg.get("event"),
                    "session_id": msg.get("session_id") or session_id,
                    "request_id": msg.get("request_id"),
                    "utterance_id": msg.get("utterance_id"),
                    "generation_id": msg.get("generation_id"),
                    "client_mono_ms": msg.get("client_mono_ms"),
                    "data": msg.get("data") or {},
                }
                print("BROWSER_PLAYBACK_EVENT " + json.dumps(telemetry, sort_keys=True), flush=True)
                session.playback_telemetry(
                    str(telemetry.get("event") or ""), telemetry.get("generation_id"), telemetry.get("data")
                )
            elif typ == "listen":
                session.set_listen(bool(msg.get("enabled")), reason=str(msg.get("reason") or ""))
            elif typ == "barge":
                if not session.listening_enabled:
                    print("barge ignored listening_enabled=false", flush=True)
                    continue
                src = msg.get("source") or "button"
                session.interrupt("user-button" if src == "button" else src)
            elif typ == "playback_done":
                session.playback_done(msg.get("generation_id"))
            elif typ == "reset":
                session.interrupt("reset")
                session.reset_history()
    except WebSocketDisconnect:
        print(f"websocket_disconnected session_id={session_id}", flush=True)
    except Exception as exc:
        print(f"websocket_error session_id={session_id} error={type(exc).__name__}", flush=True)
    finally:
        session.close()
        with _sessions_lock:
            if _sessions.get(session_id) is session:
                _sessions.pop(session_id, None)
        print(f"websocket_cleanup session_id={session_id} active_sessions={len(_sessions)}", flush=True)
        try:
            out.put_nowait(None)
        except Exception:
            pass
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)


def parse_args(argv: list[str] | None = None) -> Settings:
    p = argparse.ArgumentParser(description="Ultra-low-latency Parakeet + tiny LLM + Pocket TTS server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--llm", default="qwen08", help="qwen08 | lfm350 | gemma270 | qwen08-agent | HF id")
    p.add_argument("--voice", default="alba")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--asr-threads", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--no-quantize-tts", action="store_true")
    p.add_argument("--min-silence", type=float, default=0.65)
    p.add_argument("--tts-speed", type=float, default=None, help="Playback speed, e.g. 1.0 1.1 1.15 1.2")
    p.add_argument(
        "--allow-lan",
        action="store_true",
        help="Allow non-localhost binding; requires V2V_AUTH_TOKEN and explicit V2V_ALLOWED_ORIGINS",
    )
    args = p.parse_args(argv)
    cfg = Settings(
        host=args.host,
        port=args.port,
        llm=args.llm,
        voice=args.voice,
        cpu_threads=args.cpu_threads,
        asr_threads=args.asr_threads,
        max_new_tokens=args.max_tokens,
        quantize_tts=not args.no_quantize_tts,
        min_silence_s=args.min_silence,
        tts_speed=float(args.tts_speed) if args.tts_speed is not None else TTS_SPEED,
        allow_lan=args.allow_lan,
    )
    if not _localhost_only(cfg):
        if not cfg.allow_lan:
            p.error("non-localhost binding requires --allow-lan")
        if not cfg.auth_token:
            p.error("non-localhost binding requires V2V_AUTH_TOKEN")
        if not cfg.allowed_origins or "*" in cfg.allowed_origins:
            p.error("non-localhost binding requires explicit V2V_ALLOWED_ORIGINS")
    os.environ["OMP_NUM_THREADS"] = str(cfg.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cfg.cpu_threads)
    return cfg


def main() -> None:
    global settings
    settings = parse_args()
    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn missing")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
