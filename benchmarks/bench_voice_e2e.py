"""Replay saved microphone captures through the real v2v WebSocket path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import threading
import time
import wave
from pathlib import Path

import psutil
from websockets.asyncio.client import connect


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    name: ROOT / "benchmarks" / "conversational_audio" / f"{name}.wav"
    for name in (
        "tell_me_a_joke",
        "sky_is_blue",
        "black_holes",
        "productivity_tip",
        "kubernetes_simple",
    )
}


def read_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise ValueError(f"Expected mono 16-bit 16 kHz PCM: {path}")
        return wav.readframes(wav.getnframes()), wav.getframerate()


class UsageSampler:
    def __init__(self, root: psutil.Process):
        self.root = root
        self.processes: dict[int, psutil.Process] = {}
        for process in [root, *root.children(recursive=True)]:
            self.processes[process.pid] = process
            process.cpu_percent(None)

    def sample(self) -> tuple[float | None, float | None]:
        try:
            children = self.root.children(recursive=True)
        except psutil.Error:
            children = []
        for process in [self.root, *children]:
            self.processes[process.pid] = process
        cpu = 0.0
        rss = 0
        valid_cpu = False
        for process in list(self.processes.values()):
            try:
                value = process.cpu_percent(None)
                cpu += value
                valid_cpu = valid_cpu or value > 0
                rss += process.memory_info().rss
            except psutil.Error:
                self.processes.pop(process.pid, None)
        return (cpu if valid_cpu else None), (rss / (1024 * 1024) if rss else None)


def process_usage(process: psutil.Process) -> tuple[float | None, float | None]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    cpu = 0.0
    rss = 0
    for child in processes:
        try:
            cpu += child.cpu_percent(None)
            rss += child.memory_info().rss
        except psutil.Error:
            pass
    return (cpu if cpu > 0 else None), (rss / (1024 * 1024) if rss else None)


class Server:
    def __init__(self, launcher: Path, port: int, log_path: Path):
        self.launcher = launcher
        self.port = port
        self.log_path = log_path
        self.proc: subprocess.Popen[str] | None = None
        self.events: list[dict] = []
        self.browser_events: list[dict] = []
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log_path.open("w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            ["cmd", "/c", str(self.launcher), "--port", str(self.port)],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        def read_events() -> None:
            seen_lines = 0
            while self.proc is not None and self.proc.poll() is None:
                try:
                    lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    time.sleep(0.1)
                    continue
                for line in lines[seen_lines:]:
                    if line.startswith("E2E_EVENT "):
                        try:
                            self.events.append(json.loads(line[10:]))
                        except json.JSONDecodeError:
                            pass
                    elif line.startswith("BROWSER_PLAYBACK_EVENT "):
                        try:
                            self.browser_events.append(json.loads(line[23:]))
                        except json.JSONDecodeError:
                            pass
                seen_lines = len(lines)
                time.sleep(0.1)

        self._reader = threading.Thread(target=read_events, daemon=True)
        self._reader.start()

    async def wait_ready(self) -> dict:
        deadline = time.perf_counter() + 180
        uri = f"ws://127.0.0.1:{self.port}/ws"
        while time.perf_counter() < deadline:
            try:
                ws = await connect(uri, open_timeout=2)
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                await ws.close()
                return ready
            except Exception:
                await asyncio.sleep(0.5)
        raise TimeoutError(f"Server did not become ready; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        self.proc = None


async def run_case(server: Server, case: str, wav_path: Path, iteration: int, warmup: bool) -> dict:
    pcm, sample_rate = read_pcm(wav_path)
    process = psutil.Process(server.proc.pid)  # type: ignore[union-attr]
    sampler = UsageSampler(process)
    event_start = len(server.events)
    browser_event_start = len(server.browser_events)
    ws = await connect(f"ws://127.0.0.1:{server.port}/ws", open_timeout=5)
    ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    await ws.send(json.dumps({"type": "listen", "enabled": True, "reason": "benchmark"}))
    await asyncio.sleep(0.4)
    process.cpu_percent(None)
    started = time.perf_counter()
    first_audio_received = None
    audio_end = None
    transcript = None
    metrics = None
    usage: list[tuple[float | None, float | None]] = []
    chunk_bytes = 1024 * 2
    replay_pcm = pcm + b"\x00" * sample_rate * 2
    for offset in range(0, len(replay_pcm), chunk_bytes):
        await ws.send(replay_pcm[offset : offset + chunk_bytes])
        await asyncio.sleep(1024 / sample_rate)
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            usage.append(sampler.sample())
            continue
        usage.append(sampler.sample())
        now = time.perf_counter()
        if isinstance(message, bytes):
            if first_audio_received is None:
                first_audio_received = now
            continue
        msg = json.loads(message)
        if msg.get("type") == "user":
            transcript = msg.get("text")
        elif msg.get("type") == "audio_end":
            audio_end = now
            await ws.send(json.dumps({"type": "playback_done"}))
        elif msg.get("type") == "metrics":
            metrics = msg
            break
    await ws.close()
    await asyncio.sleep(0.3)
    relevant = server.events[event_start:]
    by_event = {}
    for event in relevant:
        event_name = event.get("event")
        if event_name == "audio_sent_to_browser":
            by_event.setdefault(event_name, event)
        else:
            by_event[event_name] = event
    route_tool = by_event.get("routing_complete", {}).get("tool")
    valid_qwen = bool(transcript and route_tool is None and by_event.get("first_llm_token"))
    event_times = {name: event.get("mono") for name, event in by_event.items()}
    first_token = event_times.get("first_llm_token")
    generation_complete = event_times.get("qwen_generation_complete")
    result = {
        "case": case,
        "iteration": iteration,
        "warmup": warmup,
        "wav": str(wav_path),
        "ready": ready,
        "transcript": transcript,
        "metrics": metrics,
        "valid_qwen": valid_qwen,
        "route_tool": route_tool,
        "speech_end_to_audio_received_s": (first_audio_received - event_times["speech_end"]) if first_audio_received and event_times.get("speech_end") else None,
        "audio_received_to_audio_end_s": (audio_end - first_audio_received) if audio_end and first_audio_received else None,
        "stt_latency_s": event_times.get("stt_ready", 0) - event_times.get("speech_end", 0) if event_times.get("stt_ready") and event_times.get("speech_end") else None,
        "routing_latency_s": event_times.get("routing_complete", 0) - event_times.get("stt_ready", 0) if event_times.get("routing_complete") and event_times.get("stt_ready") else None,
        "qwen_ttft_s": first_token - event_times["routing_complete"] if first_token and event_times.get("routing_complete") else None,
        "stt_ready_to_first_qwen_token_s": first_token - event_times["stt_ready"] if first_token and event_times.get("stt_ready") else None,
        "speech_end_to_first_tts_audio_s": event_times.get("first_tts_audio", 0) - event_times.get("speech_end", 0) if event_times.get("first_tts_audio") and event_times.get("speech_end") else None,
        "speech_end_to_first_browser_audio_s": event_times.get("audio_sent_to_browser", 0) - event_times.get("speech_end", 0) if event_times.get("audio_sent_to_browser") and event_times.get("speech_end") else None,
        "text_ready_to_first_tts_audio_s": event_times.get("first_tts_audio", 0) - event_times.get("first_final_text_to_tts", 0) if event_times.get("first_tts_audio") and event_times.get("first_final_text_to_tts") else None,
        "first_tts_audio_to_browser_s": event_times.get("audio_sent_to_browser", 0) - event_times.get("first_tts_audio", 0) if event_times.get("audio_sent_to_browser") and event_times.get("first_tts_audio") else None,
        "total_response_latency_s": event_times.get("final_response_complete", 0) - event_times.get("speech_end", 0) if event_times.get("final_response_complete") and event_times.get("speech_end") else None,
        "generated_tokens": by_event.get("qwen_generation_complete", {}).get("generated_tokens"),
        "qwen_decode_tps": by_event.get("qwen_generation_complete", {}).get("generated_tokens") / (generation_complete - first_token) if generation_complete and first_token and generation_complete > first_token else None,
        "cpu_percent_avg": sum(item[0] for item in usage if item[0] is not None) / len([item for item in usage if item[0] is not None]) if any(item[0] is not None for item in usage) else None,
        "ram_mib_max": max((item[1] for item in usage if item[1] is not None), default=None),
        "server_events": by_event,
        "browser_events": server.browser_events[browser_event_start:],
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


async def run_mode(name: str, launcher: Path, port: int, cases: dict[str, Path], iterations: int, output: Path) -> None:
    server = Server(launcher, port, ROOT / "benchmarks" / f"voice_{name}.log")
    server.start()
    try:
        ready = await server.wait_ready()
        print(json.dumps({"mode": name, "ready": ready}, ensure_ascii=True), flush=True)
        all_results = []
        warmup_case = next(iter(cases.items()))
        await run_case(server, warmup_case[0], warmup_case[1], 0, True)
        for case, wav_path in cases.items():
            for iteration in range(1, iterations + 1):
                all_results.append(await run_case(server, case, wav_path, iteration, False))
        with output.open("a", encoding="utf-8") as handle:
            for result in all_results:
                result["mode"] = name
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    finally:
        server.stop()


async def main_async(args: argparse.Namespace) -> None:
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    cases = {name: path for name, path in CASES.items() if path.exists()}
    missing = [name for name, path in CASES.items() if not path.exists()]
    print(json.dumps({"skipped_exact_voice_cases": missing}, ensure_ascii=True), flush=True)
    for mode, launcher, port in (
        ("cpu", ROOT / "run_v2v.bat", args.cpu_port),
        ("vulkan_full_npu", ROOT / "run_v2v_gpu_test.bat", args.gpu_port),
    ):
        await run_mode(mode, launcher, port, cases, args.iterations, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cpu-port", type=int, default=8090)
    parser.add_argument("--gpu-port", type=int, default=8091)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "voice_e2e_results.jsonl")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()