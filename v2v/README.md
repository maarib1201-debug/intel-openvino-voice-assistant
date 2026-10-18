# v2v — local real-time voice assistant

Local duplex cascade:

**Browser microphone** → **Silero VAD** → **Parakeet TDT 0.6B v2** → **deterministic tools / Needle fallback** → **Qwen3.5-0.8B GGUF or CPU fallback** → **Pocket TTS streaming** → **browser playback**

Parakeet's OpenVINO encoder uses the NPU when available. Its mel frontend, decoder, and joint remain on CPU. The normal launcher prefers the vendored Vulkan runtime, verifies actual Intel Arc offload from native llama.cpp startup diagnostics, and explicitly falls back to CPU if verification fails.

The thing people actually hear is **time-to-first-audio (TTFA)**, not full-reply RTF. This server is built around that.

## Run

From `voice-edge-probe` (the venv that already has Parakeet / Pocket TTS):

```bat
run_v2v.bat
```

Strict Vulkan verification (fail instead of falling back):

```bat
run_v2v_gpu_test.bat
```

CPU-only fallback:

```bat
run_v2v_cpu.bat
```

or:

```bat
.venv\Scripts\python.exe -m v2v --llm lfm350 --port 8088
```

Open http://127.0.0.1:8088 — tap **Start listening**, then talk. No push-to-talk.

| flag | default | why |
|---|---|---|
| `--llm qwen08` | Qwen3.5-0.8B GGUF | default small conversational model |
| `--llm lfm350` | LiquidAI/LFM2.5-350M | faster, much weaker at open questions |
| `--llm gemma270` | gemma-3-270m-it | smaller, not smarter |
| `--min-silence 0.32` | 320 ms | VAD endpoint. Lower = snappier, more cut-offs |
| `--cpu-threads 4` | 4 | leave cores for Parakeet |
| `--no-quantize-tts` | off | Pocket TTS `quantize=True` is on by default |
| `--allow-lan` | off | non-loopback only; also requires `V2V_AUTH_TOKEN` and explicit `V2V_ALLOWED_ORIGINS` |

Operational endpoints:

- `/ready` reports loaded model/backend readiness and selected runtime settings.
- `/health` reports STT/NPU, requested and actual Qwen backend, verified Vulkan device/layers, queue occupancy, RAG chunk count, Needle, TTS, tools, and active-session count. It does not expose prompts or transcripts.

The service binds to localhost by default. Local browser use needs no token. For an internal LAN bind, set a strong environment token and exact browser origins before launching:

```powershell
$env:V2V_AUTH_TOKEN = '<random-secret>'
$env:V2V_ALLOWED_ORIGINS = 'https://voice.internal.example'
.\.venv\Scripts\python.exe -m v2v --host 0.0.0.0 --allow-lan
```

HTTP clients should send `Authorization: Bearer <token>`. The browser can open `https://voice.internal.example/?token=<token>`; the page passes it to the WebSocket. Terminate TLS at an authenticated internal reverse proxy, restrict the network, and do not expose this service directly to the internet. LAN startup fails closed if the token or explicit origins are missing. WebSocket origin, message size, audio chunk size, message rate, and bounded inference queues are enforced. Sensitive transcript/model text logging stays off unless `V2V_LOG_SENSITIVE=1` is deliberately set.

## Verified Vulkan packaging

The coherent `llama_cpp` Python package and matching `ggml*.dll`, `llama.dll`, and `ggml-vulkan.dll` set live in `models/vulkan_runtime`. Do not mix DLLs from another llama.cpp build. GPU status is accepted only when captured startup logs identify Intel Arc 140V, Vulkan0, full layer offload, and zero layers assigned to CPU. `n_gpu_layers=-1` alone is not evidence. `run_v2v.bat` uses verified Vulkan when available and otherwise reports the reason and uses CPU; `run_v2v_gpu_test.bat` is strict.

## Company knowledge

Put non-confidential pilot documents in `company_knowledge/`, or point `V2V_KNOWLEDGE_DIR` at another local directory. `.txt`, `.md`, and `.json` are indexed at process startup, so restart the server after changing files. `/health` reports the loaded chunk count. Retrieval events identify source filenames and chunk IDs without logging retrieved text. Verify a new corpus with known questions, paraphrases, unrelated questions, and hostile instructions embedded in documents; retrieved text is wrapped as untrusted data. Production remains word plus character 3-gram TF-IDF. Reconsider the measured MiniLM option only if a representative real corpus shows inadequate lexical recall.

## Why it feels fast

Patterns used by Pipecat / LiveKit voice agents, Moshi-style duplex, and on-device cascades:

1. **Never wait for the full reply before speaking.** LLM tokens flush into Pocket TTS at the first clause (`SentenceBuffer`), not at EOS.
2. **Stream TTS chunks.** Pocket TTS `generate_audio_stream` yields audio as it decodes. The browser starts playback on the **first chunk** with a 20 ms jitter buffer, not a 500 ms preroll.
3. **Silero VAD endpointing**, not "press enter when done". 320 ms of silence ends the utterance. Default Silero 500 ms feels slow.
4. **Parakeet is offline-but-cheap.** RTF is tiny, so we re-decode the growing buffer every ~450 ms for live partials, then a final decode on VAD pop. No streaming transducer required.
5. **Duplex + barge-in.** Mic stays open while the bot talks (browser `echoCancellation: true`). ~380 ms of real speech cancels TTS/LLM immediately. Short blips during playback are treated as echo and dropped.
6. **Post-TTS ignore window (160 ms)** so the tail of the loudspeaker doesn't become a fake user turn.
7. **Warmup** of STT, LLM, and TTS on boot so the first real turn isn't a cold-start.
8. **Short spoken system prompt + bounded generation.** Voice models should not write essays.
9. **Thread split.** ASR 2 threads, Torch 4, no oversubscription. `POCKET_TTS_NO_BEARTYPE=1` kills per-call type checks.
10. **INT8 Parakeet + optional quantized Pocket TTS.** The 350M LLM is the right size: 270M is faster and dumber, 1B+ adds TTFA for little voice-quality gain on CPU.

## What not to do

- Native speech-to-speech (LFM2.5-Audio etc.) is simpler but **first audio is slower** — it can't speak the first sentence while the rest is still generating.
- Don't play a fake "um" filler. Make TTFA actually small.
- Don't overlap two Torch generators on a 4-core machine if RTF tanks; sentence-level TTS already hides LLM tail latency.

## Metrics

The HUD shows:

- **TTFA** — VAD endpoint → first PCM chunk (this is the number that matters)
- **STT RTF** — Parakeet decode / audio duration (want ≪ 1, typically ~0.05–0.15)
- **LLM s** — full generation, but speech started much earlier
- **TTS RTF** — generate time / audio duration (want ≪ 1 so the speaker never underruns)
