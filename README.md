# Intel OpenVINO Voice Assistant

This repository contains the recovered Windows implementation from the local `voice-edge-probe` project. It is a browser-based, full-duplex voice assistant with local speech recognition, language-model inference, tools and retrieval, and speech synthesis.

## Architecture

```mermaid
flowchart LR
  Mic[Browser microphone] -->|16 kHz PCM over WebSocket| VAD[Silero VAD\nCPU]
  VAD --> ASR[Parakeet TDT 0.6B v2\nOpenVINO: encoder NPU\nmel, decoder, joint CPU]
  ASR --> Route[STT correction\ntools and local RAG]
  Route --> LLM[Qwen3.5 0.8B\nllama.cpp: Vulkan Arc\nor CPU fallback]
  LLM --> TTS[Pocket TTS\nCPU, streamed chunks]
  TTS -->|PCM over WebSocket| Browser[Browser playback\nbarge-in supported]
```

The FastAPI service serves the browser UI and WebSocket endpoint. Calculator and time tools run locally; weather uses Open-Meteo. Optional local knowledge is indexed from `.txt`, `.md`, and `.json` files. Needle retrieval is optional and requires its separately installed runtime.

| Stage | Runtime | Target |
| --- | --- | --- |
| Voice activity detection | sherpa-onnx / Silero | CPU |
| Speech recognition | OpenVINO Parakeet TDT | Encoder on NPU; frontend, decoder, and joint on CPU |
| Language model | llama.cpp with Qwen3.5-0.8B GGUF | Arc GPU through Vulkan, or CPU fallback |
| Speech generation | Pocket TTS | CPU, streamed to the browser |

## Windows setup

Use 64-bit Windows 11 and Python 3.12. Open PowerShell in this directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Place or export the required model assets under `models/` (model weights are intentionally not tracked):

- `silero_vad.onnx`
- `parakeet-tdt-0.6b-v2-ov/` with the four OpenVINO IR graphs and vocabulary listed below
- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/` for CPU fallback
- `qwen3.5-0.8b-gguf/Qwen3.5-0.8B-Q8_0.gguf` for llama.cpp (Apache-2.0 model, downloaded from the pinned `ggml-org` conversion); alternatively, the Transformers fallback downloads `Qwen/Qwen3.5-0.8B`

To fetch the published Parakeet OpenVINO graphs, sherpa-onnx CPU fallback, and Silero VAD at pinned upstream revisions, run `.\prepare_models.ps1`. Use `-DryRun` to verify the Hub snapshots without downloading their ~1.9 GB weights. This retrieves existing artifacts; it does not run a conversion. The exact revisions and licenses are recorded in `models/README.md`. The original export scripts and complete exporter environment were not recovered, so regenerating identical split graphs from the NeMo checkpoint is not presently reproducible.

Then start the default Vulkan-preferred launcher:

```powershell
.\run_v2v.bat
```

Open `http://127.0.0.1:8088` and select **Start listening**. The default server binds to loopback. Use `run_v2v_cpu.bat` to force CPU inference, or `run_v2v_gpu_test.bat` to require verified Vulkan offload and NPU selection. The normal launcher permits a CPU fallback when Vulkan verification fails.

Set model locations without editing source code:

```powershell
$env:V2V_MODEL_DIR = (Resolve-Path .\models).Path
$env:V2V_QWEN_GGUF = (Resolve-Path .\models\qwen3.5-0.8b-gguf\Qwen3.5-0.8B-Q8_0.gguf).Path
```

Download the documented GGUF from ggml-org's Qwen3.5 conversion, pinned to the verified repository revision:

```powershell
& .\.venv\Scripts\python.exe -c "from huggingface_hub.cli.hf import main; main()" download ggml-org/Qwen3.5-0.8B-GGUF `
  --revision 8fea620810c4afa23dd6443f999a48574c1611a3 `
  --local-dir .\models\qwen3.5-0.8b-gguf Qwen3.5-0.8B-Q8_0.gguf
```

The model is based on the official `Qwen/Qwen3.5-0.8B` checkpoint. The GGUF repository declares Apache-2.0 and identifies the source checkpoint. The historical Q5_K_M file used in the saved benchmarks has unknown publisher and conversion provenance; it is not needed for this setup and should not be redistributed as part of this project.

For LAN use, pass `--host 0.0.0.0 --allow-lan` and set a strong `V2V_AUTH_TOKEN` plus exact `V2V_ALLOWED_ORIGINS`; the server rejects an open wildcard origin for non-loopback binds. Keep TLS and network access behind an authenticated reverse proxy.

## Model paths and hardware

The validated target was a Windows 11 machine with Intel Core Ultra 7 258V (Lunar Lake), 31.5 GiB RAM, Intel Arc 140V GPU, Intel NPU, OpenVINO devices `CPU/GPU/NPU`, and Vulkan 1.4.323 driver 101.7026. This is a known-good configuration, not a minimum specification. CPU fallback is available without an NPU or Vulkan GPU, at lower speed. `prepare_models.ps1` downloads about 1.9 GB of Parakeet weights; the historical Qwen Q5 GGUF is about 590 MB. Keep additional disk space for the Python environment, model caches, and compiled OpenVINO cache.

### Parakeet / OpenVINO

The runtime loads these files from `models/parakeet-tdt-0.6b-v2-ov/`:

- `parakeet_melspectogram.xml`
- `parakeet_encoder.xml`
- `parakeet_decoder.xml`
- `parakeet_joint.xml`
- `parakeet_vocab.json`

At startup, `v2v/asr.py` checks OpenVINO's available devices and compiles the encoder to NPU when available (`V2V_ASR_DEVICE=NPU` forces that choice); the mel frontend, decoder, and joint compile for CPU. The compiled-model cache goes under `ov-cache/parakeet`. If the OpenVINO IR is absent or cannot compile, startup falls back to the sherpa-onnx INT8 CPU model.

The recovered runtime does **not** include a complete, reproducible Parakeet-to-OpenVINO export pipeline. The published upstream IR is the supported preparation path; the four graph files are not tracked here. The local card and upstream `FluidInference/eddy` repository identify FluidInference as the converter. The published artifacts are CC BY 4.0, as is the NVIDIA base model. Preserve attribution and indicate changes if redistributing adapted model files. OpenVINO's generic export tools are not claimed to reproduce these NeMo split graphs.

OpenVINO compiles a model by reading the IR and calling `compile_model` with a device such as `NPU`; see [OpenVINO's NPU guide](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html). This application uses that runtime path rather than converting on each launch.

### Qwen / Vulkan

The reproducible Transformers fallback is `Qwen/Qwen3.5-0.8B` pinned to revision `2fc06364715b967f1860aea9cf38778875588b17` (override with `V2V_QWEN_REVISION`). The default GGUF is `ggml-org/Qwen3.5-0.8B-GGUF` at revision `8fea620810c4afa23dd6443f999a48574c1611a3`, file `Qwen3.5-0.8B-Q8_0.gguf`; it is a published conversion of that Qwen checkpoint. The historical Q5_K_M file used for benchmark data has unknown publisher/revision and conversion provenance, so the historical results cannot be reproduced bit-for-bit from it. The runtime tries the configured GGUF through `llama-cpp-python`, verifies native Vulkan startup diagnostics identify Intel Arc 140V and full model-layer offload with no CPU-assigned layers, then falls back to CPU if permitted. `n_gpu_layers=-1` by itself is not treated as proof of GPU use.

Vulkan-enabled `llama-cpp-python` is an optional separately built dependency. Install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) and Visual Studio Build Tools with the C++ workload first. `setup.ps1` installs pinned Ninja 1.13.2; CMake must be on `PATH`. From the repository root, build with:

```powershell
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_ARGS = '-DGGML_NATIVE=off -DGGML_VULKAN=on'
$env:FORCE_CMAKE = '1'
$env:PATH = (Resolve-Path .\.venv\Scripts).Path + ';' + $env:PATH
$env:TEMP = 'C:\tmp'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python==0.3.35
vulkaninfo --summary
```

Use a short temporary path because the source archive contains deeply nested paths that can exceed Windows path limits during extraction. This follows the upstream [Vulkan Windows wheel build workflow](https://github.com/abetlen/llama-cpp-python/blob/main/.github/workflows/build-wheels-vulkan.yaml). Verify actual offload with `run_v2v_gpu_test.bat`; compilation alone does not verify the active device. CPU-only install and Vulkan build are optional and separate from `setup.ps1`.

### VAD, audio, and speech synthesis

Silero VAD runs on CPU through sherpa-onnx. Browser JavaScript captures and resamples microphone audio to 16 kHz mono PCM, sends it over the local WebSocket, and plays streamed Pocket TTS audio. Pocket TTS runs on CPU through PyTorch; the selected voice defaults to `alba`. The UI supports duplex listening and barge-in cancellation.

## Dependencies and tests

`requirements.txt` pins the application runtime dependencies. `requirements-dev.txt` adds test, benchmark, and license-inventory tools. `requirements-lock-win-py312.txt` pins the complete tested Windows Python 3.12 environment, including transitive packages; `setup.ps1` installs that lock. llama-cpp-python is a separate optional native build. The standard package may be CPU-only; Vulkan requires a source build using the steps above, plus the Vulkan SDK, CMake, Ninja, and a C++ toolchain. Hardware acceleration also needs the OpenVINO/NPU driver stack and local model assets.

Run the CPU-safe unit suite and browser playback harness:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node .\tests\browser_playback_harness.js
node .\tests\capture_resampler_harness.js
```

## Troubleshooting

- If PowerShell blocks a script, allow scripts only in the current shell with `Set-ExecutionPolicy -Scope Process Bypass`, then rerun it.
- If OpenVINO reports no NPU, check `(.venv) python -c "import openvino as ov; print(ov.Core().available_devices)"`, update the Intel NPU driver, or leave `V2V_ASR_DEVICE=AUTO` for CPU fallback.
- If the strict GPU launcher exits, run `vulkaninfo --summary`, check that it lists Intel Arc, and rebuild `llama-cpp-python` with the Vulkan SDK/CMake/Ninja instructions above. `run_v2v.bat` allows a CPU fallback.
- The first Pocket TTS start may download its model and predefined voice embedding. Keep the machine online and allow disk space for the Hugging Face cache. Custom voice prompts use the optional gated voice-cloning model; accept its conditions and authenticate with `hf auth login` before enabling that path.
- If the browser does not capture audio, open the app at `http://127.0.0.1:8088`, grant microphone access to localhost, check the selected input device and OS microphone privacy settings, and verify the browser reports 16 kHz mono capture in its ready state.
- If replies use CPU unexpectedly, inspect `/health` and the server startup diagnostics. Vulkan is reported as active only when native logs identify Intel Arc and verify model layer placement.

`benchmarks/bench_production_backend.py` measures Qwen llama.cpp TTFT and decode rate on both Vulkan and CPU. `benchmarks/evaluate_stt.py` accepts a JSONL manifest of mono 16-bit, 16 kHz WAV files and references. `benchmarks/bench_voice_e2e.py` replays locally supplied audio through the real server WebSocket path. Provide your own audio; captured clips, transcripts, and historical raw result logs are not included.

Historical aggregate results and their limitations are in [benchmarks/verified-results.md](benchmarks/verified-results.md). They are evidence from the local machine, not a promise that another installation will match them.

## Privacy and local data

Do not commit model weights, compiled caches, local voice recordings, transcripts, `.venv` folders, or private knowledge documents. The app defaults to redacted transcript logging. `company_knowledge/` is intentionally empty in this repository; populate it only with material you have permission to use.

## License and attribution

Original project source and documentation are licensed under [Apache License 2.0](LICENSE). Copyright © 2026 Mohammed Omer. Third-party packages and model assets retain their own licenses; the project license does not cover those works. The dependency inventory and model sources are listed in [DEPENDENCY-LICENSES.md](DEPENDENCY-LICENSES.md), [models/README.md](models/README.md), and [NOTICE](NOTICE).

No model weights are included in this repository. Parakeet and its FluidInference OpenVINO conversion, the sherpa-onnx fallback model, Qwen, and Pocket TTS assets are downloaded separately and carry their upstream terms. The Pocket TTS voice-cloning model is access-gated and its terms restrict certain uses, including unauthorized voice impersonation; the default predefined voice uses the separately licensed without-voice-cloning path. Review the license and access conditions attached to the exact artifact revision before use or redistribution. The historical Q5_K_M benchmark model remains unattributed and is not the recommended setup artifact.
