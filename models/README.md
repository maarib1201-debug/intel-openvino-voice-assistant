Model artifacts are not tracked. See the root README for the exact runtime paths and expected model files. Set `V2V_MODEL_DIR` to an external model directory if preferred.
# Model assets

This repository does not redistribute model weights. Fetch the pinned ASR models and VAD using `..\prepare_models.ps1` from the repository root (`-DryRun` checks Hub revisions without downloading weights).

The exact public snapshot is [`FluidInference/parakeet-tdt-0.6b-v2-ov`](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov) at revision `e115adf711921df9378e383d01190901f0057704`. Its nine runtime files are four OpenVINO IR XML/BIN pairs (`parakeet_melspectogram`, `parakeet_encoder`, `parakeet_decoder`, `parakeet_joint`) plus `parakeet_vocab.json`. It is based on [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2); both model cards state [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute NVIDIA and FluidInference, link the license, and identify modifications when redistributing adapted artifacts.

The [FluidInference/eddy repository](https://github.com/FluidInference/eddy) documents a model fetcher that obtains these published files. Its source and build setup do not provide a Python conversion command that has been verified to regenerate these exact four graphs. This project therefore documents acquisition of the published revision, not conversion from NeMo. Regeneration of identical IR remains unverified.

The CPU fallback snapshot is [`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8) at `1ab9323565ddb038682214b292f588070a538ce2`. It contains `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, and `tokens.txt`; its model card is CC BY 4.0 and derives from the same NVIDIA model.

Silero VAD comes from [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad/tree/1e261b036686cd0017d500ee96acd1c4ba572a9d) at Git commit `1e261b036686cd0017d500ee96acd1c4ba572a9d`, file `src/silero_vad/data/silero_vad.onnx`, under MIT. Preserve its license notice if redistributing.

Other required separately obtained artifacts:

- `silero_vad.onnx` for Silero VAD.
- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/` as a CPU fallback.
- Qwen llama.cpp setup uses [`ggml-org/Qwen3.5-0.8B-GGUF`](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF), revision `8fea620810c4afa23dd6443f999a48574c1611a3`, file `Qwen3.5-0.8B-Q8_0.gguf`. The GGUF repository declares Apache-2.0 and identifies the official [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) checkpoint as its source. The original Q5_K_M benchmark file's publisher and conversion provenance remain unknown; it is not the documented setup model.
- Pocket TTS code is `pocket-tts==3.0.2` (MIT at the upstream source repository). Package configuration pins the English model weights to `kyutai/pocket-tts/languages/english/model.safetensors` at `39592ff23c9ef80098bb74895d104c26275fe2c9`; where the gated voice-cloning model is unavailable, it uses `kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors` at `d29db7978e464fb90cb3359ee0c69a273b9142cc`. Both model cards declare CC BY 4.0. The application uses the predefined `alba` voice, whose embedding is pinned by Pocket TTS to `kyutai/pocket-tts-without-voice-cloning/languages/english/embeddings/alba.safetensors` at `e81d79e8194ad4c7ce879c87a4258ef20cbf2487`. Its source voice, `alba-mackenna/casual.wav`, is at `kyutai/tts-voices` revision `323332d33f997de8394f24a193e1a76df720e01a` and is released under CC BY 4.0. Attribute Kyutai and voice performer Alba MacKenna as required by CC BY 4.0. The voice-cloning model is access-gated and lists prohibited uses; using custom voice prompts requires accepting its conditions and authenticating with Hugging Face. The default predefined voice path does not require voice-cloning access.

These artifacts have separate upstream licenses and versions. This repository does not bundle or mirror them. Check the license terms attached to the exact revisions you obtain before redistribution.
