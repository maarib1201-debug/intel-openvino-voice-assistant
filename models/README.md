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
- A Qwen3.5-0.8B GGUF, when using llama.cpp. The historical local GGUF publisher and conversion provenance were not recovered; the pinned Transformers model is the documented reproducible alternative.
- Pocket TTS weights (model card CC BY 4.0) and voice audio. The default `alba` samples are CC BY 4.0; check the selected voice's source terms before redistribution.

These artifacts have separate upstream licenses and versions. This repository does not bundle or mirror them. Check the license terms attached to the exact revisions you obtain before redistribution.
