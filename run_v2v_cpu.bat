@echo off
cd /d "%~dp0"
set POCKET_TTS_NO_BEARTYPE=1
set TOKENIZERS_PARALLELISM=false
set V2V_LLM_N_GPU_LAYERS=0
set V2V_ASR_DEVICE=NPU
".venv\Scripts\python.exe" -m v2v %*
