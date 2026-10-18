@echo off
cd /d "%~dp0"
set POCKET_TTS_NO_BEARTYPE=1
set TOKENIZERS_PARALLELISM=false
".venv\Scripts\python.exe" -m v2v %*
