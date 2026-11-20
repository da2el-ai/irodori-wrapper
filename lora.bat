@echo off
setlocal
rem LoRA 学習: lora.bat コマンド 学習フォルダ [オプション]
rem   コマンド: init / transcribe / suggest / latent / train / sample
rem   使い方の詳細: lora.bat -h、lora.bat コマンド -h
set "PY=%~dp0Irodori-TTS\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [エラー] Irodori-TTS 本体が入っていません。先に setup.bat を実行してください。
  exit /b 1
)
"%PY%" "%~dp0scripts\lora.py" %*
exit /b %ERRORLEVEL%
