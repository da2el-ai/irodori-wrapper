@echo off
setlocal
rem 一括音声生成: generate.bat 入力.yaml [--dry-run] [--overwrite]
set "PY=%~dp0Irodori-TTS\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [エラー] Irodori-TTS 本体が入っていません。先に setup.bat を実行してください。
  exit /b 1
)
"%PY%" "%~dp0scripts\generate.py" %*
exit /b %ERRORLEVEL%
