@echo off
setlocal
rem 本体の準備: setup.bat [PyTorch の種類] [Python のバージョン]
rem   Irodori-TTS 本体を Irodori-TTS\ に clone し（すでにあれば git pull で更新し）、uv sync で .venv を作る
rem   PyTorch の種類は cu128 / cpu / rocm / xpu のどれか。省略すると cu128（NVIDIA GPU）
rem   Python のバージョン（3.10 か 3.11。例: 3.11）を書くと、uv が管理する Python ではなく、システムに入れた Python を使う
rem   （uv sync が os error 448 で失敗するときに使う。README の「注意」を参照）
set "EXTRA=%~1"
if "%EXTRA%"=="" set "EXTRA=cu128"
set "PYVER=%~2"
set "PYARGS="
if not "%PYVER%"=="" set "PYARGS=--python %PYVER% --no-managed-python"
set "DIR=%~dp0Irodori-TTS"
set "REPO=https://github.com/Aratako/Irodori-TTS.git"

where git >nul 2>nul
if errorlevel 1 (
  echo [エラー] git が見つかりません。Git for Windows を入れてください。
  exit /b 1
)
where uv >nul 2>nul
if errorlevel 1 (
  echo [エラー] uv が見つかりません。uv を入れてください。
  exit /b 1
)

if exist "%DIR%\.git" (
  echo [更新] 本体を更新します: %DIR%
  git -C "%DIR%" pull --ff-only
  if errorlevel 1 exit /b 1
) else (
  echo [取得] 本体を clone します: %DIR%
  git clone "%REPO%" "%DIR%"
  if errorlevel 1 exit /b 1
)

echo [準備] uv sync --extra %EXTRA% %PYARGS%
pushd "%DIR%"
uv sync --extra %EXTRA% %PYARGS%
set "CODE=%ERRORLEVEL%"
popd
if not "%CODE%"=="0" (
  echo [エラー] uv sync が失敗しました。終了コード %CODE%
  echo   os error 448 のときは、システムに入れた Python 3.10 か 3.11 を使ってください。例: setup.bat %EXTRA% 3.11
  exit /b %CODE%
)
echo [完了] 本体の準備ができました。
exit /b 0
