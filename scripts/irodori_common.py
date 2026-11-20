"""irodori-wrapper の共通処理。

設定はプロジェクトのルートの .env から読む（すでにある環境変数のほうを優先する）。
Irodori-TTS 本体は、ラッパーのサブフォルダ Irodori-TTS/ に置く。
このファイルを import すると、.env の読み込み・出力の UTF-8 化・FFmpeg の設定・datasets のキャッシュの設定を行う。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 既定のベースモデル（.env の IRODORI_HF_CHECKPOINT で上書きできる）
DEFAULT_HF_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"

# ラッパーのプロジェクトのルート（このファイルは scripts/ の中にある）
WRAPPER_DIR = Path(__file__).resolve().parent.parent
# 設定ファイル
ENV_FILE = WRAPPER_DIR / ".env"
# Irodori-TTS 本体のフォルダ（setup.bat で clone する）
IRODORI_DIR = WRAPPER_DIR / "Irodori-TTS"
# datasets のキャッシュの既定の場所（ラッパーのルートから見たパス）
DEFAULT_DATASETS_CACHE = ".cache/datasets"


def load_env_file(path: Path = ENV_FILE) -> None:
  """.env の「名前=値」を環境変数に入れる。すでにある環境変数は上書きしない。

  - # で始まる行と空行は読み飛ばす
  - 値を " か ' で囲んだときは、囲みを外す
  - 値の中の %VAR% と ${VAR} は、環境変数に置き換える
  """
  if not path.is_file():
    return
  for line in path.read_text(encoding="utf-8-sig").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    name, value = line.split("=", 1)
    name = name.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
      value = value[1:-1]
    if name and name not in os.environ:
      os.environ[name] = os.path.expandvars(value)


def setup_console() -> None:
  """コンソールで絵文字や日本語が化けないように、出力を UTF-8 にする。"""
  for stream in (sys.stdout, sys.stderr):
    try:
      stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
      pass
  # 本体のスクリプトを subprocess で呼ぶときにも UTF-8 にする
  os.environ.setdefault("PYTHONUTF8", "1")
  os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def setup_ffmpeg() -> None:
  """FFMPEG_SHARED_BIN のフォルダがあれば、PATH の先頭と DLL を探す場所に加える。

  torchcodec が FFmpeg 9 に対応していないため、shared 版の 8 を優先して使う。
  """
  value = os.environ.get("FFMPEG_SHARED_BIN", "").strip()
  if not value:
    return
  bin_dir = Path(value).expanduser()
  if not bin_dir.is_dir():
    return
  paths = os.environ.get("PATH", "").split(os.pathsep)
  if str(bin_dir) not in paths:
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
  # Windows の Python 3.8 以降は、DLL を探すときに PATH を見ないことがあるため
  if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(bin_dir))


def setup_hf_cache() -> None:
  """datasets のキャッシュの場所を決め、HF_DATASETS_CACHE に絶対パスで入れる。

  既定はラッパーの .cache/datasets。ユーザーフォルダのキャッシュに書き込めないと、
  datasets のファイルロックがエラーを出さずに待ち続けるため、プロジェクトの中に置く。
  相対パスはラッパーのルートから見たパスにする（本体は Irodori-TTS/ を作業フォルダにして動くため）。
  datasets を import する前に呼ぶ必要がある。
  """
  value = os.environ.get("HF_DATASETS_CACHE", "").strip() or DEFAULT_DATASETS_CACHE
  path = Path(value).expanduser()
  if not path.is_absolute():
    path = WRAPPER_DIR / path
  os.environ["HF_DATASETS_CACHE"] = str(path.resolve())


def default_hf_checkpoint() -> str:
  """環境変数から既定のベースモデルを返す。"""
  value = os.environ.get("IRODORI_HF_CHECKPOINT", "").strip()
  return value or DEFAULT_HF_CHECKPOINT


def setup_irodori() -> Path:
  """本体のフォルダを確かめ、import できるように sys.path に加える。"""
  root = IRODORI_DIR
  if not (root / "irodori_tts").is_dir():
    sys.exit(
      f"[エラー] Irodori-TTS 本体が見つかりません: {root}\n"
      "  setup.bat を実行して、本体を入れてください。"
    )
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  return root


# import したときに、設定を読み込んで環境を整える
load_env_file()
setup_console()
setup_ffmpeg()
setup_hf_cache()
