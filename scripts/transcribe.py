#!/usr/bin/env python3
"""LoRA の学習データ用に、フォルダの音声ファイルを Whisper で文字起こしするスクリプト。

使い方:
  transcribe.py <音声フォルダ> [--model モデル] [--output 出力CSV] [--overwrite]

- フォルダ直下の wav / flac / mp3 / ogg を名前順に読む（サブフォルダは見ない）
- 結果は file_name,text の CSV（UTF-8）に書き出す（既定: <音声フォルダ>/transcribe_raw.csv）
- lora.bat transcribe から使うときは、<学習フォルダ>/train/metadata.csv に書き出す（そのまま学習に使えるように）
- 本体の .venv にある transformers を使う。モデルは初回だけダウンロードする
- transformers が内部で torchcodec（FFmpeg shared 版）を使う。
  FFmpeg の場所は .env の FFMPEG_SHARED_BIN から irodori_common が設定する
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# .env の読み込み・出力の UTF-8 化・FFmpeg の設定（import するだけで行う）
import irodori_common  # noqa: F401

# 既定の Whisper モデル（日本語向け）
DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
# 既定の出力ファイル名
DEFAULT_OUTPUT_NAME = "transcribe_raw.csv"
# 使える音声ファイルの拡張子
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# Whisper が受け取るサンプリングレート
SAMPLE_RATE = 16000
# 分割せずに文字起こしする音声の長さの上限（秒）。Whisper は 30 秒ずつ処理する
MAX_SINGLE_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="音声フォルダを Whisper で文字起こしする")
  parser.add_argument("folder", help="音声ファイルがあるフォルダ")
  parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper のモデル（既定: {DEFAULT_MODEL}）")
  parser.add_argument("--output", help="出力する CSV（既定: <音声フォルダ>/<--default-name の名前>）")
  parser.add_argument(
    "--default-name",
    default=DEFAULT_OUTPUT_NAME,
    help=f"--output を省いたときの、出力する CSV の名前（既定: {DEFAULT_OUTPUT_NAME}）",
  )
  parser.add_argument("--device", help="使うデバイス（cuda / cpu など。既定: 自動）")
  parser.add_argument("--overwrite", action="store_true", help="出力する CSV があれば上書きする")
  return parser.parse_args()


def list_audio_files(folder: Path) -> list[Path]:
  """フォルダ直下の音声ファイルを名前順に返す。"""
  return sorted(
    (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
    key=lambda p: p.name,
  )


def load_audio(path: Path):
  """音声を 16kHz モノラルの配列として読む（FFmpeg を使わないよう librosa で読む）。"""
  import librosa

  audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
  return audio


def transcribe_folder(
  folder: Path, output: Path, model: str = DEFAULT_MODEL, device: str | None = None, overwrite: bool = False,
) -> None:
  """フォルダ直下の音声を文字起こしし、file_name,text の CSV に書き出す。"""
  if not folder.is_dir():
    sys.exit(f"[エラー] フォルダが見つかりません: {folder}")
  if output.exists() and not overwrite:
    sys.exit(f"[エラー] 出力する CSV がすでにあります: {output}（上書きするときは --overwrite）")

  files = list_audio_files(folder)
  if not files:
    sys.exit(f"[エラー] 音声ファイルがありません: {folder}")
  print(f"[情報] 音声ファイル: {len(files)} 個", flush=True)

  # 重いライブラリは引数の確認が終わってから読み込む
  import torch
  from transformers import pipeline

  device = device or ("cuda" if torch.cuda.is_available() else "cpu")
  dtype = torch.float16 if device.startswith("cuda") else torch.float32
  print(f"[情報] モデル: {model}（{device}）", flush=True)
  asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    device=device,
    dtype=dtype,
  )

  rows: list[dict[str, str]] = []
  for i, path in enumerate(files, 1):
    audio = load_audio(path)
    seconds = len(audio) / SAMPLE_RATE
    options: dict = {"generate_kwargs": {"language": "ja", "task": "transcribe"}}
    # 30 秒を超える音声だけ、分割して文字起こしする（分割すると精度が落ちることがあるため）
    if seconds > MAX_SINGLE_SECONDS:
      options["chunk_length_s"] = 15
    result = asr({"raw": audio, "sampling_rate": SAMPLE_RATE}, **options)
    text = str(result.get("text", "")).strip()
    print(f"[{i}/{len(files)}] {path.name} ({seconds:.1f}秒): {text}", flush=True)
    rows.append({"file_name": path.name, "text": text})

  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
    writer.writeheader()
    writer.writerows(rows)
  print(f"[完了] {output}", flush=True)


def main() -> None:
  args = parse_args()
  folder = Path(args.folder).expanduser().resolve()
  output = Path(args.output).expanduser().resolve() if args.output else folder / args.default_name
  transcribe_folder(folder, output, model=args.model, device=args.device, overwrite=args.overwrite)


if __name__ == "__main__":
  main()
