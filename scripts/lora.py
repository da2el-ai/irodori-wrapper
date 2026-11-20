#!/usr/bin/env python3
"""Irodori-TTS の LoRA 学習をまとめて行うスクリプト（lora.bat から使う）。

使い方:
  lora.py init       <学習フォルダ>               学習フォルダのひな形を作る
  lora.py transcribe <学習フォルダ>               train/ の音声を文字起こしして metadata.csv を作る
  lora.py suggest    <学習フォルダ>               学習パラメーターを提案し、train.yaml に書き出す
  lora.py latent     <学習フォルダ>               学習用データ（latent とマニフェスト）を作る
  lora.py train      <学習フォルダ> [オプション]  LoRA を学習する（train.yaml があれば読む）
  lora.py sample     <学習フォルダ> [--watch]     チェックポイントごとにサンプル音声を作る

学習フォルダの中身:
  train/             学習素材（音声ファイルと metadata.csv）。ユーザーが用意する
  sample.yaml        サンプル出力の定義
  train.yaml         学習オプション（suggest が作る）
  sample/            サンプル音声（sample が作る）
  output/            作業フォルダ（このスクリプトが作る）
    manifest.jsonl   学習用データの一覧
    latents/         音声を変換したデータ
    train/           学習の結果（checkpoint_final が完成した LoRA アダプター）
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from irodori_common import default_hf_checkpoint, setup_irodori

# 既定の LoRA 学習設定（本体の configs フォルダの中のファイル）
DEFAULT_CONFIG = "configs/train_v4_small_lora.yaml"
# 学習データフォルダに置く、音声とテキストの対応表
METADATA_NAME = "metadata.csv"
# 使える音声ファイルの拡張子
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# 学習フォルダの中の名前
DATA_DIR_NAME = "train"
WORK_DIR_NAME = "output"
SAMPLE_YAML_NAME = "sample.yaml"
TRAIN_YAML_NAME = "train.yaml"

# 学習オプション: (train.yaml / コマンドでの名前, 型, train.py の引数, 説明)
TRAIN_OPTIONS: list[tuple[str, type, str, str]] = [
  ("max-steps", int, "--max-steps", "学習の総ステップ数"),
  ("batch-size", int, "--batch-size", "バッチサイズ"),
  ("grad-accum", int, "--gradient-accumulation-steps", "勾配を何回分ためてから更新するか"),
  ("lr", float, "--lr", "学習率"),
  ("warmup-steps", int, "--warmup-steps", "ウォームアップのステップ数"),
  ("stable-steps", int, "--stable-steps", "学習率を一定に保つステップ数（WSD）"),
  ("save-every", int, "--save-every", "何ステップごとに保存するか"),
  ("log-every", int, "--log-every", "何ステップごとにログを出すか"),
  ("num-workers", int, "--num-workers", "データ読み込みのプロセス数"),
  ("valid-ratio", float, "--valid-ratio", "検証に回すデータの割合（0 で検証なし）"),
  ("valid-every", int, "--valid-every", "何ステップごとに検証するか"),
  ("lora-r", int, "--lora-r", "LoRA のランク"),
  ("lora-alpha", int, "--lora-alpha", "LoRA の alpha"),
  ("checkpoint-best-n", int, "--checkpoint-best-n", "残すチェックポイントの数の設定（0 ですべて残す。既定: 0）"),
]
# 指定がないときに、設定ファイルの値ではなくこの値を渡す項目
TRAIN_FORCED_DEFAULTS: dict[str, Any] = {
  # 本体の既定（5）だと古いチェックポイントが消え、サンプルで聞き比べられなくなるため
  "checkpoint-best-n": 0,
}

# lora.py init で作る sample.yaml のひな形
SAMPLE_YAML_TEMPLATE = """\
# 学習中のサンプル出力の定義（lora.bat sample が読む）
# 書き方は generate.bat の入力 YAML と同じ。ただし、次の 2 つは書けない
#   lora    : チェックポイントごとに、スクリプトが決める
#   out-dir : この学習フォルダの sample\\ に出力する
# 相対パスは、この YAML ファイルがあるフォルダ（学習フォルダ）から見たパス
#
# ファイル名: <filename>_step0001000_seed0.wav
#   - filename を省くと voice1, voice2 ... になる
#   - step0000000 は LoRA なし（ベースモデル）、final は checkpoint_final
# seed を書かないときは 0 から数える（チェックポイントどうしを同じ seed で聞き比べるため）

setting:
  # caption: "落ち着いた、近い距離感の女性話者"
  seed: 0

voice:
  - text: こんにちは、今日はいい天気ですね。
    filename: 挨拶
  - text: えっ、本当に？😲
    filename: 驚き
"""


# ---------- 共通 ----------

def project_dir_for(args: argparse.Namespace, must_exist: bool = True) -> Path:
  """学習フォルダを確かめて返す。"""
  project = Path(args.project).expanduser().resolve()
  if must_exist and not project.is_dir():
    sys.exit(f"[エラー] 学習フォルダが見つかりません: {project}\n  先に lora.bat init で作ってください。")
  return project


def data_dir_for(project: Path) -> Path:
  """学習素材のフォルダを確かめて返す。"""
  data_dir = project / DATA_DIR_NAME
  if not data_dir.is_dir():
    sys.exit(
      f"[エラー] 学習素材のフォルダが見つかりません: {data_dir}\n"
      f"  学習フォルダの中の {DATA_DIR_NAME}\\ に、音声ファイルと metadata.csv を置いてください。"
    )
  return data_dir


def run(cmd: list[str], cwd: Path) -> None:
  """コマンドを実行し、失敗したら終了する。"""
  print("[実行] " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
  code = subprocess.call(cmd, cwd=str(cwd))
  if code != 0:
    sys.exit(f"[エラー] コマンドが失敗しました（終了コード {code}）")


def read_metadata(data_dir: Path) -> list[dict[str, str]] | None:
  """metadata.csv を読んで行の一覧を返す。ないときは None。"""
  meta = data_dir / METADATA_NAME
  if not meta.is_file():
    return None
  try:
    content = meta.read_text(encoding="utf-8-sig")
  except UnicodeDecodeError:
    sys.exit(f"[エラー] {meta} は UTF-8 で保存してください。")
  return list(csv.DictReader(content.splitlines()))


def check_metadata(data_dir: Path) -> list[str]:
  """metadata.csv を確かめ、列名の一覧を返す。"""
  meta = data_dir / METADATA_NAME
  rows = read_metadata(data_dir)
  if rows is None:
    sys.exit(
      f"[エラー] {meta} がありません。\n"
      "  lora.bat transcribe で作るか、file_name,text の列を持つ metadata.csv を置いてください。"
    )
  columns = list(rows[0].keys()) if rows else []
  for required in ("file_name", "text"):
    if required not in columns:
      sys.exit(f"[エラー] {meta} に {required} 列がありません（今の列: {columns}）")
  errors: list[str] = []
  for line_no, row in enumerate(rows, start=2):
    name = (row.get("file_name") or "").strip()
    text = (row.get("text") or "").strip()
    if not name or not text:
      errors.append(f"{line_no}行目: file_name か text が空です")
      continue
    path = data_dir / name
    if path.suffix.lower() not in AUDIO_EXTS:
      errors.append(f"{line_no}行目: 対応していない拡張子です: {name}")
    elif not path.is_file():
      errors.append(f"{line_no}行目: 音声ファイルがありません: {name}")
  if errors:
    print(f"[エラー] {meta} に問題があります:", file=sys.stderr)
    for message in errors:
      print(f"  {message}", file=sys.stderr)
    sys.exit(1)
  print(f"[データ] {len(rows)} 件（列: {', '.join(columns)}）")
  return columns


def add_speaker_id(manifest: Path, speaker_id: str) -> int:
  """speaker_id がない行に、指定した話者 ID を書き足す。マニフェストの件数を返す。"""
  lines = manifest.read_text(encoding="utf-8").splitlines()
  out: list[str] = []
  added = 0
  for line in lines:
    if not line.strip():
      continue
    item = json.loads(line)
    if not item.get("speaker_id"):
      item["speaker_id"] = speaker_id
      added += 1
    out.append(json.dumps(item, ensure_ascii=False))
  manifest.write_text("\n".join(out) + "\n", encoding="utf-8")
  print(f"[データ] マニフェスト {len(out)} 件（話者 ID を {added} 件に付けました: {speaker_id}）")
  return len(out)


# ---------- init ----------

def cmd_init(args: argparse.Namespace) -> None:
  project = project_dir_for(args, must_exist=False)
  created: list[str] = []
  data_dir = project / DATA_DIR_NAME
  if not data_dir.is_dir():
    data_dir.mkdir(parents=True)
    created.append(f"{DATA_DIR_NAME}\\")
  sample_yaml = project / SAMPLE_YAML_NAME
  if not sample_yaml.exists():
    sample_yaml.write_text(SAMPLE_YAML_TEMPLATE, encoding="utf-8")
    created.append(SAMPLE_YAML_NAME)
  if created:
    print(f"[作成] {project}: {', '.join(created)}")
  else:
    print(f"[情報] 作るものはありません（すべてあります）: {project}")
  print("  次の手順:")
  print(f"    1. {data_dir} に音声ファイルを入れる")
  print(f'    2. lora.bat transcribe "{project}"   （metadata.csv を作り、内容を確かめて直す）')
  print(f'    3. lora.bat suggest "{project}"      （学習オプションを train.yaml に書き出す）')
  print(f'    4. lora.bat latent "{project}"')
  print(f'    5. lora.bat train "{project}"        （別のウィンドウで lora.bat sample "{project}" --watch）')


# ---------- transcribe ----------

def cmd_transcribe(args: argparse.Namespace) -> None:
  from transcribe import transcribe_folder

  project = project_dir_for(args)
  data_dir = data_dir_for(project)
  transcribe_folder(
    data_dir, data_dir / METADATA_NAME, model=args.model, device=args.device, overwrite=args.overwrite,
  )
  print("  kotoba-whisper は句読点を出さず、聞き間違いもあります。metadata.csv を開いて確かめてください。")
  print(f'  次は学習オプションの提案です: lora.bat suggest "{project}"')


# ---------- suggest / sample（別のファイル） ----------

def cmd_suggest(args: argparse.Namespace) -> None:
  from lora_suggest import run_suggest

  project = project_dir_for(args)
  run_suggest(project, data_dir_for(project), epochs=args.epochs, overwrite=args.overwrite)


def cmd_sample(args: argparse.Namespace) -> None:
  from lora_sample import run_sample

  project = project_dir_for(args)
  run_sample(
    project, watch=args.watch, interval=args.interval, force=args.force, base=not args.no_base,
    dry_run=args.dry_run,
  )


# ---------- latent ----------

def cmd_latent(args: argparse.Namespace) -> None:
  root = setup_irodori()
  project = project_dir_for(args)
  name = args.name or project.name
  data_dir = data_dir_for(project)
  columns = check_metadata(data_dir)

  work = project / WORK_DIR_NAME
  manifest = work / "manifest.jsonl"
  if manifest.exists() and not args.force:
    sys.exit(f"[エラー] すでにマニフェストがあります: {manifest}\n  作り直すときは --force を付けてください。")
  work.mkdir(parents=True, exist_ok=True)

  # pandas 3.0 と datasets の組み合わせで metadata.csv が読めないため、run_upstream.py を通す
  cmd = [
    sys.executable, str(Path(__file__).resolve().parent / "run_upstream.py"),
    str(root / "prepare_manifest.py"),
    "--dataset", str(data_dir),
    "--split", "train",
    "--audio-column", "audio",
    "--text-column", "text",
    "--output-manifest", str(manifest),
    "--latent-dir", str(work / "latents"),
  ]
  if "caption" in columns:
    cmd += ["--caption-column", "caption"]
  if "speaker" in columns:
    cmd += ["--speaker-column", "speaker", "--speaker-id-prefix", name]
  if args.device:
    cmd += ["--device", args.device]
  run(cmd, cwd=root)

  if not manifest.is_file():
    sys.exit(f"[エラー] マニフェストが作られませんでした: {manifest}")
  # speaker 列がないときは、全部を同じ話者として扱う
  count = add_speaker_id(manifest, f"{name}:speaker")
  if count == 0:
    sys.exit(
      "[エラー] マニフェストが 0 件です。音声を 1 つも読めませんでした。\n"
      "  上の「skip breakdown」で、飛ばされた理由を確かめてください。\n"
      "  dataset_iter_error のときは、音声の読み込み（FFmpeg の shared 版と .env の FFMPEG_SHARED_BIN）を確かめてください。"
    )
  rows = read_metadata(data_dir)
  if rows is not None and count < len(rows):
    print(f"[注意] metadata.csv の {len(rows)} 件のうち、{len(rows) - count} 件が飛ばされました（上の「skip breakdown」を参照）。")
  print(f"[完了] 学習用データを作りました: {work}")
  print(f'  次は学習です: lora.bat train "{project}"')


# ---------- train ----------

def load_train_yaml(path: Path) -> dict[str, Any]:
  """train.yaml を読み、学習オプションの値を返す。ないときは空。"""
  if not path.is_file():
    return {}
  import yaml

  try:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
  except (UnicodeDecodeError, yaml.YAMLError) as exc:
    sys.exit(f"[エラー] {path} を読めません（UTF-8 の YAML で保存してください）: {exc}")
  if not isinstance(data, dict):
    sys.exit(f"[エラー] {path} には「名前: 値」の形で書いてください。")
  types = {key: typ for key, typ, _flag, _help in TRAIN_OPTIONS}
  values: dict[str, Any] = {}
  errors: list[str] = []
  for key, value in data.items():
    if key not in types:
      errors.append(f"知らない項目です: {key}（使える項目: {', '.join(types)}）")
      continue
    if value is None:
      continue
    try:
      if isinstance(value, bool):
        raise ValueError
      values[key] = types[key](value)
    except (TypeError, ValueError):
      errors.append(f"{key} の値が正しくありません: {value!r}")
  if errors:
    print(f"[エラー] {path} に問題があります:", file=sys.stderr)
    for message in errors:
      print(f"  {message}", file=sys.stderr)
    sys.exit(1)
  return values


def cmd_train(args: argparse.Namespace) -> None:
  root = setup_irodori()
  project = project_dir_for(args)
  work = project / WORK_DIR_NAME
  manifest = work / "manifest.jsonl"
  if not manifest.is_file():
    sys.exit(f"[エラー] マニフェストがありません: {manifest}\n  先に lora.bat latent を実行してください。")
  output_dir = work / "train"

  config = Path(args.config)
  if not config.is_absolute():
    config = root / config
  if not config.is_file():
    sys.exit(f"[エラー] 学習設定ファイルが見つかりません: {config}")

  cmd = [
    sys.executable, str(root / "train.py"),
    "--config", str(config),
    "--manifest", str(manifest),
    "--output-dir", str(output_dir),
  ]
  if args.resume:
    cmd += ["--resume", str(Path(args.resume).expanduser().resolve())]
  else:
    if args.init_checkpoint:
      init_ckpt = Path(args.init_checkpoint).expanduser().resolve()
      if not init_ckpt.is_file():
        sys.exit(f"[エラー] ベースモデルが見つかりません: {init_ckpt}")
    else:
      from irodori_tts.inference_runtime import download_hf_checkpoint

      repo_id = args.hf_checkpoint or default_hf_checkpoint()
      print(f"[モデル] {repo_id} を準備しています（初回はダウンロードします）")
      init_ckpt = Path(download_hf_checkpoint(repo_id))
    cmd += ["--init-checkpoint", str(init_ckpt)]

  # 学習オプション: train.yaml の値を、コマンドで指定した値で上書きする
  # どちらにもないものは、設定ファイルの値を使う（TRAIN_FORCED_DEFAULTS を除く）
  train_yaml = project / TRAIN_YAML_NAME
  options = dict(TRAIN_FORCED_DEFAULTS)
  from_yaml = load_train_yaml(train_yaml)
  options.update(from_yaml)
  if from_yaml:
    print(f"[設定] {train_yaml} を使います")
  for key, _typ, _flag, _help in TRAIN_OPTIONS:
    value = getattr(args, key.replace("-", "_"))
    if value is not None:
      options[key] = value
  if options:
    print("[設定] " + "  ".join(f"{k}={v}" for k, v in options.items()))
  for key, _typ, flag, _help in TRAIN_OPTIONS:
    if key in options:
      cmd += [flag, str(options[key])]
  if args.device:
    cmd += ["--device", args.device]
  cmd += args.extra

  run(cmd, cwd=root)
  final = output_dir / "checkpoint_final"
  print(f"[完了] LoRA の学習が終わりました: {output_dir}")
  if final.exists():
    print("  生成で使うときは、入力の YAML の setting に書きます:")
    print(f'    lora: "{final}"')


# ---------- 引数 ----------

def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="lora.bat", description="Irodori-TTS の LoRA 学習を行います。")
  sub = parser.add_subparsers(dest="command", required=True, metavar="コマンド")

  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("project", help="学習フォルダ（train\\ に学習素材を置く）")

  p = sub.add_parser("init", parents=[common], help="学習フォルダのひな形を作る")
  p.set_defaults(func=cmd_init)

  p = sub.add_parser("transcribe", parents=[common], help="train\\ の音声を文字起こしして metadata.csv を作る")
  p.add_argument("--model", default="kotoba-tech/kotoba-whisper-v2.0", help="Whisper のモデル")
  p.add_argument("--device", default=None, help="使うデバイス（既定: 自動）")
  p.add_argument("--overwrite", action="store_true", help="metadata.csv があっても上書きする")
  p.set_defaults(func=cmd_transcribe)

  p = sub.add_parser("suggest", parents=[common], help="学習パラメーターを提案し、train.yaml に書き出す")
  p.add_argument("--epochs", type=int, default=200, help="データを何周させるか（既定: 200）")
  p.add_argument("--overwrite", action="store_true", help="train.yaml があっても上書きする")
  p.set_defaults(func=cmd_suggest)

  p = sub.add_parser("latent", parents=[common], help="学習用データ（latent とマニフェスト）を作る")
  p.add_argument("--name", default=None, help="LoRA の名前。話者 ID に使う（既定: 学習フォルダの名前）")
  p.add_argument("--device", default=None, help="使うデバイス（既定: 自動）")
  p.add_argument("--force", action="store_true", help="マニフェストがあっても作り直す")
  p.set_defaults(func=cmd_latent)

  t = sub.add_parser("train", parents=[common], help="LoRA を学習する（train.yaml があれば読む）")
  t.add_argument("--config", default=DEFAULT_CONFIG, help=f"学習設定ファイル（既定: {DEFAULT_CONFIG}）")
  base = t.add_mutually_exclusive_group()
  base.add_argument("--init-checkpoint", default=None, help="ベースモデルのファイル（.safetensors）")
  base.add_argument("--hf-checkpoint", default=None, help="ベースモデルの Hugging Face ID（既定: .env の値）")
  base.add_argument("--resume", default=None, help="途中から再開するときのチェックポイントのフォルダ")
  for key, typ, _flag, help_text in TRAIN_OPTIONS:
    t.add_argument(f"--{key}", type=typ, default=None, help=help_text)
  t.add_argument("--device", default=None, help="使うデバイス（既定: 自動）")
  # -- のあとの引数は main() で切り分ける（argparse はサブコマンドの位置引数と -- をうまく扱えないため）
  t.epilog = "-- のあとに書いた引数は、そのまま train.py に渡します（例: lora.bat train 学習フォルダ -- --seed 1）"
  t.set_defaults(func=cmd_train)

  s = sub.add_parser("sample", parents=[common], help="チェックポイントごとにサンプル音声を作る")
  s.add_argument("--watch", action="store_true", help="学習と並行して、新しいチェックポイントを待って作る")
  s.add_argument("--interval", type=float, default=30.0, help="--watch で確かめる間隔（秒。既定: 30）")
  s.add_argument("--force", action="store_true", help="作ったことがあるチェックポイントも作り直す")
  s.add_argument("--no-base", action="store_true", help="LoRA なし（step0000000）のサンプルを作らない")
  s.add_argument("--dry-run", action="store_true", help="モデルを読まずに、作るファイル名だけを表示する")
  s.set_defaults(func=cmd_sample)
  return parser


def main() -> None:
  argv = sys.argv[1:]
  extra: list[str] = []
  if "--" in argv:
    index = argv.index("--")
    argv, extra = argv[:index], argv[index + 1:]
  args = build_parser().parse_args(argv)
  if extra and args.command != "train":
    sys.exit("[エラー] -- のあとの引数は、train でだけ使えます。")
  args.extra = extra
  args.func(args)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n[中断] 途中で止めました。")
    sys.exit(130)
