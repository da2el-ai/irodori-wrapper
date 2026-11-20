"""学習素材を調べて、LoRA 学習のオプションを提案する（lora.py suggest から使う）。

考え方は .claude/knowledge_train.md のとおり。値は目安で、実際に聞き比べて直す前提。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# 本体が学習に使う音声の長さの上限（秒）。max_latent_steps 750 / 25 fps
MAX_SECONDS = 30.0
# これより短い音声は警告する（秒）。本体の参照音声の下限 ref_min_seconds と同じ
MIN_SECONDS = 1.0
# 使える音声ファイルの拡張子
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# データがこれより少ないときは警告する（個）
FEW_FILES = 10


def audio_seconds(path: Path) -> float:
  """音声の長さ（秒）を返す。"""
  import soundfile

  try:
    return float(soundfile.info(str(path)).duration)
  except Exception:
    # soundfile で読めない形式は librosa に任せる
    import librosa

    return float(librosa.get_duration(path=str(path)))


def fmt_seconds(sec: float) -> str:
  """秒を「1分23.4秒」のように表示する。"""
  if sec < 60:
    return f"{sec:.1f}秒"
  minutes, rest = divmod(sec, 60)
  return f"{int(minutes)}分{rest:.1f}秒"


def round_to(value: float, unit: int) -> int:
  """unit の倍数に丸める。"""
  return int(round(value / unit)) * unit


def suggest_options(count: int, epochs: int) -> tuple[dict[str, Any], dict[str, float]]:
  """データの数から学習オプションを決める。(オプション, 計算の内訳) を返す。"""
  # 少ないデータでは 1 回の更新に使う数を減らし、更新の回数を増やす
  batch = 8 if count >= 32 else max(1, count // 4)
  steps_per_epoch = count / batch
  # 100 の倍数に切り上げる。少なすぎると似ないので、下限を 500 にする
  max_steps = max(500, math.ceil(epochs * steps_per_epoch / 100) * 100)
  # WSD: ウォームアップは全体の 5%、下げる区間は全体の 25%
  warmup = max(10, round_to(max_steps * 0.05, 10))
  decay = round_to(max_steps * 0.25, 10)
  stable = max_steps - warmup - decay
  # 途中のチェックポイントを 8 つほど残して、聞き比べられるようにする
  save_every = max(50, round_to(max_steps / 8, 50))
  options = {
    "max-steps": max_steps,
    "batch-size": batch,
    "grad-accum": 1,
    "warmup-steps": warmup,
    "stable-steps": stable,
    "save-every": save_every,
    "log-every": 10,
    "valid-ratio": 0,
    "num-workers": 4,
    "checkpoint-best-n": 0,
  }
  detail = {
    "steps_per_epoch": steps_per_epoch,
    "epochs": max_steps / steps_per_epoch,
    "decay": decay,
  }
  return options, detail


def collect_files(data_dir: Path) -> tuple[list[Path], list[str]]:
  """学習に使う音声の一覧と、注意の一覧を返す。

  metadata.csv があれば、そこに書かれたファイルを使う。なければフォルダ直下の音声をすべて使う。
  """
  from lora import read_metadata

  notes: list[str] = []
  on_disk = sorted(
    (p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
    key=lambda p: p.name.lower(),
  )
  rows = read_metadata(data_dir)
  if rows is None:
    notes.append("metadata.csv がありません。フォルダの音声をすべて数えました（lora.bat transcribe で作れます）")
    return on_disk, notes

  files: list[Path] = []
  listed: set[str] = set()
  for line_no, row in enumerate(rows, start=2):
    name = (row.get("file_name") or "").strip()
    if not name:
      continue
    path = data_dir / name
    listed.add(path.resolve().as_posix().lower())
    if path.is_file():
      files.append(path)
    else:
      notes.append(f"metadata.csv {line_no}行目の音声がありません: {name}")
  unlisted = [p.name for p in on_disk if p.resolve().as_posix().lower() not in listed]
  if unlisted:
    notes.append(f"metadata.csv に書かれていない音声が {len(unlisted)} 個あります（学習には使われません）: "
                 + ", ".join(unlisted[:5]) + (" ..." if len(unlisted) > 5 else ""))
  return files, notes


def build_train_yaml(options: dict[str, Any], summary: list[str]) -> str:
  """train.yaml の中身を作る。"""
  lines = [
    "# lora.bat train が読む学習オプション（lora.bat suggest が作成）",
    "# 値は目安。サンプルを聞き比べて直す（.claude/knowledge_train.md を参照）",
    "# コマンドで同じオプションを書くと、そちらを優先する",
    "#",
  ]
  lines += [f"# {s}" for s in summary]
  lines.append("")
  for key, value in options.items():
    lines.append(f"{key}: {value}")
  lines += [
    "",
    "# 既定のまま（本体の設定ファイルの値）。変えるときはコメントを外す",
    "# lr: 0.0001",
    "# lora-r: 16",
    "# lora-alpha: 32",
    "",
  ]
  return "\n".join(lines)


def run_suggest(project: Path, data_dir: Path, epochs: int, overwrite: bool) -> None:
  if epochs < 1:
    sys.exit("[エラー] --epochs は 1 以上にしてください。")
  files, notes = collect_files(data_dir)
  if not files:
    sys.exit(f"[エラー] 学習に使う音声がありません: {data_dir}")

  durations: list[tuple[Path, float]] = []
  for path in files:
    try:
      durations.append((path, audio_seconds(path)))
    except Exception as exc:  # noqa: BLE001
      notes.append(f"長さを読めませんでした: {path.name}（{exc}）")
  if not durations:
    sys.exit("[エラー] 音声の長さを 1 つも読めませんでした。")

  secs = [d for _, d in durations]
  count = len(durations)
  total = sum(secs)
  used = sum(min(d, MAX_SECONDS) for d in secs)
  too_long = [(p, d) for p, d in durations if d > MAX_SECONDS]
  too_short = [(p, d) for p, d in durations if d < MIN_SECONDS]

  # ---- データの概要 ----
  print(f"[データ] {data_dir}")
  print(f"  個数: {count} 個")
  print(f"  合計: {fmt_seconds(total)}" + (f"（学習に使われるのは {fmt_seconds(used)}）" if used < total else ""))
  print(f"  長さ: 最短 {min(secs):.1f}秒 / 平均 {total / count:.1f}秒 / 最長 {max(secs):.1f}秒")

  # ---- 警告 ----
  warnings: list[str] = []
  if too_long:
    warnings.append(
      f"{MAX_SECONDS:.0f} 秒を超える音声が {len(too_long)} 個あります。"
      f"{MAX_SECONDS:.0f} 秒を過ぎた部分は学習に使われず、テキストと音声が合わなくなります。"
      "分けるか、除いてください:"
    )
    warnings += [f"    {p.name}（{d:.1f}秒）" for p, d in too_long]
  if too_short:
    warnings.append(
      f"{MIN_SECONDS:.0f} 秒未満の音声が {len(too_short)} 個あります。"
      "短すぎて、学習の効果がほとんどありません（本体の参照音声の下限も 1 秒です）。除くことをおすすめします:"
    )
    warnings += [f"    {p.name}（{d:.2f}秒）" for p, d in too_short]
  if count < FEW_FILES:
    warnings.append(f"音声が {count} 個しかありません。少なすぎると、声が似なかったり、覚えすぎたりします")
  for note in notes:
    print(f"[注意] {note}")
  for message in warnings:
    print(message if message.startswith("    ") else f"[警告] {message}")

  # ---- 提案 ----
  options, detail = suggest_options(count, epochs)
  m = options["max-steps"]
  print("[提案] 目安の値です。サンプルを聞き比べて直してください")
  print(f"  1 周のステップ数: {count} ÷ ({options['batch-size']} × {options['grad-accum']}) ≒ {detail['steps_per_epoch']:.1f}")
  print(f"  総ステップ数: {m}（約 {detail['epochs']:.0f} 周）")
  print(f"  学習率: {options['warmup-steps']} ステップで上げ、{options['stable-steps']} ステップ保ち、"
        f"最後の {int(detail['decay'])} ステップで下げる")
  print(f"  チェックポイント: {options['save-every']} ステップごと（約 {m // options['save-every']} 個。すべて残す）")

  summary = [
    f"データ: {count} 個 / 合計 {fmt_seconds(total)} / 最長 {max(secs):.1f}秒",
    f"1 周 ≒ {detail['steps_per_epoch']:.1f} ステップ、{m} ステップで約 {detail['epochs']:.0f} 周",
  ]
  train_yaml = project / "train.yaml"
  text = build_train_yaml(options, summary)
  if train_yaml.exists() and not overwrite:
    print(f"[情報] {train_yaml} がすでにあるので、書き出しませんでした（上書きするときは --overwrite）")
    print("  提案した値:")
    for key, value in options.items():
      print(f"    {key}: {value}")
  else:
    train_yaml.write_text(text, encoding="utf-8")
    print(f"[作成] {train_yaml}")
  print(f'  次は学習用データの作成です: lora.bat latent "{project}"')
