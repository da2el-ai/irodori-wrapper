#!/usr/bin/env python3
"""YAML ファイルから Irodori-TTS で音声を一括で作るスクリプト。

入力ファイルの形式:
  setting:            # 全体の設定
    ref-wav: ...      # ファイル / フォルダ / ファイルのリスト
    caption: ...
    lora: ...         # LoRA アダプターのフォルダ（1 つだけ）
  voice:
    - text: ...       # 必須
      count: 2        # 省略すると 1
      filename: "..." # 省略すると {%Y%m%d-%H%M%S}
      setting:        # この voice だけの設定（項目ごとに全体の設定を上書き）
        caption: ...

相対パスは、YAML ファイルがあるフォルダから見たパスとして扱う。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from irodori_common import default_hf_checkpoint, setup_irodori

# ファイル名を省いたときの書式
DEFAULT_NAME_TEMPLATE = "{%Y%m%d-%H%M%S}"
# Windows のファイル名に使えない文字
INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')
# {...} の日時書式
TIME_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
# 参照音声として使う拡張子
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# 「指定なし」とみなす文字列
NONE_WORDS = {"none", "null", "off", ""}

# voice ごとに変えられる項目と既定値
VOICE_DEFAULTS: dict[str, Any] = {
  "ref-wav": None,
  "ref-embed": None,
  "caption": None,
  "lora": None,
  "seed": None,
  "num-steps": None,
  "seconds": None,
  "duration-scale": 1.0,
  "cfg-scale-text": 3.0,
  "cfg-scale-caption": 3.0,
  "cfg-scale-speaker": 5.0,
  "length-check": False,
}
# 全体の setting にだけ書ける項目と既定値（モデルを読み込むときに決まるもの）
GLOBAL_DEFAULTS: dict[str, Any] = {
  "out-dir": "outputs",
  "checkpoint": None,
  "hf-checkpoint": None,
  "device": None,
  "precision": "fp32",
}
# 参照として 1 組で扱う項目
REF_KEYS = ("ref-wav", "ref-embed")
# voice に書ける項目
VOICE_ITEM_KEYS = {"text", "count", "filename", "setting"}


class InputError(Exception):
  """入力ファイルの内容の誤り。"""


@dataclass
class VoiceSettings:
  """1 つの voice に使う、確定した設定。"""

  ref_wavs: list[str] | None = None
  ref_embed: str | None = None
  caption: str | None = None
  lora: str | None = None
  seed: int | None = None
  seed_is_local: bool = False
  num_steps: int | None = None
  seconds: float | None = None
  duration_scale: float = 1.0
  cfg_scale_text: float = 3.0
  cfg_scale_caption: float = 3.0
  cfg_scale_speaker: float = 5.0
  length_check: bool = False


@dataclass
class Job:
  """voice 1 件分の生成内容。"""

  index: int
  text: str
  count: int
  name_template: str
  settings: VoiceSettings = field(default_factory=VoiceSettings)
  # filename を書いたかどうか（書かなかったときは name_template が既定の書式になる）
  name_given: bool = False


# ---------- 値の変換 ----------

def is_none(value: Any) -> bool:
  """「指定なし」を表す値かどうか。"""
  return value is None or (isinstance(value, str) and value.strip().lower() in NONE_WORDS)


def to_int(key: str, value: Any) -> int:
  if isinstance(value, bool) or not isinstance(value, (int, str)):
    raise InputError(f"{key} は整数で書いてください: {value!r}")
  try:
    return int(value)
  except ValueError as exc:
    raise InputError(f"{key} は整数で書いてください: {value!r}") from exc


def to_float(key: str, value: Any) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise InputError(f"{key} は数値で書いてください: {value!r}")
  try:
    return float(value)
  except ValueError as exc:
    raise InputError(f"{key} は数値で書いてください: {value!r}") from exc


def to_bool(key: str, value: Any) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, str) and value.strip().lower() in {"on", "true", "yes"}:
    return True
  if is_none(value) or (isinstance(value, str) and value.strip().lower() in {"false", "no"}):
    return False
  raise InputError(f"{key} は on / off で書いてください: {value!r}")


def resolve_path(base: Path, value: Any, key: str) -> Path:
  if not isinstance(value, (str, int, float)) or isinstance(value, bool):
    raise InputError(f"{key} はパスで書いてください: {value!r}")
  path = Path(str(value)).expanduser()
  if not path.is_absolute():
    path = base / path
  return path.resolve()


def resolve_ref_wavs(base: Path, value: Any) -> list[str]:
  """ref-wav をファイルの一覧にする。フォルダは直下の音声ファイルを名前順に使う。"""
  items = value if isinstance(value, list) else [value]
  out: list[str] = []
  for item in items:
    path = resolve_path(base, item, "ref-wav")
    if path.is_dir():
      files = sorted(
        (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
        key=lambda p: p.name.lower(),
      )
      if not files:
        raise InputError(f"ref-wav のフォルダに音声ファイルがありません: {path}")
      out.extend(str(p) for p in files)
    elif path.is_file():
      if path.suffix.lower() not in AUDIO_EXTS:
        raise InputError(f"ref-wav に対応していない拡張子です: {path}")
      out.append(str(path))
    else:
      raise InputError(f"ref-wav が見つかりません: {path}")
  if not out:
    raise InputError("ref-wav が空です")
  return out


def resolve_lora(base: Path, value: Any) -> str:
  if isinstance(value, list):
    raise InputError("lora は 1 つだけ指定できます")
  path = resolve_path(base, value, "lora")
  if not path.is_dir():
    raise InputError(f"lora のフォルダが見つかりません: {path}")
  if not (path / "adapter_config.json").is_file():
    raise InputError(f"lora のフォルダに adapter_config.json がありません: {path}")
  return str(path)


def build_settings(merged: dict[str, Any], base: Path, seed_is_local: bool) -> VoiceSettings:
  """まとめた設定の値を確かめて、VoiceSettings にする。"""
  s = VoiceSettings(seed_is_local=seed_is_local)
  if not is_none(merged["ref-wav"]) and not is_none(merged["ref-embed"]):
    raise InputError("ref-wav と ref-embed は、どちらか一方だけ指定してください")
  if not is_none(merged["ref-wav"]):
    s.ref_wavs = resolve_ref_wavs(base, merged["ref-wav"])
  if not is_none(merged["ref-embed"]):
    path = resolve_path(base, merged["ref-embed"], "ref-embed")
    if not path.is_file():
      raise InputError(f"ref-embed が見つかりません: {path}")
    s.ref_embed = str(path)
  if not is_none(merged["caption"]):
    s.caption = str(merged["caption"]).strip() or None
  if not is_none(merged["lora"]):
    s.lora = resolve_lora(base, merged["lora"])
  if not is_none(merged["seed"]):
    s.seed = to_int("seed", merged["seed"])
  if not is_none(merged["num-steps"]):
    s.num_steps = to_int("num-steps", merged["num-steps"])
  if not is_none(merged["seconds"]):
    s.seconds = to_float("seconds", merged["seconds"])
  s.duration_scale = to_float("duration-scale", merged["duration-scale"])
  s.cfg_scale_text = to_float("cfg-scale-text", merged["cfg-scale-text"])
  s.cfg_scale_caption = to_float("cfg-scale-caption", merged["cfg-scale-caption"])
  s.cfg_scale_speaker = to_float("cfg-scale-speaker", merged["cfg-scale-speaker"])
  s.length_check = to_bool("length-check", merged["length-check"])
  return s


# ---------- 入力ファイルの読み込み ----------

def check_keys(data: dict, allowed: set[str], where: str) -> None:
  unknown = [str(k) for k in data if k not in allowed]
  if unknown:
    raise InputError(f"{where}: 知らない項目です: {', '.join(unknown)}（使える項目: {', '.join(sorted(allowed))}）")


def load_input(path: Path, forbidden: frozenset[str] = frozenset()) -> tuple[dict[str, Any], list[Job]]:
  """YAML を読み、全体の設定と voice の一覧を返す。

  forbidden に書いた項目は、setting に書くとエラーにする（スクリプトが値を決める項目）。
  """
  import yaml

  try:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
  except UnicodeDecodeError:
    sys.exit(f"[エラー] {path} は UTF-8 で保存してください。")
  except yaml.YAMLError as exc:
    if "unknown escape character" in str(exc):
      # " で囲んだ値の中に、Windows のパスの \ をそのまま書いたとき
      hint = (
        "  ヒント: \" で囲んだ値の中では、\\ が特別な意味になります。パスの区切りは \\\\ にしてください\n"
        "         （例: lora: \"D:\\\\irodori-wrapper\\\\lora_study\\\\名前\\\\output\\\\train\\\\checkpoint_final\"）。/ を使うか、' で囲んでもかまいません"
      )
    else:
      hint = "  ヒント: { や : や # を含む値は \" で囲んでください（例: filename: \"{%Y%m%d}_挨拶\"）"
    sys.exit(f"[エラー] YAML の書き方に誤りがあります: {path}\n{exc}\n{hint}")
  if not isinstance(data, dict):
    sys.exit("[エラー] YAML の一番上には setting: と voice: を書いてください。")

  base = path.parent.resolve()
  errors: list[str] = []

  try:
    check_keys(data, {"setting", "voice"}, "一番上")
  except InputError as exc:
    errors.append(str(exc))

  # 全体の setting
  global_setting = data.get("setting") or {}
  if not isinstance(global_setting, dict):
    errors.append("setting: には項目を書いてください")
    global_setting = {}
  try:
    check_keys(global_setting, set(VOICE_DEFAULTS) | set(GLOBAL_DEFAULTS), "setting")
  except InputError as exc:
    errors.append(str(exc))
  bad = [k for k in global_setting if k in forbidden]
  if bad:
    errors.append(f"setting: {', '.join(bad)} は書けません（スクリプトが決めます）")
    # 同じ項目のエラーが重ならないよう、このあとの確認からは外す
    global_setting = {k: v for k, v in global_setting.items() if k not in forbidden}
  global_opts = {k: global_setting.get(k, v) for k, v in GLOBAL_DEFAULTS.items()}
  global_voice = {k: global_setting.get(k, v) for k, v in VOICE_DEFAULTS.items()}
  try:
    build_settings(global_voice, base, seed_is_local=False)
  except InputError as exc:
    errors.append(f"setting: {exc}")

  # voice の一覧
  voices = data.get("voice")
  if not isinstance(voices, list) or not voices:
    errors.append("voice: に 1 件以上の項目を - で並べて書いてください")
    voices = []

  jobs: list[Job] = []
  for index, item in enumerate(voices, start=1):
    where = f"voice {index}件目"
    try:
      if not isinstance(item, dict):
        raise InputError("text: などの項目を書いてください")
      check_keys(item, VOICE_ITEM_KEYS, where)
      text = item.get("text")
      if is_none(text) or not isinstance(text, (str, int, float)):
        raise InputError("text がありません")
      text = str(text).strip()
      count = 1 if item.get("count") is None else to_int("count", item["count"])
      if count < 1:
        raise InputError(f"count は 1 以上にしてください: {count}")
      name = item.get("filename")
      if isinstance(name, (dict, list)):
        raise InputError("filename は \" で囲んでください（例: filename: \"{%Y%m%d}_挨拶\"）")
      name_given = not is_none(name)
      name = str(name) if name_given else DEFAULT_NAME_TEMPLATE

      local = item.get("setting") or {}
      if not isinstance(local, dict):
        raise InputError("setting: には項目を書いてください")
      bad = [k for k in local if k in forbidden]
      if bad:
        raise InputError(f"{', '.join(bad)} は書けません（スクリプトが決めます）")
      only_global = [k for k in local if k in GLOBAL_DEFAULTS]
      if only_global:
        raise InputError(f"{', '.join(only_global)} は全体の setting にだけ書けます")
      check_keys(local, set(VOICE_DEFAULTS), f"{where}の setting")
      # 項目ごとに上書きする。参照（ref-wav / ref-embed）は 1 組で置き換える
      merged = dict(global_voice)
      if any(k in local for k in REF_KEYS):
        for k in REF_KEYS:
          merged[k] = None
      merged.update(local)
      settings = build_settings(merged, base, seed_is_local="seed" in local)
      jobs.append(Job(
        index=index, text=text, count=count, name_template=name, settings=settings, name_given=name_given,
      ))
    except InputError as exc:
      errors.append(f"{where}: {exc}")

  if errors:
    print(f"[エラー] 入力ファイルに問題があります: {path}", file=sys.stderr)
    for message in errors:
      print(f"  {message}", file=sys.stderr)
    sys.exit(1)
  global_opts["_base"] = base
  global_opts["_seed"] = build_settings(global_voice, base, seed_is_local=False).seed
  return global_opts, jobs


# ---------- ファイル名 ----------

def render_stem(template: str, now: datetime) -> str:
  """ファイル名の書式の {...} を日時に置き換え、使えない文字を _ にする（拡張子なし）。"""
  stem = TIME_PLACEHOLDER.sub(lambda m: now.strftime(m.group(1)), template)
  # 拡張子 .wav が書かれていたら外す（あとに文字を付け足すため）
  if stem.lower().endswith(".wav"):
    stem = stem[:-4]
  return INVALID_NAME_CHARS.sub("_", stem).strip(" .")


def render_name(template: str, now: datetime, serial: int) -> str:
  """ファイル名の書式を置き換え、末尾に 4 桁の連番を付ける（拡張子なし）。"""
  return f"{render_stem(template, now)}_{serial:04d}"


def unique_path(out_dir: Path, stem: str, overwrite: bool) -> Path:
  """同じ名前のファイルがあるときは、上書きせずに -2, -3 ... を付けた名前を返す。"""
  path = out_dir / f"{stem}.wav"
  if overwrite or not path.exists():
    return path
  n = 2
  while True:
    candidate = out_dir / f"{stem}-{n}.wav"
    if not candidate.exists():
      print(f"[注意] 同じ名前のファイルがあるため名前を変えます: {candidate.name}")
      return candidate
    n += 1


def short_path(path: str, base: Path) -> str:
  """表示用に、YAML のフォルダからの相対パスにする（できないときはそのまま）。"""
  try:
    return os.path.relpath(path, base)
  except ValueError:
    return path


def describe(s: VoiceSettings, base: Path) -> str:
  """設定の概要を 1 行にする。"""
  parts = []
  if s.ref_wavs:
    parts.append("参照=" + "; ".join(short_path(p, base) for p in s.ref_wavs))
  elif s.ref_embed:
    parts.append("参照(embed)=" + short_path(s.ref_embed, base))
  else:
    parts.append("参照なし")
  if s.caption:
    parts.append(f"caption=「{s.caption}」")
  if s.lora:
    parts.append("lora=" + short_path(s.lora, base))
  if s.seed is not None:
    if s.seed_is_local:
      parts.append(f"seed={s.seed}から(個別)")
    else:
      parts.append(f"seed=全体の {s.seed} からの続き")
  return "  ".join(parts)


# ---------- 生成（lora_sample.py からも使う） ----------

# 生成の記録の列
LOG_HEADER = ["日時", "ファイル", "テキスト", "seed", "caption", "参照", "LoRA", "入力ファイル", "voice"]


def iter_items(jobs: list[Job], global_seed: int | None) -> Iterator[tuple[Job, int, int | None]]:
  """作る音声を 1 個ずつ (voice, 連番, seed) で返す。

  - 全体の seed は、voice をまたいで 1 つ作るたびに +1 する
  - voice の中に seed を書いたときは、その voice はその値から数え直す（空ならランダム）
  """
  for job in jobs:
    s = job.settings
    local_seed = s.seed if s.seed_is_local else None
    for serial in range(1, job.count + 1):
      yield job, serial, (local_seed if s.seed_is_local else global_seed)
      if s.seed_is_local:
        if local_seed is not None:
          local_seed += 1
      elif global_seed is not None:
        global_seed += 1


def load_runtime(opts: dict[str, Any]) -> Any:
  """全体の設定（checkpoint / device / precision）に合わせて、モデルを読み込む。"""
  setup_irodori()
  from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    default_runtime_device,
    download_hf_checkpoint,
  )

  base: Path = opts["_base"]
  if not is_none(opts["checkpoint"]) and not is_none(opts["hf-checkpoint"]):
    sys.exit("[エラー] checkpoint と hf-checkpoint は、どちらか一方だけ指定してください。")
  if not is_none(opts["checkpoint"]):
    checkpoint_path = resolve_path(base, opts["checkpoint"], "checkpoint")
    if not checkpoint_path.is_file():
      sys.exit(f"[エラー] モデルファイルが見つかりません: {checkpoint_path}")
    checkpoint = str(checkpoint_path)
  else:
    repo_id = default_hf_checkpoint() if is_none(opts["hf-checkpoint"]) else str(opts["hf-checkpoint"])
    print(f"[モデル] {repo_id} を準備しています（初回はダウンロードします）")
    checkpoint = download_hf_checkpoint(repo_id)

  precision = str(opts["precision"])
  if precision not in {"fp32", "bf16"}:
    sys.exit(f"[エラー] precision は fp32 か bf16 にしてください: {precision}")
  device = default_runtime_device() if is_none(opts["device"]) else str(opts["device"])
  return InferenceRuntime.from_key(
    RuntimeKey(
      checkpoint=checkpoint,
      model_device=device,
      model_precision=precision,
      codec_device=device,
      codec_precision=precision,
    )
  )


def synthesize(runtime: Any, text: str, s: VoiceSettings, seed: int | None, lora: str | None) -> Any:
  """音声を 1 個作る。lora が None のときは LoRA なし（ベースモデル）で作る。"""
  from irodori_tts import length_check
  from irodori_tts.inference_runtime import SamplingRequest, resolve_cfg_scales

  no_ref = s.ref_wavs is None and s.ref_embed is None
  use_speaker = bool(runtime.model_cfg.use_speaker_condition_resolved and not no_ref)
  cfg_text, cfg_caption, cfg_speaker, _messages = resolve_cfg_scales(
    cfg_guidance_mode="independent",
    cfg_scale_text=s.cfg_scale_text,
    cfg_scale_caption=s.cfg_scale_caption,
    cfg_scale_speaker=s.cfg_scale_speaker,
    cfg_scale=None,
    use_caption_condition=bool(runtime.model_cfg.use_caption_condition and s.caption),
    use_speaker_condition=use_speaker,
  )
  request = SamplingRequest(
    text=text,
    caption=s.caption,
    ref_wavs=s.ref_wavs,
    ref_embed=s.ref_embed,
    no_ref=no_ref,
    seconds=s.seconds,
    duration_scale=s.duration_scale,
    num_steps=s.num_steps,
    cfg_scale_text=cfg_text,
    cfg_scale_caption=cfg_caption,
    cfg_scale_speaker=cfg_speaker,
    seed=seed,
    # LoRA を使わないときは "none" を渡して、前の生成の LoRA を外す
    lora_adapter=lora or "none",
  )
  if s.length_check:
    result, _info = length_check.synthesize(runtime, request, log_fn=None)
  else:
    result = runtime.synthesize(request, log_fn=None)
  return result


def save_result(path: Path, result: Any) -> None:
  """作った音声を wav で保存する。"""
  from irodori_tts.inference_runtime import save_wav

  save_wav(path, result.audio, result.sample_rate)


def open_log(out_dir: Path) -> tuple[Any, Any]:
  """生成の記録（generation_log.csv）を追記で開き、(ファイル, csv.writer) を返す。"""
  out_dir.mkdir(parents=True, exist_ok=True)
  log_path = out_dir / "generation_log.csv"
  new_log = not log_path.exists()
  log_file = log_path.open("a", encoding="utf-8-sig", newline="")
  log = csv.writer(log_file)
  if new_log:
    log.writerow(LOG_HEADER)
  return log_file, log


def log_row(
  now: datetime, out_path: Path, job: Job, seed: Any, lora: str | None, base: Path, input_path: Path,
) -> list[Any]:
  """生成の記録の 1 行を作る。"""
  s = job.settings
  ref_text = ""
  if s.ref_wavs:
    ref_text = "; ".join(short_path(p, base) for p in s.ref_wavs)
  elif s.ref_embed:
    ref_text = short_path(s.ref_embed, base)
  return [
    now.strftime("%Y-%m-%d %H:%M:%S"), out_path.name, job.text, seed,
    s.caption or "", ref_text, short_path(lora, base) if lora else "",
    str(input_path), job.index,
  ]


# ---------- メイン ----------

def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="YAML ファイルから Irodori-TTS で音声を一括生成します。")
  parser.add_argument("input", help="入力ファイル（.yaml / .yml）")
  parser.add_argument("--overwrite", action="store_true", help="同じ名前のファイルがあれば上書きする")
  parser.add_argument("--dry-run", action="store_true", help="モデルを読まずに、設定と作るファイル名の例だけを表示する")
  return parser


def main() -> None:
  args = build_parser().parse_args()

  input_path = Path(args.input).expanduser().resolve()
  if not input_path.is_file():
    sys.exit(f"[エラー] 入力ファイルが見つかりません: {input_path}")
  opts, jobs = load_input(input_path)
  base: Path = opts["_base"]
  total = sum(job.count for job in jobs)
  out_dir = resolve_path(base, opts["out-dir"], "out-dir")
  print(f"[入力] {input_path}  {len(jobs)} 件 / 合計 {total} 個")
  print(f"[出力] {out_dir}")

  if args.dry_run:
    now = datetime.now()
    for job in jobs:
      first = render_name(job.name_template, now, 1)
      print(f"  {job.index}件目 x{job.count}: {first}.wav ...  「{job.text}」")
      print(f"      {describe(job.settings, base)}")
    return

  # ここから本体を読み込む（読み込みに時間がかかるため、入力の確認を先に済ませる）
  runtime = load_runtime(opts)
  started = time.perf_counter()
  done = 0
  current = 0

  # 生成の記録（あとで同じ seed で作り直せるように）
  log_file, log = open_log(out_dir)
  with log_file:
    for job, serial, seed in iter_items(jobs, opts["_seed"]):
      if job.index != current:
        current = job.index
        print(f"[voice {job.index}] {describe(job.settings, base)}")
      done += 1
      print(f"[{done}/{total}] voice {job.index} ({serial}/{job.count}) 「{job.text}」", flush=True)
      result = synthesize(runtime, job.text, job.settings, seed, job.settings.lora)
      # ファイル名の日時は生成が終わった時刻にする
      now = datetime.now()
      out_path = unique_path(out_dir, render_name(job.name_template, now, serial), args.overwrite)
      save_result(out_path, result)
      log.writerow(log_row(now, out_path, job, result.used_seed, job.settings.lora, base, input_path))
      log_file.flush()
      print(f"  -> {out_path.name}  (seed={result.used_seed})")

  elapsed = time.perf_counter() - started
  print(f"[完了] {done} 個の音声を作りました（{elapsed:.1f} 秒）: {out_dir}")
  print(f"[記録] {out_dir / 'generation_log.csv'}")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n[中断] 途中で止めました。")
    sys.exit(130)
