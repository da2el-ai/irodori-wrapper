"""LoRA のチェックポイントごとに、サンプル音声を作る（lora.py sample から使う）。

- 定義は <学習フォルダ>/sample.yaml（generate.bat の入力 YAML と同じ書き方。lora と out-dir は書けない）
- 出力は <学習フォルダ>/sample/<filename>_<ステップ>_seed<seed>.wav
  - step0000000 は LoRA なし（ベースモデル）、final は checkpoint_final
- 作り終えたチェックポイントは sample/done.txt に記録し、次からは飛ばす
- --watch を付けると、新しいチェックポイントを待って作る。checkpoint_final を作ったら終わる
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import generate as bt

# チェックポイントのフォルダ名（train.py が作る）
CHECKPOINT_RE = re.compile(r"^checkpoint_(\d{7})$")
FINAL_NAME = "checkpoint_final"
# チェックポイントのフォルダで、最後に書かれるファイル
TRAINER_STATE_NAME = "trainer_state.pt"
# 書き終わったとみなすまでの待ち時間（秒）
SETTLE_SECONDS = 5.0
# LoRA なし（ベースモデル）のサンプルの名前
BASE_TAG = "step0000000"
# seed を書かなかったときの値
DEFAULT_SEED = 0
# 作り終えたチェックポイントの記録
DONE_NAME = "done.txt"


def list_checkpoints(train_dir: Path) -> list[tuple[str, Path]]:
  """書き終わったチェックポイントを (名前, フォルダ) でステップ順に返す。final は最後。"""
  if not train_dir.is_dir():
    return []
  steps: list[tuple[int, str, Path]] = []
  now = time.time()
  for path in train_dir.iterdir():
    if not path.is_dir():
      continue
    match = CHECKPOINT_RE.match(path.name)
    if match:
      step, tag = int(match.group(1)), f"step{match.group(1)}"
    elif path.name == FINAL_NAME:
      step, tag = sys.maxsize, "final"
    else:
      continue
    state = path / TRAINER_STATE_NAME
    if not (path / "adapter_config.json").is_file() or not state.is_file():
      continue
    if now - state.stat().st_mtime < SETTLE_SECONDS:
      continue
    steps.append((step, tag, path))
  steps.sort(key=lambda item: item[0])
  return [(tag, path) for _, tag, path in steps]


def read_done(path: Path) -> set[str]:
  if not path.is_file():
    return set()
  return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def label_for(job: bt.Job, now: datetime) -> str:
  """ファイル名の先頭の部分。filename を省いたときは voice1, voice2 ...。"""
  return bt.render_stem(job.name_template, now) if job.name_given else f"voice{job.index}"


def run_sample(project: Path, watch: bool, interval: float, force: bool, base: bool, dry_run: bool) -> None:
  sample_yaml = project / "sample.yaml"
  if not sample_yaml.is_file():
    sys.exit(f"[エラー] {sample_yaml} がありません。lora.bat init で、ひな形を作れます。")
  opts, jobs = bt.load_input(sample_yaml, forbidden=frozenset({"lora", "out-dir"}))
  # 聞き比べのため、seed は必ず決まった値にする
  for job in jobs:
    if job.settings.seed_is_local and job.settings.seed is None:
      sys.exit(f"[エラー] {sample_yaml}: voice {job.index}件目の seed が空です。サンプルでは seed を空にできません。")
  global_seed = DEFAULT_SEED if opts["_seed"] is None else opts["_seed"]
  items = list(bt.iter_items(jobs, global_seed))

  out_dir = project / "sample"
  train_dir = project / "output" / "train"
  done_path = out_dir / DONE_NAME
  done = set() if force else read_done(done_path)
  base_dir: Path = opts["_base"]
  print(f"[入力] {sample_yaml}  {len(jobs)} 件 / チェックポイントごとに {len(items)} 個")
  print(f"[出力] {out_dir}")

  if dry_run:
    now = datetime.now()
    tags = ([BASE_TAG] if base else []) + [tag for tag, _ in list_checkpoints(train_dir)]
    print(f"[予定] {', '.join(tags) if tags else '（チェックポイントはまだありません）'}")
    for job, _serial, seed in items:
      print(f"  {label_for(job, now)}_<ステップ>_seed{seed}.wav  「{job.text}」")
      print(f"      {bt.describe(job.settings, base_dir)}")
    if done:
      print(f"[済み] {', '.join(sorted(done))}（--force で作り直せます）")
    return

  runtime = None
  waiting = False
  log_file, log = bt.open_log(out_dir)
  with log_file:
    while True:
      targets: list[tuple[str, Path | None]] = []
      if base and BASE_TAG not in done:
        targets.append((BASE_TAG, None))
      targets += [(tag, path) for tag, path in list_checkpoints(train_dir) if tag not in done]

      for tag, ckpt in targets:
        if runtime is None:
          # モデルは 1 回だけ読み込む。LoRA はチェックポイントごとに差し替える
          runtime = bt.load_runtime(opts)
        waiting = False
        lora = str(ckpt) if ckpt else None
        print(f"[{tag}] {'LoRA なし（ベースモデル）' if ckpt is None else ckpt}", flush=True)
        started = time.perf_counter()
        try:
          for job, _serial, seed in items:
            result = bt.synthesize(runtime, job.text, job.settings, seed, lora)
            now = datetime.now()
            stem = f"{label_for(job, now)}_{tag}_seed{result.used_seed}"
            out_path = bt.unique_path(out_dir, stem, overwrite=force)
            bt.save_result(out_path, result)
            log.writerow(bt.log_row(now, out_path, job, result.used_seed, lora, base_dir, sample_yaml))
            log_file.flush()
            print(f"  -> {out_path.name}", flush=True)
        except (FileNotFoundError, ValueError) as exc:
          # 学習側がチェックポイントを消したときなど。記録せずに次へ進む
          print(f"[注意] {tag} のサンプルを作れませんでした: {exc}", flush=True)
          continue
        done.add(tag)
        with done_path.open("a", encoding="utf-8") as f:
          f.write(tag + "\n")
        print(f"  （{time.perf_counter() - started:.1f} 秒）", flush=True)

      if not watch:
        break
      if "final" in done:
        print("[完了] checkpoint_final のサンプルを作りました。監視を終わります。")
        break
      if not waiting:
        print(f"[待機] 新しいチェックポイントを待っています（{interval:.0f} 秒ごとに確認。Ctrl+C で止める）: {train_dir}",
              flush=True)
        waiting = True
      time.sleep(interval)

  print(f"[完了] サンプル: {out_dir}")
  print(f"[記録] {out_dir / 'generation_log.csv'}")
