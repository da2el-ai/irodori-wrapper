# irodori-wrapper LoRA 学習方法

irodori-wrapper で LoRA 学習するための方法です。


## 1. FFmpeg（shared 版）を入れる

学習データを作るときに使う `torchcodec` は、FFmpeg 4〜8 の **shared 版**（DLL 付き）でないと動きません。FFmpeg 9 や、DLL が付かない full_build 版では動きません。

```powershell
winget install Gyan.FFmpeg.Shared --version 8.1.2
```

`.env` の `FFMPEG_SHARED_BIN` に書いたフォルダがあれば、ラッパーを動かすときだけ PATH の先頭（と DLL を探す場所）に加えます。別のバージョンを入れたときは、このフォルダを書き換えてください。

## 2. 流れ

LoRA の学習は、すべて `lora.bat` で行います。`lora.bat コマンド -h` で、コマンドごとの使い方を表示できます。

```bat
lora.bat init       学習フォルダ    1. ひな形を作る
（学習フォルダ\train\ に音声ファイルを入れる）
lora.bat transcribe 学習フォルダ    2. 文字起こしして metadata.csv を作る（内容を確かめて直す）
lora.bat suggest    学習フォルダ    3. 学習パラメーターを提案し、train.yaml に書き出す
lora.bat latent     学習フォルダ    4. 学習用データ（latent とマニフェスト）を作る
lora.bat train      学習フォルダ    5. 学習する
lora.bat sample     学習フォルダ --watch   （5 と並行して、別のウィンドウで）サンプルを作る
```

学習フォルダは、LoRA ごとに 1 つ作ります（例: `lora_study\my_voice`）。次項の `init` でひな形を作成できます。

```text
my_voice\                  学習フォルダ（名前は自由）
  train\                   学習素材（音声ファイルと metadata.csv）
    metadata.csv
    001.wav
    ...
  sample.yaml              サンプル出力の定義（init が作る）
  train.yaml               学習オプション（suggest が作る）
  sample\                  サンプル音声（sample が作る）
  output\                  スクリプトが作る
    manifest.jsonl
    latents\
    train\
      checkpoint_0000100\  途中のチェックポイント
      checkpoint_final\    完成した LoRA アダプター
```

## 3. ひな形を作る（init）

```bat
lora.bat init 学習フォルダ
```

- `train\`（空のフォルダ）と `sample.yaml`（説明のコメント付き）を作ります
- 学習フォルダがすでにあるときは、ないものだけを作ります。今あるファイルは上書きしません

## 4. 学習データを用意する

学習素材の音声ファイルと、内容のテキストをペアにした `metadata.csv` を用意します。手動で作成してもよいですが、次で説明する `transcribe` で出力できます。

```csv
file_name,text
001.wav,おはようございます。
002.wav,今日は晴れています。
```

| 列 | 必須 | 内容 |
|---|---|---|
| `file_name` | ○ | 音声ファイルの名前（`metadata.csv` からの相対パス）。wav / flac / mp3 / ogg が使えます |
| `text` | ○ | 音声で話している内容 |
| `caption` | | 声やスタイルの説明文 |
| `speaker` | | 話者の名前。省略すると、全部を同じ話者として扱います |

- 文字コードは UTF-8 で保存する必要があります
- 学習素材にはノイズや音楽、複数の話者が入っていないものが向いています

### 文字起こしで metadata.csv を作る（transcribe）

```bat
lora.bat transcribe 学習フォルダ [--model モデル] [--device cuda] [--overwrite]
```

| オプション | 役割 |
| --- | --- |
| `--model` | 文字起こしに使うモデルを変更します。指定がないと `kotoba-tech/kotoba-whisper-v2.0`（日本語向け）を使います |
| `--device` | 使うデバイス（`cuda` / `cpu` など）。指定がないと自動で選びます |
| `--overwrite` | `metadata.csv` がすでにあっても上書きします。指定がないと、上書きせずに止まります |

- 初回だけモデルをダウンロードします
- FFmpeg（shared 版）が必要です（上の「1. FFmpeg（shared 版）を入れる」を参照）

### テキストについて

- kotoba-whisper は句読点を出しません。また、聞き間違いもあります
- Claude Code で `.claude/skills/make-lora-dataset` スキルを使うと、ある程度修正してくれます
- 学習前に必ず `metadata.csv` を目視確認して、句読点を付けたり、話した内容と合っているか確かめたりしてください
- 吐息、言い淀み、「あの…」のようなフィラー要素も記入すると精度が上がります
- 絵文字を付けると、生成するときに絵文字で話し方を指定しやすくなります（[EMOJI_LIST.md](EMOJI_LIST.md)）


## 5. 学習パラメーターを提案する（suggest）

```bat
lora.bat suggest 学習フォルダ [--epochs 周回数] [--overwrite]
```

- `train\` の音声を調べて、個数・合計の長さ・最短 / 平均 / 最長の長さを表示します。`metadata.csv` があるときは、そこに書かれた音声だけを調べます
- 次の音声があると、ファイル名を挙げて警告します
  - **30 秒を超える音声**: Irodori-TTS 本体は 1 つの音声の 30 秒までしか学習に使いません。残りの部分は捨てられ、テキストと音声が合わなくなります。分けるか、除いてください
  - **1 秒未満の音声**: 短すぎて、学習の効果がほとんどありません
- 学習オプションを計算し、`学習フォルダ\train.yaml` に書き出します

| オプション | 役割 |
| --- | --- |
| `--epochs` | データを何周させるか。指定がないと 200 周です |
| `--overwrite` | `train.yaml` がすでにあっても上書きします。指定がないと、上書きせずに値を表示するだけです |

**計算のしかた**（目安。考え方は [.claude/knowledge_train.md](.claude/knowledge_train.md)）:

- バッチサイズは 8（音声が 32 個より少ないときは、個数 ÷ 4）
- データを `--epochs` の回数だけ周回させる。ステップ数は 100 の倍数に切り上げ、500 以上にします
- 学習率は、全体の 5% で上げ、最後の 25% で下げる
- チェックポイントは、全体の約 1/8 ごとに保存する

## 6. 学習用データ（マニフェスト）を作る（latent）

```bat
lora.bat latent 学習フォルダ [--force] [--name 名前] [--device cuda]
```

- `学習フォルダ\output\` に `manifest.jsonl` と `latents\` ができます

| オプション | 役割 |
| --- | --- |
| `--force` | `manifest.jsonl` がすでにあっても作り直します。指定がないと、エラーで止まります |
| `--name` | 話者 ID に使う名前。指定がないと、学習フォルダの名前を使います |
| `--device` | 使うデバイス（`cuda` / `cpu` など）。指定がないと自動で選びます |

## 7. 学習する（train）

```bat
lora.bat train 学習フォルダ [オプション]
```

- 結果は `学習フォルダ\output\train\` に保存され、`checkpoint_final` が完成した LoRA アダプターです
- `学習フォルダ\train.yaml` があれば、その値を使います。コマンドで同じオプションを書くと、コマンドのほうを優先します
- 学習の設定は、Irodori-TTS 本体の `configs/train_v4_small_lora.yaml` を使います。`train.yaml` にもコマンドにもないオプションは、この設定ファイルの値になります。この値は Irodori-TTS 本体の公式の学習向け（バッチサイズ 40、30000 ステップなど）なので、`suggest` で `train.yaml` を作っておくことをおすすめします
- チェックポイントは、すべて残します（`--checkpoint-best-n 0` を渡します）。Irodori-TTS 本体の既定では、古いチェックポイントが消されて、サンプルで聞き比べられなくなるためです
- ベースモデルは `.env` の `IRODORI_HF_CHECKPOINT` から自動でダウンロードします

`train.yaml` の項目は、下のオプションの `--` を外した名前です（例: `max-steps: 800`）。

| オプション | 内容 |
|---|---|
| `--max-steps` | 学習の総ステップ数 |
| `--batch-size` / `--grad-accum` | バッチサイズ / 勾配をためる回数 |
| `--lr` | 学習率 |
| `--warmup-steps` / `--stable-steps` | 学習率のウォームアップ / 一定に保つステップ数（WSD スケジューラー） |
| `--save-every` / `--log-every` | 保存 / ログの間隔（ステップ） |
| `--num-workers` | データ読み込みのプロセス数 |
| `--valid-ratio` / `--valid-every` | 検証に回すデータの割合 / 検証の間隔 |
| `--lora-r` / `--lora-alpha` | LoRA のランク / alpha |
| `--checkpoint-best-n` | 残すチェックポイントの数の設定（既定: 0 = すべて残す） |
| `--config` | 学習設定ファイル（既定: Irodori-TTS 本体の `configs/train_v4_small_lora.yaml`）。コマンドだけで使えます |
| `--hf-checkpoint` | ベースモデルの Hugging Face ID（既定: `.env` の `IRODORI_HF_CHECKPOINT`）。コマンドだけで使えます |
| `--init-checkpoint` | ベースモデルのファイル（`.safetensors`）。`--hf-checkpoint` の代わりに使います。コマンドだけで使えます |
| `--device` | 使うデバイス（`cuda` / `cpu` など。既定: 自動）。コマンドだけで使えます |
| `--resume フォルダ` | 途中のチェックポイントから再開する（例: `--resume my_voice\output\train\checkpoint_0001000`）。コマンドだけで使えます |
| `-- 引数...` | `--` のあとに書いた引数は、そのまま Irodori-TTS 本体の `train.py` に渡す（例: `lora.bat train my_voice -- --seed 1`）。コマンドだけで使えます |

## 8. チェックポイントごとにサンプルを作る（sample）

```bat
lora.bat sample 学習フォルダ [--watch] [--interval 秒] [--force] [--no-base] [--dry-run]
```

- `学習フォルダ\sample.yaml` の定義で、チェックポイントごとに音声を作り、`学習フォルダ\sample\` に保存します
- `lora.bat train` 実行中に別ウィンドウで `--watch` 付きを実行すると、チェックポイントが作られたらすぐにサンプルボイスを生成します
- ただし学習と同時に動かすと、GPU のメモリを両方で使い、学習が少し遅くなります
- 作り終えたチェックポイントは `sample\done.txt` に記録し、次からは飛ばします。途中で止めても、続きから作れます
- LoRA なし（ベースモデル）の音声も、比べるために作ります（`step0000000`）

| オプション | 役割 |
| --- | --- |
| `--watch` | 学習と並行して動き、新しいチェックポイントができるたびに作ります。`checkpoint_final` の分を作ったら終わります。学習とは別のウィンドウで実行してください。指定がないと、今あるチェックポイントのうち、まだ作っていない分だけを作ります |
| `--interval` | `--watch` で新しいチェックポイントを確かめる間隔（秒）。指定がないと 30 秒です |
| `--force` | 作ったことがあるチェックポイントも作り直します |
| `--no-base` | LoRA なし（`step0000000`）の音声を作りません |
| `--dry-run` | モデルを読まずに、作るファイル名と対象のチェックポイントだけを表示します |

ファイル名は、`<filename>_<ステップ>_seed<seed>.wav` です。文ごとに並ぶので、聞き比べやすくなります。

```text
挨拶_step0000000_seed0.wav     LoRA なし（ベースモデル）
挨拶_step0000100_seed0.wav     checkpoint_0000100
挨拶_final_seed0.wav           checkpoint_final
驚き_step0000000_seed1.wav
...
```

### `sample.yaml` の注意点

書き方は、一括生成の入力 YAML（[README.md](README.md) の「入力ファイル（YAML）」）と同じです。違うのは次の点です。

- `lora` と `out-dir` は書けません（スクリプトが決めます）
- `filename` を省くと、`voice1`、`voice2` … になります。末尾の連番（`_0001`）は付きません
- `seed` を書かないときは、0 から数えます。どのチェックポイントでも同じ seed で作るので、聞き比べられます。voice の中で `seed` を空にすることはできません
- 相対パスは、学習フォルダから見たパスです

## 9. LoRA を使って生成する

入力の YAML の `setting` に、LoRA のフォルダを書きます。

```yaml
setting:
  # YAML ファイルを irodori-wrapper のフォルダに、学習フォルダを lora_study\名前\ に置いた場合の例
  # （相対パスは YAML ファイルがあるフォルダが基準）
  lora: lora_study/名前/output/train/checkpoint_final
```

