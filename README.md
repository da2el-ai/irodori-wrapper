# irodori-wrapper

音声生成 AI「[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)」を便利に使うためのラッパースクリプト集です。

Irodori-TTS のインストールから生成、学習まで揃えました。

- **一括生成**: YAML ファイルから音声をまとめて作る（参照音声・キャプション・LoRA も指定できる）
- **絵文字リスト**: 話し方を指定する絵文字の一覧（[EMOJI_LIST.md](EMOJI_LIST.md)）
- **LoRA 学習**: 自分で用意した音声で LoRA を学習する（[README_train.md](README_train.md)）

## ファイル一覧

### ユーザーが直接触るもの

| ファイル | 内容 |
|---|---|
| `setup.bat` | Irodori-TTS 本体を入れる・更新する（`git clone` / `git pull` と `uv sync`） |
| `generate.bat` | 音声ファイル一括生成バッチファイル |
| `lora.bat` | LoRA 学習バッチファイル。ひな形作り・文字起こし・パラメーターの提案・学習・サンプル出力 |
| `README_train.md` | LoRA 学習の方法 |
| `EMOJI_LIST.md` | 絵文字リスト |
| `sample.yaml` | 一括生成の設定サンプル |
| `.env.example` | `.env` のひな形（コピーして `.env` を作る） |
| `.env` | 設定（ベースモデル、FFmpeg の場所） |

### 直接触る必要がないもの

| ファイル | 内容 |
|---|---|
| `Irodori-TTS/` | Irodori-TTS 本体（`setup.bat` が clone する。`.venv` もこの中に作る） |
| `scripts/generate.py` | 一括生成スクリプト |
| `scripts/lora.py` | LoRA 学習スクリプト（`lora_suggest.py` / `lora_sample.py` / `transcribe.py` を使う） |
| `scripts/run_upstream.py` | Irodori-TTS 本体のスクリプトを、互換のための設定（pandas 3.0 対策）をしてから動かす |
| `scripts/irodori_common.py` | 共通の処理（`.env` の読み込み、FFmpeg の設定など） |
| `CLAUDE.md` | Claude Code 向けの作業の決まり |
| `.claude/` | Claude Code のスキル（学習素材の文字起こしと整形）と、学習オプションの決め方のメモ（`knowledge_train.md`） |
| `pyrightconfig.json` | エディター（VS Code の Pylance など）が Irodori-TTS 本体の `.venv` と `irodori_tts` を見つけるための設定 |


## 必要環境

- [git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/)

## インストールと準備

irodori-wrapper をクローンします。

```bat
git clone https://github.com/da2el-ai/irodori-wrapper.git
```

フォルダに入って `setup.bat` を実行します。

```bat
cd irodori-wrapper
.\setup.bat
```

> PowerShell では、今いるフォルダの bat を実行するときに、`.\setup.bat` のように先頭に `.\` を付けます（cmd では付けなくても動きます）。以下の説明では `.\` を省いています。

### setup.bat のオプションについて

通常はオプションは不要ですが、GPU を使えない環境や、`os error` が発生した時はオプションが必要です。

| 状況 | 実行するコマンド |
|---|---|
| NVIDIA GPU を使う（ふつう） | `setup.bat` |
| GPU を使わない | `setup.bat cpu` |
| 上の方法で `os error 448` が出た | `setup.bat cu128 ＜インストールされている Python のバージョン＞`<br>例: `setup.bat cu128 3.11` |

  - バージョンは **3.10 か 3.11** で、パソコンに入っているものを書きます
  - 3.12 以上は使えません

### Irodori-TTS 本体のアップデートについて

インストール時と同じ引数で `setup.bat` を実行します。引数なしで入れたときは `setup.bat`、`setup.bat cu128 3.11` で入れたときは同じ `setup.bat cu128 3.11` です。`Irodori-TTS\` がすでにあれば、`git pull` と `uv sync` だけを行います。

<details>
<summary>その他、AIに読ませると良い詳細情報</summary>

- Irodori-TTS 本体を `Irodori-TTS\` に clone し、その中で `uv sync --extra cu128` を実行して `.venv` を作ります
- PyTorch の種類は `cu128`（NVIDIA GPU。既定）/ `cpu` / `rocm` / `xpu` から選びます（Irodori-TTS 本体の README と同じです）
- 引数なしのときは、uv が Python 3.10 を自分でダウンロードして使います。パソコンに Python が入っていなくてもかまいません
- Python のバージョンを書くと、uv の Python ではなく、パソコンに入れた Python（python.org 版など）を使います
  - バージョンは **3.10 か 3.11** で、パソコンに入っているものを書きます
  - 3.12 以上は使えません。Irodori-TTS 本体が使う `sentencepiece` 0.1.99 には、Windows の Python 3.12 以上向けの完成品（wheel）がなく、ソースからのビルドに失敗するためです
  - 3.11 では pandas 3.0 が入ります（3.10 では 2.x）。pandas 3.0 のままだと、latent を作るときに datasets が `metadata.csv` を読めません（`` `file_name` or `*_file_name` must be present `` のエラー）。そのため `lora.bat latent` は、pandas の設定を以前の読み方に戻してから Irodori-TTS 本体を動かします（`scripts/run_upstream.py`）
  - 3.10 / 3.11 が入っていないときは、python.org から入れます。インストーラーの「Add python.exe to PATH」のチェックを外しておけば、今までの Python（`python` コマンド）はそのまま使えます
    - 3.11 はセキュリティ修正だけの時期に入っており、新しい 3.11.x には Windows 用のインストーラーがありません。Windows 用のインストーラーがある最後の版、[3.11.9](https://www.python.org/downloads/release/python-3119/) の「Windows installer (64-bit)」を使います
    - 入れたあとに `uv python list --no-managed-python --only-installed` を実行し、一覧に `cpython-3.11.9` があることを確かめてから `setup.bat cu128 3.11` を実行します
  - バージョンを書くときは、1 つ目の PyTorch の種類（`cu128` など）も省略できません
- `.venv` はフォルダを移すと壊れることがあります。別の場所に入れた Irodori-TTS 本体を `Irodori-TTS\` に移さず、`setup.bat` で入れ直してください

</details>


### 設定ファイル（.env）

`.env.example` をコピーして `.env` を作り、自分の環境に合わせて書き換えます。

```ini
# 音声生成と LoRA 学習で使うベースモデル (Hugging Face のリポジトリ ID)
IRODORI_HF_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small

# FFmpeg (shared 版) の bin フォルダ
FFMPEG_SHARED_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\...\bin

# LoRA の学習データ作成（latent）で使う datasets のキャッシュの場所
# HF_DATASETS_CACHE=.cache\datasets
```

- `#` で始まる行はコメントです
- 値の中の `%VAR%`（例: `%LOCALAPPDATA%`）は、環境変数に置き換わります
- 同じ名前の環境変数がすでにあるときは、環境変数のほうを使います
- `IRODORI_HF_CHECKPOINT` を書き換えると、派生モデルも使えます
- `HF_DATASETS_CACHE` は、latent を作るときの datasets のキャッシュの場所です。書かなければ `irodori-wrapper\.cache\datasets` を使います。相対パスは `irodori-wrapper\` から見たパスです

> ユーザーフォルダの既定のキャッシュ（`~\.cache\huggingface\datasets`）に書き込めないと、datasets はエラーを出さずに `Resolving data files: 100%` のまま止まります。それを避けるため、プロジェクトの中に置いています


## 音声の生成

```bat
generate.bat ＜入力.yaml＞ [--dry-run] [--overwrite]
```
例：

```bat
generate.bat sample.yaml
```

| オプション | 役割 |
| --- | --- |
| `--dry-run` | 音声を生成せず設定の確認に使う |
| `--overwrite` | 同じ名前のファイルを上書きする |

### 入力ファイル（YAML）

```yaml
setting:                     # 全体の設定（すべての voice に使う）
  ref-wav: voices/           # 参照音声
  caption: "落ち着いた、近い距離感の女性話者"
  lora: lora_study/名前/output/train/checkpoint_final
  out-dir: outputs

voice:                       # 作る音声の一覧
  - text: こんにちは😊
  - text: えっ、本当に？😲
    count: 3
    filename: "{%Y%m%d}_驚き"
  - text: ねえ、聞いて🤭
    setting:                 # この voice だけの設定
      caption: "ささやくような話し方"
      lora: none
```

- 入力例は [sample.yaml](sample.yaml) にあります。YAML ファイルは UTF-8 で保存します。
- `text` `filename` などは `""` で囲むほうが安全です。特に `{` で始まる値は囲まないとエラーになります

#### voice の項目

| 項目 | 必須 | 内容 |
|---|---|---|
| `text` | ○ | 生成するセリフのテキスト |
| `count` | | 生成数。省略すると `1` |
| `filename` | | ファイル名の書式。省略すると `{%Y%m%d-%H%M%S}` |
| `setting` | | この voice だけの設定 |

**ファイル名のルール:**

- `{}` の中に Python の strftime の書式を書くと、**生成した時刻**に置き換わります（例: `{%Y%m%d-%H%M%S}` → `20261121-121300`）
  - 月は `%m`、分は `%M` です。大文字と小文字で意味が変わるので気をつけてください
- 末尾には必ず `_0001` のような 4 桁の連番が付きます。連番は voice ごとに 1 から数えます
  - 例: `"挨拶_{%Y%m%d-%H%M%S}"` → `挨拶_20261121-121300_0001.wav`
- ファイル名に使えない文字（`\ / : * ? " < > |`）は `_` に置き換えます
- 同じ名前のファイルがあるときは上書きせず、`内緒話_0001-2.wav` のように `-2`, `-3` … を付けます

#### setting の項目

- 全体の `setting` と、voice の中の `setting` の両方に書けます
- voice の中に書いた項目は、その音声だけに適用されます

**全体の `setting` と voice の `setting` の両方に書ける項目：**

| 項目 | 内容 |
|---|---|
| `ref-wav` | 参照音声（声をまねる元の音声）。ファイル1つ、フォルダ、ファイルのリストのどれかで書く。フォルダのときは、直下の wav / flac / mp3 / ogg を名前順にすべて使う（サブフォルダは見ない）。複数指定可能。複数指定の書式は [sample.yaml](sample.yaml) を参考にしてください |
| `ref-embed` | Speaker Inversion の埋め込み（`.speaker.safetensors`）。`ref-wav` とは同時に使えない |
| `caption` | 声やスタイルの説明文 |
| `lora` | LoRA アダプターのフォルダ（1つだけ） |
| `seed` | seed。1つ作るたびに +1 する（省略するとランダム） |
| `num-steps` | サンプリングのステップ数 |
| `seconds` | 音声の長さ（秒）。省略すると自動 |
| `duration-scale` | 自動の長さに掛ける倍率（既定: 1.0） |
| `cfg-scale-text` / `cfg-scale-caption` / `cfg-scale-speaker` | 各条件の強さ（既定: 3.0 / 3.0 / 5.0） |
| `length-check` | `on` にすると、長すぎる音声を作り直す 2 パス生成をする |

**全体の `setting` だけに書ける項目：**

| 項目 | 内容 |
|---|---|
| `out-dir` | 出力フォルダ（既定: YAML と同じフォルダの `outputs`） |
| `checkpoint` / `hf-checkpoint` | 使うモデル（既定: `.env` の `IRODORI_HF_CHECKPOINT`） |
| `device` | 使うデバイス（`cuda` / `cpu` など。既定: 自動） |
| `precision` | 計算精度（`fp32` / `bf16`。既定: `fp32`） |

**細かい決まり:**

- 相対パスは、**YAML ファイルがあるフォルダ**から見たパスとして扱います
- `ref-wav` と `ref-embed` は「参照」として 1 組で扱います。voice でどちらかを書くと、全体の参照はどちらも使いません
- voice の中で `ref-wav: none` や `lora: none` と書くと、その voice だけ参照なし・LoRA なしにできます。`caption:` のように値を空にしても、なしになります
- 全体の `seed` は、voice をまたいで 1 つ作るたびに +1 します。voice の中に `seed` を書くと、その voice はその値から数え直します
- 参照音声を指定しないときは、参照なしで生成します
- 知らない項目名を書いたときは、エラーにして止めます
- 各パラメーターの詳しい意味は、Irodori-TTS 本体の `Irodori-TTS/docs/parameters.md` を見てください

### 出力

- `out-dir` のフォルダに wav ファイルを作ります
- 同じフォルダの `generation_log.csv` に、ファイル名・テキスト・seed・caption・参照・LoRA を記録します。同じ設定で `seed` を指定すると、同じ音声を作り直せます

## 絵文字

テキストに絵文字を入れると、話し方や声以外の音（息、笑い声など）を指定できます。一覧は [EMOJI_LIST.md](EMOJI_LIST.md) にあります。

## 音声 LoRA の学習

[README_train.md](README_train.md) を参照してください。


## 注意

- `.bat` ファイルは Shift_JIS（cp932）で保存しています。UTF-8 で保存し直すと、日本語のコメントで cmd がエラーになります。
- このラッパーは Irodori-TTS 本体のファイルを変更しません
- Irodori-TTS 本体を更新したあと（`setup.bat` のあと）は、引数や絵文字の一覧が変わっていないか確かめてください
- `setup.bat` の `uv sync` が `os error 448`（信頼されていないマウントポイント）で失敗する場合があります
  - 原因: Irodori-TTS 本体の `.python-version` に `3.10` と書かれているため、uv は自分で管理する Python 3.10 を使おうとします。uv はその Python の入口を**ジャンクション**（フォルダへのショートカットのようなもの）で作りますが、Windows のセキュリティ機能がジャンクションをたどるのを止めることがあります
  - このエラーが出るかどうかは、Windows の版やセキュリティの設定で変わります。まず引数なしで実行し、出たときだけ次の対処をしてください
  - 対処: パソコンに入れた Python 3.10 か 3.11 を使います。例: `setup.bat cu128 3.11`（3.12 以上は、上の「Irodori-TTS 本体のアップデートについて」に書いた理由で使えません）
  - 更新するときも、同じ引数で `setup.bat` を実行してください。引数を付けないと、また uv の Python を使おうとします

