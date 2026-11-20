# irodori-wrapper の作業の決まり

Irodori-TTS のラッパースクリプト集。使い方は [README.md](README.md) を見る。

## 構成

- `Irodori-TTS\` は本体（`setup.bat` で clone する。git では管理しない）。**本体のファイルは変更しない**
- スクリプトは本体の `.venv` の Python（`Irodori-TTS\.venv\Scripts\python.exe`）で動かす。bat はそれを呼ぶだけ
- 共通の処理（`.env` の読み込み、UTF-8 化、FFmpeg の設定、本体の場所）は `scripts\irodori_common.py` にまとめる

## ファイルの決まり

- `.bat` は Shift_JIS（cp932）+ CRLF で保存する。UTF-8 で保存すると、日本語のコメントで cmd がエラーになる。編集は PowerShell の `Set-Content -Encoding Default` で行う
- `.py` / `.md` / `.yaml` / `.env` は UTF-8。Python のインデントは 2 スペース、コメントは日本語

## 実行の決まり

- 学習・latent 作成・文字起こしなど、GPU を使う重いコマンドは、ユーザーが自分で実行する。Claude はコマンドを伝えるだけにする（頼まれたときを除く）
- 学習中は GPU を使う確認をしない。確認は `--help` や `--dry-run` にとどめる

## 公開について

- このリポジトリは GitHub に公開する。個人の情報（ユーザー名を含むパス、作業の状況、学習素材の名前など）を、公開するファイルに書かない
- 個人のメモは `CLAUDE.local.md` に、作業用のファイルは `local\` に置く（どちらも `.gitignore` に入れてある）
