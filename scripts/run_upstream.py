"""本体のスクリプトを、互換のための設定をしてから動かす。

使い方: python run_upstream.py <本体のスクリプト> [引数...]

- pandas 3.0 は文字列の列を large_string 型で読む。datasets 4.5 の audiofolder は
  metadata.csv の file_name が string 型でないと受け付けないため、
  pandas の future.infer_string を False にして、以前の読み方に戻す
  （Python 3.11 以上では本体の uv.lock により pandas 3.0 が入る。3.10 では 2.x なので影響しない）
- irodori_common を読み込み、FFmpeg の DLL を探す場所などを設定する
  （環境変数の PATH は親から引き継がれるが、os.add_dll_directory は引き継がれない。
  これがないと torchcodec が FFmpeg を読めず、音声のデコードがすべて失敗する）
- 本体のファイルは変えず、このスクリプトの中で __main__ として動かす
- 注意: 本体が mp.spawn で子プロセスを作るとき（prepare_manifest.py の --num-gpus 2 以上など）は、
  子プロセスにこの設定が渡らない
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# .env の読み込み・FFmpeg の DLL の設定・datasets のキャッシュの設定を行う
import irodori_common  # noqa: F401


def apply_compat() -> None:
  """互換のための設定をする。"""
  try:
    import pandas as pd
  except ImportError:
    return
  try:
    pd.set_option("future.infer_string", False)
  except (KeyError, ValueError):
    # この設定がない版の pandas では何もしない
    pass


def main() -> None:
  if len(sys.argv) < 2:
    sys.exit("使い方: python run_upstream.py <本体のスクリプト> [引数...]")
  script = Path(sys.argv[1]).resolve()
  if not script.is_file():
    sys.exit(f"[エラー] スクリプトが見つかりません: {script}")
  apply_compat()
  # 本体のスクリプトを直接動かしたときと同じ argv と sys.path にする
  sys.argv = [str(script)] + sys.argv[2:]
  sys.path.insert(0, str(script.parent))
  runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
  main()
