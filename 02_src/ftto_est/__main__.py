"""コマンドラインからログを解析する.

例: python -m ftto_est --logs out/sample --out out/estimate
"""

from __future__ import annotations

import argparse
import sys
import time

from .pipeline import estimate_dir
from .report import format_summary, write_report
from .settings import Settings, load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ftto_est", description="ログから各部位の能力と余裕を求める")
    parser.add_argument("--logs", required=True, help="log_x/a/b/c.csv のあるディレクトリ")
    parser.add_argument("--out", required=True, help="出力先ディレクトリ")
    parser.add_argument("--settings", help="信号の時間と原点からの移動時間の設定（TOML）．ftto_est/settings.py を参照")
    args = parser.parse_args(argv)

    began = time.perf_counter()
    try:
        settings = load_settings(args.settings) if args.settings else Settings()
        result = estimate_dir(args.logs, settings)
    except (OSError, ValueError, KeyError) as e:
        print(f"ログの読み込みに失敗しました: {e!r}", file=sys.stderr)
        return 2
    write_report(result, args.out)
    print(format_summary(result))
    print(f"完了: {time.perf_counter() - began:.1f} s，出力先 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
