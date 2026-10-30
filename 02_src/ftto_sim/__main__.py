"""コマンドラインから合成データを生成する.

例: python -m ftto_sim --config configs/sample.toml --out out/sample
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

from .config import ConfigError, load_config
from .engine import DeadlockError, simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ftto_sim", description="合成ログ（X/A/B/C）と真値を生成する")
    parser.add_argument("--config", required=True, help="設定ファイル（TOML）")
    parser.add_argument("--out", required=True, help="出力先ディレクトリ")
    parser.add_argument("--duration-s", type=float, help="設定ファイルの simulation.duration_s を上書きする")
    parser.add_argument("--seed", type=int, help="設定ファイルの simulation.seed を上書きする")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 2
    overrides = {}
    if args.duration_s is not None:
        overrides["duration_s"] = args.duration_s
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        config = dataclasses.replace(config, **overrides)

    began = time.perf_counter()
    try:
        result = simulate(config)
    except DeadlockError as e:
        print(str(e), file=sys.stderr)
        return 1
    result.write(args.out)
    s = result.summary
    print(f"取り込み {s['n_input']} 枚 / 払い出し {s['n_completed']} 枚 "
          f"（{s['horizon_s']:.0f} s 分，{time.perf_counter() - began:.1f} s で生成）→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
