"""推定で共通に使う統計処理."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable, Sequence

MIN_COUNT = 5  # 「件数の多い最小の値」とみなす件数


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """昇順の列の分位点（線形補間）."""
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def center(values: Sequence[float], width: float) -> float:
    """中央値から ±width に入る値の平均．外乱で伸びた値を除く."""
    median = statistics.median(values)
    return statistics.fmean(v for v in values if abs(v - median) <= width)


def lowest_common(values: Iterable[float]) -> float | None:
    """件数が MIN_COUNT 以上の，最小の値．打刻のはずれ値と待ちの両方を避ける."""
    counts = Counter(values)
    return next((v for v in sorted(counts) if counts[v] >= MIN_COUNT), None)


def low_center(values: Sequence[float], width: float) -> float:
    """待ちを含みうる区間の，待ちの無い側の中心.

    待ちが無ければ同じ値に集まり，待ちがあれば長い側へ伸びる．最も短い山の平均を取る．
    """
    floor = lowest_common(values)
    if floor is None:
        floor = min(values)
    return statistics.fmean(v for v in values if floor <= v <= floor + width)


def mean_interval(times: Sequence[float], burst: int = 1) -> float | None:
    """連続する時刻の間隔の平均．停止などの長い間隔は除く.

    並列層では投入が束になり，下流の単層の Pos にも束のまま届く．中央値は実際の間隔より短くなるので，平均を取る．
    長い間隔の判定も，束の周期 burst（ラインの最大の層数）だけ進む間隔の中央値の3倍を超えるものとする．
    """
    if len(times) < 2:
        return None
    ordered = sorted(times)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    step = min(burst, len(ordered) - 1)
    limit = 3 * statistics.median(b - a for a, b in zip(ordered, ordered[step:]))
    kept = [g for g in gaps if g <= limit]
    return statistics.fmean(kept) if kept else None
