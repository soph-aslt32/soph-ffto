"""推定値と品質の印（algorithm.md 第9章）.

推測で埋めないために，値には必ず印を付ける．区間は lo〜hi で表す．
"""

from __future__ import annotations

from dataclasses import dataclass

ESTIMATED = "推定"
BIASED = "推定（偏りの恐れ）"
INTERVAL_ONLY = "区間のみ"
UPPER_ONLY = "上界のみ"
LOWER_ONLY = "下限のみ"
BORROWED = "借用"
UNCALIBRATED = "未校正"
TRANSIENT = "過渡"
UNOBSERVED = "未観測"


@dataclass(frozen=True, slots=True)
class Value:
    """推定値．value は代表値で，区間のみのときは上界（保守側）を入れる."""

    value: float | None
    lo: float | None
    hi: float | None
    quality: str

    @classmethod
    def of(cls, value: float) -> Value:
        return cls(value, value, value, ESTIMATED)

    @classmethod
    def unobserved(cls) -> Value:
        return cls(None, None, None, UNOBSERVED)

    def plus(self, other: Value) -> Value:
        """和．品質は悪い方を取る."""
        lo = None if self.lo is None or other.lo is None else self.lo + other.lo
        hi = None if self.hi is None or other.hi is None else self.hi + other.hi
        value = None if self.value is None or other.value is None else self.value + other.value
        return Value(value, lo, hi, worse(self.quality, other.quality))

    def scaled(self, factor: float) -> Value:
        return Value(
            None if self.value is None else self.value * factor,
            None if self.lo is None else self.lo * factor,
            None if self.hi is None else self.hi * factor,
            self.quality,
        )


# 右ほど悪い
_RANK = {ESTIMATED: 0, BORROWED: 1, BIASED: 2, UPPER_ONLY: 3, LOWER_ONLY: 3, INTERVAL_ONLY: 4, UNOBSERVED: 5}


def worse(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b
