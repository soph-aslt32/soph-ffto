"""段の中で，マテリアルを置く Pos（並列の処理層など）を選ぶルール."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..topology import Position, Stage


class LayerSelectPolicy(Protocol):
    def choose(self, stage: Stage, acceptable: Callable[[Position], bool]) -> Position | None:
        """acceptable を満たす Pos のうち 1 つを返す．無ければ None."""
        ...


class LowestIndex:
    """インデックスが若い順に，条件を満たす Pos を選ぶ（ユーザー確認済みのルール）."""

    def choose(self, stage: Stage, acceptable: Callable[[Position], bool]) -> Position | None:
        for pos in stage.positions:
            if acceptable(pos):
                return pos
        return None
