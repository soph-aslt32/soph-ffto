"""ロボットの作業を基本動作の列に展開するルール（入れ替え動作の手順）."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..ops import Move, Op, Pick, Place, PlaceNext
from ..topology import Position, Robot
from .dispatch import Task

if TYPE_CHECKING:
    from ..engine import Simulator


class ExchangeRule(Protocol):
    def can_exchange(self, robot: Robot, hand: str, pos: Position) -> bool:
        """hand のマテリアルを，処理済みマテリアルが残る pos へ入れ替えで置けるか."""
        ...

    def expand(self, robot: Robot, task: Task, sim: Simulator) -> list[Op]:
        ...


class BasicExchange:
    """ユーザー確認済みの基本手順.

    1. Upperハンドで上流から新しいマテリアルを取る
    2. 処理層へ移動する
    3. Lowerハンドで処理済みのマテリアルを取る
    4. Upperハンドで新しいマテリアルを処理層に置く
    5. 下流へ移動し，Lowerハンドのマテリアルを置く

    置き先に処理済みマテリアルが無ければ入れ替えは行わず，置くだけで作業を終える．
    """

    def can_exchange(self, robot: Robot, hand: str, pos: Position) -> bool:
        other = robot.other_hand(hand)
        stage = pos.stage
        return (
            pos.material is not None
            and pos.finished
            and robot.hands[other].is_empty
            and stage is not None
            and stage.outbound is not None
            and stage.outbound.robot is robot
            and stage.outbound.hand == other
        )

    def expand(self, robot: Robot, task: Task, sim: Simulator) -> list[Op]:
        pos = task.pos
        assert pos.stage is not None
        if task.kind == "fetch":
            return [Move(pos.stage.index, task.hand), Pick(task.hand, pos)]
        if pos.is_empty:
            return [Move(pos.stage.index, task.hand), Place(task.hand, pos)]
        other = robot.other_hand(task.hand)
        next_index = pos.stage.index + 1
        return [
            Move(pos.stage.index, other),
            Pick(other, pos),
            Place(task.hand, pos),
            Move(next_index, other),
            PlaceNext(other, next_index),
        ]
