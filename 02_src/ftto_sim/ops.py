"""ロボットの基本動作（移動・取る・置く）.

差し替え可能なルール（policies）は，これらの基本動作の列としてロボットの作業を表現する．
各動作は「実行可能になるまで待つ → 開始 → 所要時間後に完了」の順で処理される．
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Simulator
    from .topology import Position, Robot


class Op:
    """基本動作の基底クラス."""

    action: str
    hand: str  # ログCに記録するハンド

    def is_noop(self, robot: Robot) -> bool:
        return False

    def ready(self, sim: Simulator, robot: Robot) -> bool:
        raise NotImplementedError

    def start(self, sim: Simulator, robot: Robot) -> float:
        """動作を開始し，所要時間を返す."""
        raise NotImplementedError

    def complete(self, sim: Simulator, robot: Robot) -> None:
        raise NotImplementedError

    def target_label(self, sim: Simulator) -> str:
        raise NotImplementedError


@dataclass(eq=False)
class Move(Op):
    stage_index: int
    hand: str
    action = "MOVE"

    def is_noop(self, robot: Robot) -> bool:
        return robot.location == self.stage_index

    def ready(self, sim: Simulator, robot: Robot) -> bool:
        return True

    def start(self, sim: Simulator, robot: Robot) -> float:
        return robot.move_time(sim.topology.stages, robot.location, self.stage_index)

    def complete(self, sim: Simulator, robot: Robot) -> None:
        robot.location = self.stage_index

    def target_label(self, sim: Simulator) -> str:
        return sim.topology.stages[self.stage_index].name


@dataclass(eq=False)
class Pick(Op):
    """段の Pos から，処理済みのマテリアルをハンドで取る."""

    hand: str
    pos: Position
    action = "PICK"

    def ready(self, sim: Simulator, robot: Robot) -> bool:
        return robot.hands[self.hand].is_empty and self.pos.material is not None and self.pos.finished

    def start(self, sim: Simulator, robot: Robot) -> float:
        return robot.config.pick_s

    def complete(self, sim: Simulator, robot: Robot) -> None:
        sim.transfer(self.pos, robot.hands[self.hand])

    def target_label(self, sim: Simulator) -> str:
        return self.pos.label


@dataclass(eq=False)
class Place(Op):
    """ハンドのマテリアルを，指定した空きの Pos に置く."""

    hand: str
    pos: Position
    action = "PLACE"

    def ready(self, sim: Simulator, robot: Robot) -> bool:
        return robot.hands[self.hand].material is not None and self.pos.is_empty

    def start(self, sim: Simulator, robot: Robot) -> float:
        return robot.config.place_s

    def complete(self, sim: Simulator, robot: Robot) -> None:
        sim.transfer(robot.hands[self.hand], self.pos)

    def target_label(self, sim: Simulator) -> str:
        return self.pos.label


@dataclass(eq=False)
class PlaceNext(Op):
    """ハンドのマテリアルを，指定段の空いている Pos に置く．置き先は実行時に層選択ルールで決める."""

    hand: str
    stage_index: int
    action = "PLACE"
    _chosen: Position | None = None

    def ready(self, sim: Simulator, robot: Robot) -> bool:
        if robot.hands[self.hand].material is None:
            return False
        stage = sim.topology.stages[self.stage_index]
        self._chosen = sim.layer_select.choose(stage, lambda p: p.is_empty)
        return self._chosen is not None

    def start(self, sim: Simulator, robot: Robot) -> float:
        return robot.config.place_s

    def complete(self, sim: Simulator, robot: Robot) -> None:
        assert self._chosen is not None
        sim.transfer(robot.hands[self.hand], self._chosen)

    def target_label(self, sim: Simulator) -> str:
        return self._chosen.label if self._chosen else ""
