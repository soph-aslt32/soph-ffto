"""ロボットの待ちの内訳（algorithm.md 第3.2章）.

END から次の START までのうち，移動を除いた時間がロボットの待ちである．1枚あたりで3つに分ける．

- 手空き a：何も持たず，次の動作の対象も無い．余裕
- 処理待ち b：何も持たず，処理層の完了を待っている．余裕
- 保持待ち c：マテリアルを持って，待ち先が空くのを待っている．律速の印
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .actions import PICK, Action, Alignment
from .motion import Motion
from .topology import Topology


@dataclass(slots=True)
class CycleWaits:
    """ロボットが1枚のマテリアルを受け取るまでの1周の待ち.

    material_id は周の最後に受け取ったマテリアル．待ちは，そのマテリアルが来るのを待っていた時間にあたる．
    """

    robot_id: str
    material_id: str
    idle_s: float
    process_s: float
    hold_s: dict[int, float] = field(default_factory=dict)  # 待ち先の段 -> 時間
    t_entry: int = 0  # ログCの時計での受け取り時刻

    @property
    def margin_s(self) -> float:
        return self.idle_s + self.process_s


def _hold_target(holding: dict[str, int], next_action: Action) -> int:
    """保持待ちの待ち先．次の動作が持っているマテリアルの行き先なら，その段.

    そうでなければ，次の動作は別のマテリアルのためのものなので，持っているマテリアルの行き先とする．
    """
    if next_action.stage in holding.values():
        return next_action.stage
    return max(holding.values())


def collect_waits(alignment: Alignment, motion: Motion, topology: Topology) -> dict[str, list[CycleWaits]]:
    result: dict[str, list[CycleWaits]] = {}
    process_stages = {s.index for s in topology.stages if s.kind == "process"}
    for robot_id, actions in alignment.actions.items():
        cycles = motion.cycles.get(robot_id, [])
        if not cycles:
            result[robot_id] = []
            continue
        entry_stage = min(c.upstream for c in cycles)
        threshold = alignment.threshold(robot_id)
        moves = motion.moves[robot_id]
        rows: list[CycleWaits] = []
        idle = process = 0.0
        hold: Counter[int] = Counter()
        holding: dict[str, int] = {}  # ハンド名 -> 持っているマテリアルの行き先の段
        for a, b in zip(actions, actions[1:]):
            if a.kind == PICK:
                holding[a.hand.name] = a.stage + 1
            else:
                holding.pop(a.hand.name, None)
            if not b.resync:
                wait = b.start - a.end - moves.time(a.stage, b.stage)
                if holding:
                    if wait >= threshold:
                        hold[_hold_target(holding, b)] += wait
                elif b.kind == PICK and b.stage in process_stages:
                    process += max(wait, 0.0)
                else:
                    idle += max(wait, 0.0)
            if b.kind == PICK and b.stage == entry_stage:
                rows.append(CycleWaits(robot_id, b.material_id, idle, process, dict(hold), b.end))
                idle = process = 0.0
                hold = Counter()
        result[robot_id] = rows
    return result
