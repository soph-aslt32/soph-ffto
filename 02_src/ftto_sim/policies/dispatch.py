"""ロボットが次に行う作業を選ぶルール（作業の優先順位）.

正式な優先順位はユーザー側で情報収集中．確定までは上流優先（UpstreamFirst）を仮ルールとする．
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..config import HANDS
from ..topology import Position, Robot

if TYPE_CHECKING:
    from ..engine import Simulator


@dataclass(frozen=True, eq=False)
class Task:
    """ロボットの作業.

    fetch:   段の Pos から処理済みマテリアルを空のハンドで取る
    deliver: ハンドのマテリアルを次段の Pos に置く（入れ替えを含みうる）
    """

    kind: str  # "fetch" | "deliver"
    hand: str
    pos: Position
    source_index: int  # 搬送元の段


class DispatchPolicy(Protocol):
    def select(self, robot: Robot, sim: Simulator) -> Task | None:
        ...


def candidate_tasks(robot: Robot, sim: Simulator) -> list[Task]:
    """現時点で着手できる作業の一覧."""
    stages = sim.topology.stages
    tasks = []
    for hand in HANDS:
        held = robot.hands[hand].material
        if held is None:
            for stage in stages:
                feed = stage.outbound
                if feed is None or feed.robot is not robot or feed.hand != hand:
                    continue
                finished = [p for p in stage.positions if p.material is not None and p.finished]
                if finished:
                    # 同じ段に処理済みが複数あれば，先に処理が終わったものから取る（仮）
                    pos = min(finished, key=lambda p: p.finished_at)
                    tasks.append(Task("fetch", hand, pos, stage.index))
        else:
            target_stage = stages[held.stage_index + 1]
            pos = sim.layer_select.choose(
                target_stage,
                lambda p, h=hand: p.is_empty or sim.exchange.can_exchange(robot, h, p),
            )
            if pos is not None:
                tasks.append(Task("deliver", hand, pos, held.stage_index))
    return tasks


class UpstreamFirst:
    """上流優先（仮ルール）：搬送元の段がより上流の作業を先に行う."""

    def select(self, robot: Robot, sim: Simulator) -> Task | None:
        tasks = candidate_tasks(robot, sim)
        if not tasks:
            return None
        return min(tasks, key=lambda t: (t.source_index, HANDS.index(t.hand)))
