"""ロボットが次に行う作業を選ぶルール（作業の優先順位）.

正式な優先順位はユーザー側で情報収集中．確定までは上流優先（UpstreamFirst）を仮ルールとする．
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..config import HANDS
from ..topology import HOME, Position, Robot

if TYPE_CHECKING:
    from ..engine import Simulator


@dataclass(frozen=True, eq=False)
class Task:
    """ロボットの作業.

    fetch:   段の Pos から処理済みマテリアルを空のハンドで取る
    deliver: ハンドのマテリアルを次段の Pos に置く（入れ替えを含みうる）
    advance: マテリアルを持ったまま，置き先の段の前まで移動する
    home:    どの動作も要求されていないので，原点へ戻る
    """

    kind: str  # "fetch" | "deliver" | "advance" | "home"
    hand: str
    pos: Position | None  # home は None
    source_index: int  # 搬送元の段．home は -1


class DispatchPolicy(Protocol):
    waits_at_target: bool  # 置き先が処理中のとき，置き先の前で待つか

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
                # 受け入れ可能の信号は取り出し可能の信号でもある．処理の完了前でも，信号を受けて層の前へ向かう
                ready = [p for p in stage.positions if p.material is not None and p.accepting]
                if ready:
                    # 同じ段に複数あれば，先に信号を出したものから取る（仮）
                    pos = min(ready, key=lambda p: p.accepting_at)
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
    """上流優先（仮ルール）：搬送元の段がより上流の作業を先に行う.

    着手できる作業が無ければ，マテリアルを持ったまま置き先の段の前まで移動して待つ（ユーザー確認済み）．
    """

    waits_at_target = True

    def select(self, robot: Robot, sim: Simulator) -> Task | None:
        tasks = candidate_tasks(robot, sim)
        if not tasks:
            return self._advance(robot, sim) or self._go_home(robot)
        return min(tasks, key=lambda t: (t.source_index, HANDS.index(t.hand)))

    def _go_home(self, robot: Robot) -> Task | None:
        """どの動作も要求されていなければ，原点で待つ（ユーザー確認済み）."""
        if not robot.config.has_home or robot.location == HOME:
            return None
        if any(not robot.hands[hand].is_empty for hand in HANDS):
            return None
        return Task("home", HANDS[0], None, -1)

    def _advance(self, robot: Robot, sim: Simulator) -> Task | None:
        for hand in HANDS:
            held = robot.hands[hand].material
            if held is None:
                continue
            target = sim.topology.stages[held.stage_index + 1]
            if robot.location != target.index:
                return Task("advance", hand, target.positions[0], held.stage_index)
        return None


class UpstreamFirstWaitAtSource(UpstreamFirst):
    """比較用：置き先が処理中なら，マテリアルを取った場所で待ち，空いてから移動する."""

    waits_at_target = False

    def _advance(self, robot: Robot, sim: Simulator) -> Task | None:
        return None
