"""ロボットが次に行う作業を選ぶルール（作業の優先順位）.

実機の制御ロジックはユーザー側で情報収集中なので，優先順位は設定ファイルの [policies.priority] で変えられる．
既定値はユーザー確認済みの動作である．

- 信号を受けた段階で，処理層から取り出す動作が確定する
- 処理済みのマテリアルを持つと，下流への搬送が確定する
- 上流へ次のマテリアルを取りに行くのは，そのあとになる
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..config import HANDS, LOWER, PolicyConfig
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

    def preempt(self, robot: Robot, sim: Simulator, hand: str) -> Task | None:
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


class PriorityDispatch:
    """実行できる作業のうち，優先順位の高いものを選ぶ.

    実機の制御ロジックは情報収集中なので，優先順位と確定の条件は [policies.priority] で設定する．
    同じ種類の作業が複数あれば，搬送元の段が上流のものを先に行う．
    """

    def __init__(self, policy: PolicyConfig) -> None:
        self._rank = {name: i for i, name in enumerate(policy.task_order)}
        self._commit_on_signal = policy.commit_on_signal
        self._commit_delivery = policy.commit_delivery
        self.waits_at_target = policy.advance_to_target

    def _key(self, task: Task) -> tuple[int, int]:
        return (self._rank[f"{task.kind}_{task.hand}"], task.source_index)

    def select(self, robot: Robot, sim: Simulator) -> Task | None:
        tasks = candidate_tasks(robot, sim)
        if self._commit_delivery and not robot.hands[LOWER].is_empty:
            for task in tasks:
                if task.kind == "deliver" and task.hand == LOWER:
                    return task
            return self._advance(robot, sim, (LOWER,))
        if not tasks:
            return self._advance(robot, sim, HANDS) or self._go_home(robot)
        return min(tasks, key=self._key)

    def preempt(self, robot: Robot, sim: Simulator, hand: str) -> Task | None:
        """何も持たず，hand で取り出す処理の完了を待っている間に，より優先の作業へ移るか."""
        if self._commit_on_signal or any(not robot.hands[h].is_empty for h in HANDS):
            return None
        waiting = self._rank[f"fetch_{hand}"]
        tasks = [t for t in candidate_tasks(robot, sim) if self._key(t)[0] < waiting]
        return min(tasks, key=self._key, default=None)

    def _advance(self, robot: Robot, sim: Simulator, hands: tuple[str, ...]) -> Task | None:
        """持ったまま置き先の段の前まで移動する."""
        if not self.waits_at_target:
            return None
        for hand in hands:
            held = robot.hands[hand].material
            if held is None:
                continue
            target = sim.topology.stages[held.stage_index + 1]
            if robot.location != target.index:
                return Task("advance", hand, target.positions[0], held.stage_index)
        return None

    def _go_home(self, robot: Robot) -> Task | None:
        """どの動作も要求されていなければ，原点で待つ（ユーザー確認済み）."""
        if not robot.config.has_home or robot.location == HOME:
            return None
        if any(not robot.hands[hand].is_empty for hand in HANDS):
            return None
        return Task("home", HANDS[0], None, -1)
