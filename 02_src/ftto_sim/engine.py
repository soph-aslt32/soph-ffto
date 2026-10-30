"""離散事象シミュレーションの中核."""

from __future__ import annotations

import heapq
import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SimConfig
from .ops import Op
from .policies import build_policies
from .topology import Material, Position, Robot, Topology
from .writers import Recorder


class DeadlockError(RuntimeError):
    """シミュレーション期間内に，どの状態も進まなくなった."""


@dataclass
class SimResult:
    config: SimConfig
    recorder: Recorder
    summary: dict[str, Any]

    def write(self, out_dir: str | Path) -> None:
        self.recorder.write(Path(out_dir), self.summary)


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.topology = Topology.build(config)
        self.dispatch, self.exchange, self.layer_select = build_policies(config.policies)
        # 乱数は用途ごとに分け，ログの設定を変えてもシミュレーションの挙動が変わらないようにする
        self._rng_service = random.Random(f"{config.seed}:service")
        self._rng_input = random.Random(f"{config.seed}:input")
        self.recorder = Recorder(config, self.topology, random.Random(f"{config.seed}:log"))
        self.now = 0.0
        self._queue: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()
        self._n_input = 0
        self._n_completed = 0
        self._input_ready = False
        self._process_id = ""

    # --- 実行 ---

    def run(self) -> SimResult:
        horizon = self.config.duration_s
        self._schedule(0.0, self._on_release)
        while self._queue:
            t, _, callback = heapq.heappop(self._queue)
            if t > horizon:
                break
            self.now = t
            callback()
            self._progress()
        else:
            raise DeadlockError(self._describe_state())
        self.now = horizon
        summary = self.recorder.summary(horizon, self._n_input, self._n_completed)
        return SimResult(self.config, self.recorder, summary)

    def _schedule(self, t: float, callback: Callable[[], None]) -> None:
        heapq.heappush(self._queue, (t, next(self._seq), callback))

    def _progress(self) -> None:
        """現時刻で実行可能な処理を，変化が無くなるまで実行する."""
        changed = True
        while changed:
            changed = self._try_intake()
            changed |= self._try_auto_transfers()
            for robot in self.topology.robots:
                changed |= self._try_robot(robot)

    # --- 取り込み ---

    def _on_release(self) -> None:
        self._input_ready = True

    def _try_intake(self) -> bool:
        if not self._input_ready:
            return False
        first = self.topology.stages[0]
        pos = self.layer_select.choose(first, lambda p: p.is_empty)
        if pos is None:
            return False  # 先頭が埋まっていれば，取り込みは待たされる
        inp = self.config.input
        k = self._n_input
        if inp.process_id_mode == "per_material" or k % inp.cassette_size == 0:
            pids = list(inp.process_id_weights)
            self._process_id = self._rng_input.choices(pids, weights=[inp.process_id_weights[p] for p in pids])[0]
        material = Material(
            material_id=f"M{k + 1:06d}",
            process_id=self._process_id,
            cassette_id=f"C{k // inp.cassette_size + 1:05d}",
        )
        self._n_input += 1
        self._input_ready = False
        self.recorder.intake(self.now, material)
        self.transfer(None, pos, material)
        self._schedule(self._next_release_time(), self._on_release)
        return True

    def _next_release_time(self) -> float:
        """次（self._n_input 番目，0始まり）のマテリアルを取り込める時刻．実際の取り込み時刻から指定間隔後."""
        inp = self.config.input
        gap = inp.cassette_gap_s if self._n_input % inp.cassette_size == 0 else 0.0
        return self.now + inp.interval_s + gap

    # --- マテリアルの移動と段での処理 ---

    def transfer(self, src: Position | None, dst: Position | None, material: Material | None = None) -> None:
        """src から dst へマテリアルを移す．src=None は取り込み，dst=None は払い出し."""
        if src is not None:
            material = src.material
            src.material = None
            src.finished = False
        assert material is not None
        self.recorder.transfer(self.now, material, src, dst)
        if dst is None:
            material.position = None
            self._n_completed += 1
            return
        dst.material = material
        material.position = dst
        if dst.stage is not None:
            material.stage_index = dst.stage.index
            self._start_service(dst, material)

    def _start_service(self, pos: Position, material: Material) -> None:
        assert pos.stage is not None
        service = pos.stage.service
        duration = service.base_time(material.process_id)
        if service.jitter_s > 0:
            duration = max(0.0, self._rng_service.gauss(duration, service.jitter_s))
        self.recorder.service(material, pos, self.now, duration)
        self._schedule(self.now + duration, lambda: self._on_service_end(pos))

    def _on_service_end(self, pos: Position) -> None:
        pos.finished = True
        pos.finished_at = self.now

    def _try_auto_transfers(self) -> bool:
        """ロボットを介さない移動（自動搬送と，最終段からの払い出し）."""
        changed = False
        stages = self.topology.stages
        for stage in stages:
            if stage.outbound is not None and stage.outbound.robot is not None:
                continue
            finished = sorted((p for p in stage.positions if p.material is not None and p.finished),
                              key=lambda p: p.finished_at)
            for pos in finished:
                if stage.outbound is None:
                    self.transfer(pos, None)
                    changed = True
                    continue
                target = self.layer_select.choose(stages[stage.index + 1], lambda p: p.is_empty)
                if target is None:
                    break
                self.transfer(pos, target)
                changed = True
        return changed

    # --- ロボット ---

    def _try_robot(self, robot: Robot) -> bool:
        if robot.busy:
            return False
        if not robot.plan:
            task = self.dispatch.select(robot, self)
            if task is None:
                return False
            robot.plan.extend(self.exchange.expand(robot, task, self))
        changed = False
        while robot.plan and robot.plan[0].is_noop(robot):
            robot.plan.popleft()
            changed = True
        if not robot.plan:
            return changed
        op = robot.plan[0]
        if not op.ready(self, robot):
            return changed  # 条件が整うまでその場で待つ
        robot.plan.popleft()
        robot.busy = True
        t_start = self.now
        duration = op.start(self, robot)
        self._schedule(self.now + duration, lambda: self._on_op_end(robot, op, t_start))
        return True

    def _on_op_end(self, robot: Robot, op: Op, t_start: float) -> None:
        op.complete(self, robot)
        robot.busy = False
        self.recorder.robot_op(robot.robot_id, robot.hands[op.hand], op.hand, op.action,
                               op.target_label(self), t_start, self.now)

    def _describe_state(self) -> str:
        lines = [f"t={self.now:.1f}s でシミュレーションが停止しました（デッドロック）"]
        for p in self.topology.positions:
            if p.material is not None:
                state = "処理済" if p.finished else "処理中"
                lines.append(f"  {p.label}: {p.material.material_id} ({state})")
        for r in self.topology.robots:
            plan = ", ".join(type(op).__name__ for op in r.plan) or "なし"
            lines.append(f"  {r.robot_id}: 位置={self.topology.stages[r.location].name}, 残りの動作={plan}")
        return "\n".join(lines)


def simulate(config: SimConfig) -> SimResult:
    return Simulator(config).run()
