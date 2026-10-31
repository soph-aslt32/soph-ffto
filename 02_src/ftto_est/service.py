"""段での処理時間 s（algorithm.md 第5章，第6.3章）.

処理の完了はログに記録されない．ただしロボットは処理済みのマテリアルがあれば待たないので，
待っていたロボットが動き出した時刻が，処理の完了にあたる．
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .actions import PICK, PLACE, Action, Alignment
from .logs import PosKey
from .motion import Motion
from .quality import BIASED, ESTIMATED, INTERVAL_ONLY, UPPER_ONLY, Value
from .stats import center, low_center
from .topology import Stage, Topology

MIN_ADOPTION = 0.5  # 手空きサンプルの採用率がこれ未満なら，偏りの恐れがある


@dataclass
class Samples:
    idle: list[float] = field(default_factory=list)  # 手空きサンプル
    idle_times: list[float] = field(default_factory=list)  # 手空きサンプルの時刻（置かれた時刻）
    upper: list[float] = field(default_factory=list)  # 取り始めた時点で処理は終わっていた
    lower: list[float] = field(default_factory=list)  # 手空きの間は処理が終わっていなかった


@dataclass
class ServiceTime:
    """段 × 処理ID の処理時間．層ごとの値は by_pos に持つ."""

    value: Value
    n_idle: int
    n_total: int

    @property
    def adoption(self) -> float:
        return self.n_idle / self.n_total if self.n_total else 0.0


@dataclass
class ServiceTimes:
    by_stage: dict[tuple[int, str], ServiceTime]
    by_pos: dict[tuple[PosKey, str], ServiceTime]
    samples: dict[tuple[int, str], Samples]  # 能力の時系列の監視に使う
    warnings: list[str] = field(default_factory=list)


def _summarize(samples: Samples, width: float, stage: Stage, last_index: int) -> ServiceTime:
    n_idle, n_total = len(samples.idle), len(samples.upper)
    if n_total == 0:
        return ServiceTime(Value.unobserved(), 0, 0)
    upper = statistics.median(samples.upper)
    lower = statistics.median(samples.lower)
    if n_idle == 0:
        return ServiceTime(Value(upper, min(lower, upper), upper, INTERVAL_ONLY), 0, n_total)
    auto = stage.outbound is None and stage.index != last_index
    estimate = low_center(samples.idle, width) if auto else center(samples.idle, width)
    # 次のロボットが離れた位置で待っていると移動時間が混ざる．自動搬送は次の段の詰まりが混ざる
    if auto or (stage.kind != "process" and stage.index != last_index):
        return ServiceTime(Value(estimate, min(lower, estimate), estimate, UPPER_ONLY), n_idle, n_total)
    if n_idle / n_total < MIN_ADOPTION:
        return ServiceTime(Value(estimate, min(lower, estimate), max(upper, estimate), BIASED), n_idle, n_total)
    return ServiceTime(Value(estimate, estimate, estimate, ESTIMATED), n_idle, n_total)


def _collect_for_robot(
    robot_id: str,
    actions: list[Action],
    alignment: Alignment,
    motion: Motion,
    topology: Topology,
    arrivals: dict[tuple[str, int], float],
    pid_of: dict[str, str],
    by_stage: dict[tuple[int, str], Samples],
    by_pos: dict[tuple[PosKey, str], Samples],
) -> None:
    threshold = alignment.threshold(robot_id)
    moves = motion.moves[robot_id]
    # 動作 i の後で持っているマテリアルの行き先の段
    held_after: list[frozenset[int]] = []
    holding: dict[str, int] = {}
    for a in actions:
        if a.kind == PICK:
            holding[a.hand.name] = a.stage + 1
        else:
            holding.pop(a.hand.name, None)
        held_after.append(frozenset(holding.values()))

    for i, b in enumerate(actions):
        if b.kind == PLACE or i == 0:
            continue
        pid = pid_of.get(b.material_id)
        t0 = arrivals.get((b.material_id, b.stage))
        if pid is None or t0 is None:
            continue
        is_process = topology.stages[b.stage].kind == "process"

        # 別の行き先のマテリアルを持って閉塞しているロボットは，取れる状態ではない
        waiting_for = {b.stage}
        a = actions[i - 1]
        mv = moves.time(a.stage, b.stage)
        idle = None
        if not b.resync and held_after[i - 1] <= waiting_for and b.start - a.end - mv >= threshold:
            # 処理層以外は，離れた位置で待っていた移動を引かない（上界になる）
            at_stage = a.stage == b.stage or a.kind == PICK or not is_process
            idle = b.start - t0 - (0.0 if at_stage else mv)

        # 最後に手空きだった時刻．置かれた時刻までさかのぼる
        last_idle = t0
        for j in range(i, 0, -1):
            x, y = actions[j - 1], actions[j]
            if not y.resync and held_after[j - 1] <= waiting_for:
                step = moves.time(x.stage, y.stage)
                if y.start - x.end - step >= threshold:
                    last_idle = max(last_idle, y.start - step)
                    break
            if y.resync or x.end < t0:
                break

        for target in (by_stage[(b.stage, pid)], by_pos[(b.pos, pid)]):
            target.upper.append(b.start - t0)
            target.lower.append(last_idle - t0)
            if idle is not None:
                target.idle.append(idle)
                target.idle_times.append(t0)


def _final_dwell(topology: Topology, pid_of: dict[str, str], by_stage: dict, by_pos: dict) -> None:
    """ロボットが取らない段（最終段，自動搬送）は，ログBの滞在時間を手空きサンプルの代わりに使う."""
    for material_id, path in topology.visits.items():
        pid = pid_of.get(material_id)
        if pid is None:
            continue
        for prev, cur in zip(path, path[1:]):
            stage = topology.stage_of.get(cur.pos)
            if stage is None or stage.outbound is not None or cur.t_out is None or prev.t_out is None:
                continue
            dwell = cur.t_out - prev.t_out
            for target in (by_stage[(stage.index, pid)], by_pos[(cur.pos, pid)]):
                target.idle.append(dwell)
                target.idle_times.append(prev.t_out)
                target.upper.append(dwell)
                target.lower.append(0.0)


def _arrival_times(topology: Topology, alignment: Alignment) -> dict[tuple[str, int], float]:
    """(マテリアル, 段) ごとの到着時刻（ログCの時計）.

    ロボットが置いた段は，置く動作の END．ロボットが置かない段（先頭段，自動搬送）は，ログBの IN にラグを足す．
    """
    arrivals: dict[tuple[str, int], float] = {}
    for actions in alignment.actions.values():
        for a in actions:
            if a.kind == PLACE:
                arrivals[(a.material_id, a.stage)] = a.end
    for material_id, path in topology.visits.items():
        for visit in path:
            stage = topology.stage_of.get(visit.pos)
            if stage is None or stage.feed is not None or stage.outbound is None or visit.t_in is None:
                continue
            lag = alignment.lags.get(stage.outbound.robot_id)
            if lag is not None:
                arrivals[(material_id, stage.index)] = visit.t_in + lag.center_s
    return arrivals


def estimate_service(
    alignment: Alignment, motion: Motion, topology: Topology, pid_of: dict[str, str],
) -> ServiceTimes:
    by_stage: dict[tuple[int, str], Samples] = defaultdict(Samples)
    by_pos: dict[tuple[PosKey, str], Samples] = defaultdict(Samples)
    arrivals = _arrival_times(topology, alignment)
    for robot_id, actions in alignment.actions.items():
        _collect_for_robot(robot_id, actions, alignment, motion, topology, arrivals, pid_of, by_stage, by_pos)
    _final_dwell(topology, pid_of, by_stage, by_pos)

    default_width = max((alignment.width(r) for r in alignment.lags), default=1.0)
    last_index = len(topology.stages) - 1

    def width_of(stage: Stage) -> float:
        hand = stage.outbound
        return alignment.width(hand.robot_id) if hand is not None and hand.robot_id in alignment.lags else default_width

    result = ServiceTimes(
        {key: _summarize(samples, width_of(topology.stages[key[0]]), topology.stages[key[0]], last_index)
         for key, samples in by_stage.items()},
        {key: _summarize(samples, width_of(topology.stage_of[key[0]]), topology.stage_of[key[0]], last_index)
         for key, samples in by_pos.items()},
        dict(by_stage),
    )

    for (pos, pid), own in result.by_pos.items():
        stage = topology.stage_of[pos]
        shared = result.by_stage[(stage.index, pid)]
        if own.value.value is None or shared.value.value is None or own.value.quality != ESTIMATED:
            continue
        if abs(own.value.value - shared.value.value) > width_of(stage):
            result.warnings.append(
                f"{pos[0]}-{pos[1]} の処理ID {pid} の処理時間 {own.value.value:.1f} s が，"
                f"段の値 {shared.value.value:.1f} s と離れています"
            )
    return result
