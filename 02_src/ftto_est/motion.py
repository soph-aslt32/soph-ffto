"""動作の長さ・移動時間・ロボットの飽和サイクル（algorithm.md 第2章，第3.1章）."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from .actions import PICK, PLACE, Action, Alignment
from .logs import PosKey
from .quality import BORROWED, ESTIMATED, UNOBSERVED, UPPER_ONLY, Value
from .stats import MIN_COUNT, center, low_center, lowest_common
from .topology import Topology

MOVE = "MOVE"
MIN_LAYER_SAMPLES = 30  # 層ごとの動作の長さを，その層の値として使う最小の件数


@dataclass(frozen=True, slots=True)
class Step:
    """飽和サイクルの1手順．stages は，取る・置くなら1つ，移動なら2つ."""

    action: str  # PICK | PLACE | MOVE
    stages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Cycle:
    """ロボットが1枚のマテリアルに対して行う手順."""

    robot_id: str
    upstream: int
    process: int | None  # 入れ替えを行う段．単純な搬送は None
    downstream: int
    steps: tuple[Step, ...]


class CycleRule(Protocol):
    """飽和サイクルの組み立て方．装置構成で手順が変わるので，差し替えられるようにする."""

    def cycles(self, topology: Topology) -> dict[str, list[Cycle]]:
        ...


class BasicExchangeCycle:
    """確認済みの基本手順．段の前後を同じロボットの別々のハンドが担うときに入れ替えが成り立つ.

    入れ替えの段に含まれない搬送は，取る → 移動 → 置く → 戻る の単純な往復とする．
    """

    def cycles(self, topology: Topology) -> dict[str, list[Cycle]]:
        stages = topology.stages
        cycles: dict[str, list[Cycle]] = {r: [] for r in topology.robots}
        covered: set[int] = set()  # 入れ替えに含まれる搬送（搬送先の段の index）
        for stage in stages[1:-1]:
            feed, outbound = stage.feed, stage.outbound
            if feed is None or outbound is None:
                continue
            if feed.robot_id == outbound.robot_id and feed.name != outbound.name:
                u, p, d = stage.index - 1, stage.index, stage.index + 1
                cycles[feed.robot_id].append(Cycle(feed.robot_id, u, p, d, (
                    Step(PICK, (u,)), Step(MOVE, (u, p)), Step(PICK, (p,)), Step(PLACE, (p,)),
                    Step(MOVE, (p, d)), Step(PLACE, (d,)), Step(MOVE, (d, u)),
                )))
                covered.update((p, d))
        for stage in stages[1:]:
            if stage.feed is None or stage.index in covered:
                continue
            u, d = stage.index - 1, stage.index
            cycles[stage.feed.robot_id].append(Cycle(stage.feed.robot_id, u, None, d, (
                Step(PICK, (u,)), Step(MOVE, (u, d)), Step(PLACE, (d,)), Step(MOVE, (d, u)),
            )))
        return cycles


@dataclass
class Moves:
    """ロボットの段間の移動時間."""

    robot_id: str
    table: dict[tuple[int, int], Value] = field(default_factory=dict)

    def time(self, src: int, dst: int) -> float:
        """src から dst への移動時間．値が無ければ 0（待ちが長めに出る）."""
        if src == dst:
            return 0.0
        value = self.table.get((src, dst))
        return value.value if value is not None and value.value is not None else 0.0


@dataclass
class Motion:
    durations: dict[tuple[str, PosKey], Value]  # (PICK|PLACE, Pos) -> 動作の長さ
    durations_by_stage: dict[tuple[str, int], Value]  # 使われない層へ借用する値
    moves: dict[str, Moves]
    cycles: dict[str, list[Cycle]]
    capacity: dict[str, Value]  # ロボット -> W_r
    warnings: list[str] = field(default_factory=list)

    def duration(self, kind: str, pos: PosKey, stage: int) -> Value:
        own = self.durations.get((kind, pos))
        if own is not None:
            return own
        shared = self.durations_by_stage.get((kind, stage))
        if shared is None:
            return Value.unobserved()
        return Value(shared.value, shared.lo, shared.hi, BORROWED)


def _measure_durations(alignment: Alignment) -> tuple[dict, dict]:
    durations: dict[tuple[str, PosKey], Value] = {}
    shared: dict[tuple[str, int], Value] = {}
    for robot_id, actions in alignment.actions.items():
        width = alignment.width(robot_id)
        own_pos: dict[tuple[str, PosKey], list[int]] = defaultdict(list)
        own_stage: dict[tuple[str, int], list[int]] = defaultdict(list)
        for a in actions:
            own_pos[(a.kind, a.pos)].append(a.duration)
            own_stage[(a.kind, a.stage)].append(a.duration)
        # 件数の少ない層の値は使わず，段の値を借用する
        durations.update({k: Value.of(center(v, width)) for k, v in own_pos.items() if len(v) >= MIN_LAYER_SAMPLES})
        shared.update({k: Value.of(center(v, width)) for k, v in own_stage.items()})
    return durations, shared


def _travel(groups: dict[tuple[PosKey, PosKey], list[int]], width: float) -> float:
    """持ったままの移動の時間．層ごとの中心を平均する（飽和時には全層が使われる）."""
    enough = [v for v in groups.values() if len(v) >= MIN_COUNT]
    if not enough:
        enough = [[x for v in groups.values() for x in v]]
    return statistics.fmean(low_center(v, width) for v in enough)


def _measure_moves(robot_id: str, actions: list[Action], width: float, spread: float) -> Moves:
    """END から次の START までの区間から移動時間を求める.

    置く動作の直前の区間には，閉塞が無ければ待ちが入らない．取る動作の直前の区間には待ちが入りうるので，
    上界しか分からない．
    """
    carried: dict[tuple[int, int], dict[tuple[PosKey, PosKey], list[int]]] = defaultdict(lambda: defaultdict(list))
    empty: dict[frozenset[int], list[int]] = defaultdict(list)
    for a, b in zip(actions, actions[1:]):
        if b.resync or a.stage == b.stage:
            continue
        gap = b.start - a.end
        if b.kind == PLACE:
            carried[(a.stage, b.stage)][(a.pos, b.pos)].append(gap)
        else:
            empty[frozenset((a.stage, b.stage))].append(gap)

    moves = Moves(robot_id)
    for (src, dst), groups in carried.items():
        value = Value.of(_travel(groups, width))
        moves.table[(src, dst)] = value
        moves.table.setdefault((dst, src), value)  # 往復の移動時間は同じとみなす

    # 残りは上界．区間の最小の値にばらつきの幅を足した値と，経由する経路の和の小さい方
    ordered = sorted({a.stage for a in actions})
    for i, src in enumerate(ordered):
        for dst in ordered[i + 1:]:
            if (src, dst) in moves.table:
                continue
            bounds = []
            direct = lowest_common(empty.get(frozenset((src, dst)), []))
            if direct is not None:
                bounds.append(direct + spread)
            for via in ordered:
                first, second = moves.table.get((src, via)), moves.table.get((via, dst))
                if via not in (src, dst) and first is not None and second is not None:
                    bounds.append(first.value + second.value)
            if bounds:
                value = Value(min(bounds), 0.0, min(bounds), UPPER_ONLY)
                moves.table[(src, dst)] = moves.table[(dst, src)] = value
    return moves


def _sum_cycle(cycle: Cycle, motion: Motion, topology: Topology) -> Value:
    """手順に沿って足す．上界の移動があれば上界，欠けた型があれば未観測とし，既知の型の和を下界にする."""
    known = 0.0
    bound = 0.0
    missing = False
    for step in cycle.steps:
        if step.action == MOVE:
            value = motion.moves[cycle.robot_id].table.get(step.stages)
            if value is None:
                missing = True
            elif value.quality == UPPER_ONLY:
                bound += value.value
            else:
                known += value.value
            continue
        stage = topology.stages[step.stages[0]]
        values = [motion.duration(step.action, pos, stage.index) for pos in stage.positions]
        values = [v.value for v in values if v.value is not None]
        if values:
            known += statistics.fmean(values)
        else:
            missing = True
    if missing:
        return Value(None, known, None, UNOBSERVED)
    if bound > 0:
        return Value(known + bound, known, known + bound, UPPER_ONLY)
    return Value.of(known)


def measure_motion(alignment: Alignment, topology: Topology, rule: CycleRule) -> Motion:
    durations, shared = _measure_durations(alignment)
    cycles = rule.cycles(topology)
    moves: dict[str, Moves] = {}
    for robot_id in topology.robots:
        lag = alignment.lags.get(robot_id)
        if lag is None:
            moves[robot_id] = Moves(robot_id)
            continue
        moves[robot_id] = _measure_moves(robot_id, alignment.actions[robot_id], alignment.width(robot_id), lag.spread_s)
    motion = Motion(durations, shared, moves, cycles, {})

    for robot_id, robot_cycles in cycles.items():
        if not robot_cycles:
            motion.capacity[robot_id] = Value.unobserved()
            continue
        values = [_sum_cycle(c, motion, topology) for c in robot_cycles]
        if any(v.value is None for v in values):
            motion.warnings.append(f"{robot_id}: 手順の一部の動作が観測されず，W を求められません")
        if len(values) == 1:
            motion.capacity[robot_id] = values[0]
            continue
        lo = sum(v.lo or 0.0 for v in values)
        if any(v.value is None for v in values):
            motion.capacity[robot_id] = Value(None, lo, None, UNOBSERVED)
        else:
            total = sum(v.value for v in values)
            quality = UPPER_ONLY if any(v.quality == UPPER_ONLY for v in values) else ESTIMATED
            motion.capacity[robot_id] = Value(total, lo, total, quality)
    return motion
