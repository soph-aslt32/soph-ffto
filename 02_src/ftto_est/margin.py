"""カセットごとの入場間隔・余裕・稼働率（algorithm.md 第3.3章，第7.1章）."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from .actions import PICK, PLACE, Action
from .capacity import Capacity
from .logs import Logs, PosKey
from .motion import MOVE, Motion, Step
from .quality import ESTIMATED, Value, worse
from .stats import mean_interval
from .topology import Stage, Topology, pos_label
from .waits import CycleWaits

KIND_POS = "pos"
KIND_STAGE = "stage"
KIND_ROBOT = "robot"
KIND_HAND = "hand"


@dataclass
class Cassette:
    cassette_id: str
    materials: list[str]
    process_mix: Counter[str]
    t_first: int

    @property
    def n(self) -> int:
        return len(self.materials)

    @property
    def process_id(self) -> str:
        """カセットの処理ID．混在するときは出現の多い順に並べる."""
        return "+".join(p for p, _ in self.process_mix.most_common())


@dataclass
class PartMargin:
    """カセット K における部位 i の余裕."""

    cassette_id: str
    kind: str
    part: str
    capacity: Value  # C_i(K)
    interval_s: float | None  # T_{i,K}
    margin_s: float | None  # Δ_{i,K}
    rho: float | None
    wait_idle_s: float | None = None  # ロボットとハンドのみ
    wait_process_s: float | None = None
    wait_hold_s: float | None = None
    hold_target: str | None = None  # 最も長く待った待ち先
    hold_mean_s: float | None = None  # ハンドのみ．実保持時間
    hold_min_s: float | None = None  # ハンドのみ．最短保持時間

    @property
    def hold_excess_s(self) -> float | None:
        """保持中の待ち：実保持時間 − 最短保持時間."""
        if self.hold_mean_s is None or self.hold_min_s is None:
            return None
        return max(self.hold_mean_s - self.hold_min_s, 0.0)


def build_cassettes(logs: Logs) -> list[Cassette]:
    grouped: dict[str, list] = defaultdict(list)
    for row in sorted(logs.intakes, key=lambda r: r.time):
        grouped[row.cassette_id].append(row)
    return [
        Cassette(cid, [r.material_id for r in rows], Counter(r.process_id for r in rows), rows[0].time)
        for cid, rows in sorted(grouped.items(), key=lambda kv: kv[1][0].time)
    ]


def mix_capacity(values: dict[str, Value], mix: Counter[str]) -> Value:
    """カセットの処理ID の構成に応じた能力：材料の C の平均."""
    total = sum(mix.values())
    items = [(values[pid], n / total) for pid, n in mix.items() if pid in values]
    if len(items) == 1:
        return items[0][0]
    if not items or any(v.value is None for v, _ in items):
        return Value.unobserved()

    def combine(picker) -> float | None:
        parts = [(picker(v), w) for v, w in items]
        return None if any(x is None for x, _ in parts) else sum(x * w for x, w in parts)

    quality = ESTIMATED
    for v, _ in items:
        quality = worse(quality, v.quality)
    return Value(combine(lambda v: v.value), combine(lambda v: v.lo), combine(lambda v: v.hi), quality)


def _part_margin(cid: str, kind: str, part: str, capacity: Value, interval: float | None) -> PartMargin:
    """Δ = T − C．区間の部位は，保守側の上界で余裕を出す."""
    if interval is None or capacity.value is None:
        return PartMargin(cid, kind, part, capacity, interval, None, None)
    margin = interval - capacity.value
    return PartMargin(cid, kind, part, capacity, interval, margin, 1 - margin / interval)


def _entry_times(topology: Topology) -> tuple[dict[PosKey, dict[str, int]], dict[int, dict[str, int]]]:
    by_pos: dict[PosKey, dict[str, int]] = defaultdict(dict)
    by_stage: dict[int, dict[str, int]] = defaultdict(dict)
    for material_id, path in topology.visits.items():
        for visit in path:
            stage = topology.stage_of.get(visit.pos)
            if stage is None or visit.t_in is None:
                continue
            by_pos[visit.pos][material_id] = visit.t_in
            by_stage[stage.index][material_id] = visit.t_in
    return by_pos, by_stage


def _hold_times(actions: dict[str, list[Action]]) -> dict[tuple[str, str, str], int]:
    """(ロボット, ハンド名, マテリアル) ごとの実保持時間．ログBの IN にはラグが乗るので，ログCの中で時間差を取る."""
    started: dict[tuple[str, str, str], int] = {}
    held: dict[tuple[str, str, str], int] = {}
    for robot_id, rows in actions.items():
        for a in rows:
            key = (robot_id, a.hand.name, a.material_id)
            if a.kind == PICK:
                started[key] = a.end
            elif key in started:
                held[key] = a.end - started.pop(key)
    return held


def _step_hand(step: Step, stage: Stage) -> str | None:
    hand = stage.outbound if step.action == PICK else stage.feed
    return hand.name if hand is not None else None


def _min_hold(motion: Motion, robot_id: str, hand_name: str, topology: Topology) -> float | None:
    """最短保持時間：手順の中で，そのハンドが取り終えてから置き終えるまでの動作の和."""
    total = 0.0
    found = False
    for cycle in motion.cycles.get(robot_id, []):
        holding = False
        for step in cycle.steps:
            if step.action == MOVE:
                if holding:
                    total += motion.moves[robot_id].time(*step.stages)
                continue
            stage = topology.stages[step.stages[0]]
            own = _step_hand(step, stage) == hand_name
            if step.action == PICK and own:
                holding = found = True
                continue
            if holding:
                values = [motion.duration(step.action, p, stage.index).value for p in stage.positions]
                values = [v for v in values if v is not None]
                total += statistics.fmean(values) if values else 0.0
            if step.action == PLACE and own:
                holding = False
    return total if found else None


def build_margins(
    cassettes: list[Cassette],
    topology: Topology,
    motion: Motion,
    capacity: Capacity,
    waits: dict[str, list[CycleWaits]],
    actions: dict[str, list[Action]],
) -> list[PartMargin]:
    pos_in, stage_in = _entry_times(topology)
    cycles_by_material: dict[tuple[str, str], CycleWaits] = {
        (c.robot_id, c.material_id): c for rows in waits.values() for c in rows
    }
    holds = _hold_times(actions)
    min_holds = {
        (r, name): _min_hold(motion, r, name, topology)
        for r in topology.robots for name in {h.name for h in topology.hands.values() if h.robot_id == r}
    }

    burst = max(len(stage.positions) for stage in topology.stages)
    rows: list[PartMargin] = []
    for cassette in cassettes:
        cid = cassette.cassette_id
        members = cassette.materials
        for stage in topology.stages:
            for pos in stage.positions:
                values = {pid: capacity.by_pos[(pos, pid)] for pid in cassette.process_mix}
                times = [pos_in[pos][m] for m in members if m in pos_in[pos]]
                rows.append(_part_margin(cid, KIND_POS, pos_label(pos), mix_capacity(values, cassette.process_mix),
                                         mean_interval(times, burst)))
            values = {pid: capacity.by_stage[(stage.index, pid)] for pid in cassette.process_mix}
            times = [stage_in[stage.index][m] for m in members if m in stage_in[stage.index]]
            rows.append(_part_margin(cid, KIND_STAGE, stage.name, mix_capacity(values, cassette.process_mix),
                                     mean_interval(times, burst)))

        for robot_id in topology.robots:
            cycles = [cycles_by_material[(robot_id, m)] for m in members if (robot_id, m) in cycles_by_material]
            robot_row = _robot_margin(cid, robot_id, capacity.by_robot[robot_id], cycles, burst)
            robot_row.hold_target = _main_target(cycles, topology)
            rows.append(robot_row)
            for hand in (h for h in topology.hands.values() if h.robot_id == robot_id):
                held = [holds[(robot_id, hand.name, m)] for m in members if (robot_id, hand.name, m) in holds]
                rows.append(PartMargin(
                    cid, KIND_HAND, pos_label(hand.pos), robot_row.capacity, robot_row.interval_s,
                    robot_row.margin_s, robot_row.rho,
                    robot_row.wait_idle_s, robot_row.wait_process_s, robot_row.wait_hold_s, robot_row.hold_target,
                    statistics.fmean(held) if held else None, min_holds[(robot_id, hand.name)],
                ))
    return rows


def _main_target(cycles: list[CycleWaits], topology: Topology) -> str | None:
    totals: Counter[int] = Counter()
    for c in cycles:
        totals.update(c.hold_s)
    if not totals:
        return None
    return topology.stages[totals.most_common(1)[0][0]].name


def _robot_margin(cid: str, robot_id: str, capacity: Value, cycles: list[CycleWaits], burst: int) -> PartMargin:
    """Δ_r = 手空き＋処理待ち（1枚あたりの中央値）．ρ_r = 1 − Δ_r / 入場間隔."""
    if not cycles:
        return PartMargin(cid, KIND_ROBOT, robot_id, capacity, None, None, None)
    interval = mean_interval([c.t_entry for c in cycles], burst)
    margin = statistics.median(c.margin_s for c in cycles)
    return PartMargin(
        cid, KIND_ROBOT, robot_id, capacity, interval, margin,
        None if interval is None else 1 - margin / interval,
        statistics.median(c.idle_s for c in cycles),
        statistics.median(c.process_s for c in cycles),
        statistics.median(sum(c.hold_s.values()) for c in cycles),
    )
