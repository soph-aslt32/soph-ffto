"""律速の判定とライン限界（algorithm.md 第7.2章，第8章）."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .actions import Alignment
from .capacity import Capacity
from .margin import KIND_ROBOT, KIND_STAGE, Cassette, PartMargin, mix_capacity
from .motion import Motion
from .quality import INTERVAL_ONLY, LOWER_ONLY, Value, worse
from .topology import Topology
from .waits import CycleWaits

MAJORITY = 0.5  # カセットの枚数のうち，保持待ちが出ている割合
STATUS_NONE = "律速なし"
STATUS_STEADY = "律速"
STATUS_TRANSIENT = "過渡"
STATUS_MISMATCH = "不一致"

ROBOT_LIMIT_MARGIN_S = 0.5  # 空いている時間がこれ未満なら，ロボットが能力いっぱいで動いている
TIE_RATIO = 0.95  # 能力の上位2つの比がこれ以上なら拮抗とみなす
TIE_SLOTS = 2  # 間の保持枠がこれ以下なら，互いに干渉する


@dataclass
class Limit:
    """ライン限界の下限：この間隔より短くすると，少なくともこの部位が能力を超える."""

    value: Value
    candidates: list[str]  # 律速部位の候補．確定すれば1つ
    runner_up: str | None = None
    runner_up_gap_s: float | None = None
    tie: bool = False  # 拮抗の警告

    @property
    def part(self) -> str:
        return self.candidates[0] if len(self.candidates) == 1 else "未確定"


@dataclass
class CassetteJudgement:
    cassette: Cassette
    interval_s: float | None  # 律速と予測した部位の入場間隔
    limit: Limit
    observed: str | None  # 保持待ちから判定した律速部位
    status: str
    hold_share: dict[str, float] = field(default_factory=dict)  # 部位 -> 保持待ちが出た枚数の割合


@dataclass
class Calibration:
    """κ ＝ 律速が観測されたカセットの実測の間隔 ÷ 予測."""

    process_id: str
    part: str
    kappa: float
    n_cassettes: int


def _parts(topology: Topology, capacity: Capacity, pid: str) -> dict[str, Value]:
    parts = {stage.name: capacity.by_stage[(stage.index, pid)] for stage in topology.stages}
    parts.update(capacity.by_robot)
    return parts


def _location(topology: Topology, motion: Motion, part: str) -> int:
    """部位の経路上の位置（段の index）．ロボットは担当する処理層の位置."""
    for stage in topology.stages:
        if stage.name == part:
            return stage.index
    cycles = motion.cycles.get(part, [])
    return max((c.process if c.process is not None else c.upstream) for c in cycles) if cycles else 0


def _slots_between(topology: Topology, a: int, b: int) -> int:
    """段 a と段 b の間の保持枠（コンベア＋ハンド）の数."""
    lo, hi = min(a, b), max(a, b)
    hands = sum(1 for s in topology.stages[lo + 1:hi + 1] if s.feed is not None)
    conveyors = sum(len(s.positions) for s in topology.stages[lo + 1:hi] if s.kind != "process")
    return hands + conveyors


def line_limit(parts: dict[str, Value], topology: Topology, motion: Motion) -> Limit:
    """全部位の max．値の無い部位を除くと限界を過小評価するので，区間のまま max を取る."""
    known = {k: v for k, v in parts.items() if v.value is not None}
    if not known:
        return Limit(Value.unobserved(), [])
    lo = max(v.lo if v.lo is not None else 0.0 for v in parts.values())
    hi = max(v.hi if v.hi is not None else v.value for v in known.values())
    value = max(v.value for v in known.values())
    ranked = sorted(known, key=lambda k: known[k].value, reverse=True)
    candidates = [k for k in ranked if (known[k].hi if known[k].hi is not None else known[k].value) >= lo - 1e-9]
    candidates = candidates or ranked[:1]
    quality = INTERVAL_ONLY if len(candidates) > 1 else known[candidates[0]].quality
    if len(candidates) > 1:
        value = hi  # 律速部位が確定しないときは，保守側の上界を出す
    if len(known) < len(parts):
        quality = LOWER_ONLY  # 未観測の部位の方が遅いかもしれない
    runner_up = ranked[1] if len(ranked) > 1 else None
    gap = known[ranked[0]].value - known[runner_up].value if runner_up else None
    tie = False
    if runner_up and known[runner_up].value / known[ranked[0]].value >= TIE_RATIO:
        slots = _slots_between(topology, _location(topology, motion, ranked[0]),
                               _location(topology, motion, runner_up))
        tie = slots <= TIE_SLOTS
    if tie:
        quality = worse(quality, LOWER_ONLY)
    return Limit(Value(value, lo, hi, quality), candidates, runner_up, gap, tie)


def _robot_spare_time(interval: float | None, capacity: Value) -> float | None:
    """入場間隔 − W．保持している時間を引くと，ロボットが本当に空いている時間になる."""
    if interval is None or capacity.value is None:
        return None
    return interval - capacity.value


def observe_bottleneck(
    cycles_by_robot: dict[str, list[CycleWaits]],
    robot_spare: dict[str, float | None],
    topology: Topology,
    motion: Motion,
    alignment: Alignment,
) -> tuple[str | None, dict[str, float]]:
    """保持待ちが過半の枚数で出ている待ち先のうち，最も下流のもの.

    律速の手前は順に詰まり，上流のロボットも下流の閉塞で保持待ちになるため．
    ロボット自身が律速のときは保持待ちが続かないので，待ちがほぼ無く，空いている時間も無いことから判定する．
    """
    shares: dict[str, float] = {}
    candidates: list[tuple[int, str]] = []
    for robot_id, cycles in cycles_by_robot.items():
        threshold = alignment.threshold(robot_id)
        counts: dict[int, int] = defaultdict(int)
        for c in cycles:
            for stage, wait in c.hold_s.items():
                if wait >= threshold:
                    counts[stage] += 1
        held = False
        for index, n in counts.items():
            share = n / len(cycles)
            name = topology.stages[index].name
            shares[name] = max(shares.get(name, 0.0), share)
            if share >= MAJORITY:
                candidates.append((index, name))
                held = True
        # 保持待ちも含めて待ちが無ければ，ロボットは動き続けている
        no_wait = statistics.median(c.margin_s + sum(c.hold_s.values()) for c in cycles) < threshold
        spare = robot_spare[robot_id]
        if spare is not None:
            spare -= statistics.fmean(c.held_s for c in cycles)
        if not held and no_wait and (spare is None or spare < ROBOT_LIMIT_MARGIN_S):
            candidates.append((_location(topology, motion, robot_id), robot_id))
    if not candidates:
        return None, shares
    return max(candidates)[1], shares


def _cycles_by_cassette(
    cassettes: list[Cassette], waits: dict[str, list[CycleWaits]],
) -> dict[str, dict[str, list[CycleWaits]]]:
    cassette_of = {m: c.cassette_id for c in cassettes for m in c.materials}
    grouped: dict[str, dict[str, list[CycleWaits]]] = defaultdict(lambda: defaultdict(list))
    for robot_id, rows in waits.items():
        for cycle in rows:
            cassette_id = cassette_of.get(cycle.material_id)
            if cassette_id is not None:
                grouped[cassette_id][robot_id].append(cycle)
    return grouped


def judge_cassettes(
    cassettes: list[Cassette],
    topology: Topology,
    motion: Motion,
    capacity: Capacity,
    waits: dict[str, list[CycleWaits]],
    alignment: Alignment,
    margins: list[PartMargin],
) -> list[CassetteJudgement]:
    intervals = {(m.cassette_id, m.part): m.interval_s for m in margins if m.kind in (KIND_STAGE, KIND_ROBOT)}
    cycles = _cycles_by_cassette(cassettes, waits)
    result: list[CassetteJudgement] = []
    previous: CassetteJudgement | None = None
    for cassette in cassettes:
        by_part: dict[str, dict[str, Value]] = defaultdict(dict)
        for pid in cassette.process_mix:
            for part, value in _parts(topology, capacity, pid).items():
                by_part[part][pid] = value
        parts = {part: mix_capacity(values, cassette.process_mix) for part, values in by_part.items()}
        limit = line_limit(parts, topology, motion)
        spare = {
            r: _robot_spare_time(intervals.get((cassette.cassette_id, r)), capacity.by_robot[r])
            for r in topology.robots
        }
        observed, shares = observe_bottleneck(cycles[cassette.cassette_id], spare, topology, motion, alignment)
        interval = intervals.get((cassette.cassette_id, limit.candidates[0])) if limit.candidates else None
        carried = previous is not None and previous.observed is not None
        if observed is None:
            status = STATUS_NONE
        elif carried and _drains_previous(previous, limit):
            status = STATUS_TRANSIENT
        elif observed in limit.candidates:
            status = STATUS_STEADY
        elif carried:
            status = STATUS_TRANSIENT
        else:
            status = STATUS_MISMATCH
        previous = CassetteJudgement(cassette, interval, limit, observed, status, shares)
        result.append(previous)
    return result


def _drains_previous(previous: CassetteJudgement, limit: Limit) -> bool:
    """前のカセットの方が遅ければ，このカセットは前の詰まりを解消している途中である.

    このとき部位はこのカセットの能力いっぱいで動くので，律速部位も間隔も予測に合うが，定常の律速ではない．
    """
    before, now = previous.limit.value.value, limit.value.value
    return before is not None and now is not None and before > now * 1.01


def limit_by_process(topology: Topology, motion: Motion, capacity: Capacity) -> dict[str, Limit]:
    """処理ID ごとのライン限界の下限."""
    return {pid: line_limit(_parts(topology, capacity, pid), topology, motion) for pid in capacity.process_ids}


def calibrate(judgements: list[CassetteJudgement]) -> list[Calibration]:
    """κ は同じ (処理ID, 律速部位) にだけ適用する．過渡のカセットは使わない."""
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for j in judgements:
        if j.status != STATUS_STEADY or j.interval_s is None or not j.limit.value.value:
            continue
        ratios[(j.cassette.process_id, j.observed)].append(j.interval_s / j.limit.value.value)
    return [Calibration(pid, part, statistics.median(v), len(v)) for (pid, part), v in sorted(ratios.items())]


def month_limit(judgements: list[CassetteJudgement]) -> dict[str, float | None]:
    """月全体：Σ n_K max_i C_i(K) / Σ n_K．max を取ってから平均する."""
    rows = [(j.cassette.n, j.limit.value) for j in judgements if j.limit.value.value is not None]
    total = sum(n for n, _ in rows)
    if not total:
        return {"T_line_min_s": None, "lo_s": None, "hi_s": None}

    def weighted(pick) -> float | None:
        items = [(n, pick(v)) for n, v in rows]
        return None if any(x is None for _, x in items) else sum(n * x for n, x in items) / total

    return {
        "T_line_min_s": weighted(lambda v: v.value),
        "lo_s": weighted(lambda v: v.lo),
        "hi_s": weighted(lambda v: v.hi),
    }
