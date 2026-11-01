"""段での処理時間 s（algorithm.md 第5章，第6.3章）.

処理の完了はログに記録されない．ただしロボットは処理済みのマテリアルがあれば待たないので，
待っていたロボットが動き出した時刻が，処理の完了にあたる．サンプルは，ロボットが待っていた場所で2つに分かれる．

- 層の前で待っていた（次のマテリアルを持っている）：取り始めた時刻が処理の完了そのもの
- 原点などで待っていた（何も持っていない）：受け入れ可能の信号を受けて層へ向かう．
  信号が移動時間より遅いと，着くのが処理の完了より遅れ，取り始めた時刻に 遅れ ＝ max(0, 原点からの移動 − 信号の時間) が乗る
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .actions import PICK, PLACE, Action, Alignment
from .logs import PosKey
from .motion import Motion
from .quality import BIASED, ESTIMATED, INTERVAL_ONLY, UPPER_ONLY, Value
from .settings import Settings
from .stats import center, low_center
from .topology import Stage, Topology, pos_label

MIN_ADOPTION = 0.5  # 手空きサンプルの採用率がこれ未満なら，偏りの恐れがある
MIN_FRONT_SAMPLES = 30  # サンプルを信頼する最小の件数
DELAY_AGREEMENT_S = 0.5  # 処理IDごとの遅れの差がこの範囲なら，同じ遅れとみなす


@dataclass
class Samples:
    idle: list[float] = field(default_factory=list)  # 手空きサンプル（何も持たずに待っていた）
    idle_times: list[float] = field(default_factory=list)  # 手空きサンプルの時刻（置かれた時刻）
    front: list[float] = field(default_factory=list)  # 層の前で待っていたサンプル
    upper: list[float] = field(default_factory=list)  # 取り始めた時点で処理は終わっていた
    lower: list[float] = field(default_factory=list)  # 手空きの間は処理が終わっていなかった


@dataclass(frozen=True)
class Delay:
    """手空きサンプルに乗る遅れの範囲."""

    lo: float
    hi: float


NO_DELAY = Delay(0.0, 0.0)


@dataclass
class ServiceTime:
    """段 × 処理ID の処理時間．層ごとの値は by_pos に持つ."""

    value: Value
    n_idle: int  # 手空きサンプルと，層の前で待っていたサンプルの合計
    n_total: int
    n_front: int = 0

    @property
    def adoption(self) -> float:
        return self.n_idle / self.n_total if self.n_total else 0.0


@dataclass
class ServiceTimes:
    by_stage: dict[tuple[int, str], ServiceTime]
    by_pos: dict[tuple[PosKey, str], ServiceTime]
    samples: dict[tuple[int, str], Samples]  # 能力の時系列の監視に使う
    delays: dict[int, Delay]  # 段 -> 手空きサンプルに乗る遅れ
    warnings: list[str] = field(default_factory=list)


def _summarize(samples: Samples, width: float, stage: Stage, last_index: int, delay: Delay) -> ServiceTime:
    n_front = len(samples.front)
    n_idle, n_total = len(samples.idle) + n_front, len(samples.upper)
    if n_total == 0:
        return ServiceTime(Value.unobserved(), 0, 0)
    upper = statistics.median(samples.upper)
    lower = statistics.median(samples.lower)
    if n_idle == 0 or (n_idle < MIN_FRONT_SAMPLES and n_idle / n_total < MIN_ADOPTION):
        return ServiceTime(Value(upper, min(lower, upper), upper, INTERVAL_ONLY), n_idle, n_total, n_front)

    auto = stage.outbound is None and stage.index != last_index
    if n_front >= MIN_FRONT_SAMPLES or not samples.idle:
        high = low = center(samples.front, width)
    else:
        measured = low_center(samples.idle, width) if auto else center(samples.idle, width)
        high, low = measured - delay.lo, measured - delay.hi

    # 次のロボットが離れた位置で待っていると移動時間が混ざる．自動搬送は次の段の詰まりが混ざる
    if auto or (stage.kind != "process" and stage.index != last_index):
        return ServiceTime(Value(high, min(lower, high), high, UPPER_ONLY), n_idle, n_total, n_front)
    if n_idle / n_total < MIN_ADOPTION:
        return ServiceTime(Value(high, min(lower, low), max(upper, high), BIASED), n_idle, n_total, n_front)
    if high > low:
        return ServiceTime(Value(high, low, high, UPPER_ONLY), n_idle, n_total, n_front)
    return ServiceTime(Value.of(high), n_idle, n_total, n_front)


def _collect_for_robot(
    robot_id: str,
    actions: list[Action],
    alignment: Alignment,
    motion: Motion,
    topology: Topology,
    reach: dict[int, float],
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
        held = held_after[i - 1]
        waited = (
            not b.resync and held <= waiting_for and b.start - a.end - moves.time(a.stage, b.stage) >= threshold
        )
        at_front = is_process and bool(held)

        # 最後に手空きだった時刻．置かれた時刻までさかのぼる
        last_idle = t0
        for j in range(i, 0, -1):
            x, y = actions[j - 1], actions[j]
            if not y.resync and held_after[j - 1] <= waiting_for:
                step = moves.time(x.stage, y.stage)
                if y.start - x.end - step >= threshold:
                    # 何も持たずに待っていたなら，取り始める前に動き出している．原点へ戻る途中だった場合は往復分になる
                    away = y is b and not at_front
                    last_idle = max(last_idle, y.start - (max(step, 2 * reach.get(b.stage, 0.0)) if away else step))
                    break
            if y.resync or x.end < t0:
                break

        for target in (by_stage[(b.stage, pid)], by_pos[(b.pos, pid)]):
            target.upper.append(b.start - t0)
            target.lower.append(last_idle - t0)
            if waited and at_front:
                target.front.append(b.start - t0)
            elif waited:
                target.idle.append(b.start - t0)
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


def _reach(topology: Topology, motion: Motion, settings: Settings) -> dict[int, float]:
    """段ごとの，原点から段までの移動時間の上限．未設定なら，そのロボットの最も長い移動とする."""
    reach: dict[int, float] = {}
    for stage in topology.stages:
        if stage.outbound is None:
            continue
        set_values = [settings.home(pos_label(p)) for p in stage.positions]
        table = motion.moves[stage.outbound.robot_id].table.values() if stage.outbound.robot_id in motion.moves else []
        longest = max((v.value for v in table if v.value is not None), default=0.0)
        reach[stage.index] = max(longest if v is None else v for v in set_values)
    return reach


def _configured_delay(stage: Stage, settings: Settings, reach: dict[int, float]) -> Delay:
    """設定値から求めた遅れ．原点からの移動が未設定なら，0〜上限の範囲になる."""
    leads = [settings.lead(pos_label(p)) for p in stage.positions]
    homes = [settings.home(pos_label(p)) for p in stage.positions]
    low = statistics.fmean(max((h or 0.0) - lead, 0.0) for h, lead in zip(homes, leads))
    high = statistics.fmean(
        max((reach.get(stage.index, 0.0) if h is None else h) - lead, 0.0) for h, lead in zip(homes, leads)
    )
    return Delay(low, high)


def _measured_delay(stage: Stage, by_stage: dict[tuple[int, str], Samples], width: float) -> Delay | None:
    """同じ処理IDの，層の前で待っていたサンプルとの差から遅れを求める．処理IDの間で食い違えば，範囲で返す."""
    differences = []
    for (index, _), samples in by_stage.items():
        if index == stage.index and min(len(samples.front), len(samples.idle)) >= MIN_FRONT_SAMPLES:
            differences.append(max(center(samples.idle, width) - center(samples.front, width), 0.0))
    if not differences:
        return None
    if max(differences) - min(differences) <= DELAY_AGREEMENT_S:
        value = statistics.median(differences)
        return Delay(value, value)
    return Delay(min(differences), max(differences))


def _stage_delay(stage: Stage, settings: Settings, reach: dict[int, float], measured: Delay | None) -> Delay:
    """手空きサンプルに乗る遅れ．設定値が揃っていればそれを使い，足りなければログから測った値を使う."""
    configured = _configured_delay(stage, settings, reach)
    if configured.lo == configured.hi or measured is None:
        return configured
    return measured


def estimate_service(
    alignment: Alignment, motion: Motion, topology: Topology, pid_of: dict[str, str], settings: Settings,
) -> ServiceTimes:
    by_stage: dict[tuple[int, str], Samples] = defaultdict(Samples)
    by_pos: dict[tuple[PosKey, str], Samples] = defaultdict(Samples)
    arrivals = _arrival_times(topology, alignment)
    reach = _reach(topology, motion, settings)
    for robot_id, actions in alignment.actions.items():
        _collect_for_robot(robot_id, actions, alignment, motion, topology, reach, arrivals, pid_of, by_stage, by_pos)
    _final_dwell(topology, pid_of, by_stage, by_pos)

    default_width = max((alignment.width(r) for r in alignment.lags), default=1.0)
    last_index = len(topology.stages) - 1

    def width_of(stage: Stage) -> float:
        hand = stage.outbound
        return alignment.width(hand.robot_id) if hand is not None and hand.robot_id in alignment.lags else default_width

    delays: dict[int, Delay] = {}
    for stage in topology.stages:
        if stage.kind != "process" or stage.outbound is None:
            delays[stage.index] = NO_DELAY
        else:
            delays[stage.index] = _stage_delay(
                stage, settings, reach, _measured_delay(stage, by_stage, width_of(stage)),
            )

    def summarize(samples: Samples, stage: Stage) -> ServiceTime:
        return _summarize(samples, width_of(stage), stage, last_index, delays[stage.index])

    result = ServiceTimes(
        {key: summarize(samples, topology.stages[key[0]]) for key, samples in by_stage.items()},
        {key: summarize(samples, topology.stage_of[key[0]]) for key, samples in by_pos.items()},
        dict(by_stage),
        delays,
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
