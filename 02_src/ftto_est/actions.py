"""ログCの動作を，ログBのイベントへ対応付ける（algorithm.md 第1章）.

Upper と Lower は役割が分離し，各ハンドは取る・置くを交互に繰り返す．MOVE の行も無いので，
ハンドごとに k 番目の START，k 番目の END，ログBの k 番目のイベントを対応させる．
動作名は使わない．同じ秒の中で行の順序が入れ替わっても，行ごとのラグが大きくても影響を受けない．
行が欠けて件数が合わないときや，途中で対応がずれているときだけ，時刻を見て区切り直す．

ログCの行のうち，ハンド以外の Pos は対象にしない．他の Pos の動作開始・終了が記録されるようになったら，
ここへ対応付けを足す．
"""

from __future__ import annotations

import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .logs import END, START, Logs, PosKey
from .stats import lowest_common, quantile
from .topology import Hand, Topology, pos_label

PICK = "PICK"
PLACE = "PLACE"


@dataclass(slots=True)
class Event:
    """ログBから読み取った，ハンドの動作."""

    time: int
    kind: str  # PICK | PLACE
    material_id: str
    pos: PosKey  # 取る，または置く対象の Pos


@dataclass(slots=True)
class Action:
    """ログBのイベントとログCの START/END を対応させた動作．時間の計算は start/end で行う."""

    hand: Hand
    kind: str
    material_id: str
    pos: PosKey
    stage: int
    event_time: int  # ログBの時刻．ロボット単位の順序を決める
    start: int
    end: int
    resync: bool = False  # 直前の動作との間に欠けがある（間の区間を使わない）

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class Lag:
    """ログCの END − ログBのイベント時刻 の，ロボットごとの中心とばらつきの幅.

    ログBの打刻が取る動作と置く動作で違っても幅が広がらないように，動作の種類ごとに測る．
    中心は，置く動作の値とする．
    """

    center_s: float
    spread_s: float


@dataclass
class Alignment:
    actions: dict[str, list[Action]]  # ロボット -> 動作の列
    lags: dict[str, Lag]
    n_events: int
    n_matched: int
    warnings: list[str] = field(default_factory=list)

    def width(self, robot_id: str) -> float:
        """g などの中心を取るときの許容幅."""
        return 1.0 + self.lags[robot_id].spread_s

    def threshold(self, robot_id: str) -> float:
        """手空きとみなす最小の待ち．打刻の丸めと行ごとのラグを超える長さ."""
        return 2.0 + self.lags[robot_id].spread_s


def hand_events(topology: Topology) -> dict[Hand, list[Event]]:
    """ハンドごとのイベント．取る動作は，ハンドに入る直前の Pos の OUT．置く動作は，ハンドの OUT."""
    events: dict[Hand, list[Event]] = defaultdict(list)
    for material_id, path in topology.visits.items():
        for i, visit in enumerate(path):
            hand = topology.hands.get(visit.pos)
            if hand is None:
                continue
            prev = path[i - 1] if i > 0 else None
            nxt = path[i + 1] if i + 1 < len(path) else None
            if prev is not None and prev.t_out is not None:
                events[hand].append(Event(prev.t_out, PICK, material_id, prev.pos))
            if nxt is not None and visit.t_out is not None:
                events[hand].append(Event(visit.t_out, PLACE, material_id, nxt.pos))
    for rows in events.values():
        rows.sort(key=lambda e: e.time)
    return events


SHIFT_RUN = 8  # 時刻差の外れがこの数だけ続いたら，対応がずれたとみなす．単発の外れは行ごとのラグや外乱でも起きる
OFFSET_RANGE_S = 30  # ログ間のずれを探す範囲
OFFSET_SAMPLES = 5000  # ずれを求めるのに使う行の数
PEAK_RATIO = 0.9  # 最も高い山のこの割合以上の山は，同じ高さとみなす


def _separation(times: list[int]) -> float:
    """同じハンドの隣り合うイベントの，典型的な最小の間隔．対応が1つずれると，時刻差がこの分だけ飛ぶ."""
    gaps = [b - a for a, b in zip(times, times[1:])]
    return lowest_common(gaps) or (min(gaps) if gaps else 1.0)


def _common_offset(a: list[int], b: list[int]) -> float:
    """b − a の全体のずれ．近くの時刻どうしの差の分布で，最も高い山の位置を取る.

    最も近い時刻との差だと，ずれがイベントの間隔の半分を超えたときに隣のイベントを指してしまう．
    正しい組の差はすべての行に現れるが，隣のイベントとの差は一部の行にしか現れない．
    ただし周期的なログでは，隣の周期との差も同じ高さの山になるので，同じ高さなら 0 に近い山を取る．

    行を間引くと，取る動作と置く動作が交互に並ぶ列の片方だけを拾うことがあるので，連続した区間を使う．
    """
    begin = max((len(b) - OFFSET_SAMPLES) // 2, 0)
    counts: Counter[int] = Counter()
    for t in b[begin:begin + OFFSET_SAMPLES]:
        counts.update(t - x for x in a[bisect_left(a, t - OFFSET_RANGE_S):bisect_right(a, t + OFFSET_RANGE_S)])
    if not counts:
        return 0.0
    height = {d: counts[d - 1] + counts[d] + counts[d + 1] for d in counts}
    highest = max(height.values())
    return min((d for d, h in height.items() if h >= PEAK_RATIO * highest), key=lambda d: (abs(d), d))


def _is_shifted(a: list[int], b: list[int], tolerance: float) -> bool:
    """順に対応させたときに，途中でずれていないか.

    ずれた区間の時刻差は，全体のずれから離れ続ける．
    """
    offset = _common_offset(a, b)
    run = 0
    for x, y in zip(a, b):
        run = run + 1 if abs(y - x - offset) > tolerance else 0
        if run >= SHIFT_RUN:
            return True
    return False


def _align_by_time(a: list[int], b: list[int], tolerance: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """2つの時刻の列を，全体のずれを除いて順に対応させる．(添字の対, a の余り, b の余り) を返す."""
    offset = _common_offset(a, b)
    pairs = []
    rest_a: list[int] = []
    rest_b: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        difference = b[j] - a[i] - offset
        if abs(difference) <= tolerance:
            pairs.append((i, j))
            i += 1
            j += 1
        elif difference > 0:
            rest_a.append(i)
            i += 1
        else:
            rest_b.append(j)
            j += 1
    rest_a.extend(range(i, len(a)))
    rest_b.extend(range(j, len(b)))
    return pairs, rest_a, rest_b


def _pair_rows(starts: list[int], ends: list[int], tolerance: float) -> tuple[list[tuple[int, int]], int]:
    """START と END を対にする．相手の無い行の数も返す."""
    if len(starts) == len(ends) and not _is_shifted(starts, ends, tolerance):
        return list(zip(starts, ends)), 0
    pairs, rest_starts, rest_ends = _align_by_time(starts, ends, tolerance)
    return [(starts[i], ends[j]) for i, j in pairs], len(rest_starts) + len(rest_ends)


def _match_events(
    events: list[Event], pairs: list[tuple[int, int]], tolerance: float,
) -> tuple[list[tuple[Event, tuple[int, int]]], list[float]]:
    """イベントと動作を対応させる．順に対応しないときは区切り直し，欠けた時刻（ログBの時計）を返す."""
    times = [e.time for e in events]
    finishes = [end for _, end in pairs]
    if len(times) == len(finishes) and not _is_shifted(times, finishes, tolerance):
        return list(zip(events, pairs)), []
    matched, rest_events, rest_pairs = _align_by_time(times, finishes, tolerance)
    offset = _common_offset(times, finishes)
    skipped = [times[i] for i in rest_events] + [finishes[j] - offset for j in rest_pairs]
    return [(events[i], pairs[j]) for i, j in matched], skipped


def _measure_lag(actions: list[Action]) -> Lag:
    spreads = []
    placed: list[int] = []
    for kind in (PICK, PLACE):
        diffs = sorted(a.end - a.event_time for a in actions if a.kind == kind)
        if diffs:
            spreads.append(quantile(diffs, 0.95) - quantile(diffs, 0.05))
        if kind == PLACE:
            placed = diffs
    # 打刻が秒単位なので，中央値だと 0 か 1 に丸まる．平均を取る
    return Lag(statistics.fmean(placed) if placed else 0.0, max(spreads))


def align_actions(logs: Logs, topology: Topology) -> Alignment:
    starts: dict[Hand, list[int]] = defaultdict(list)
    ends: dict[Hand, list[int]] = defaultdict(list)
    for row in logs.actions:
        hand = topology.hands.get(row.pos)
        if hand is None:
            continue
        if row.event == START:
            starts[hand].append(row.time)
        elif row.event == END:
            ends[hand].append(row.time)

    warnings: list[str] = []
    actions: dict[str, list[Action]] = defaultdict(list)
    breaks: dict[str, list[float]] = defaultdict(list)
    n_events = n_matched = 0
    for hand, events in sorted(hand_events(topology).items(), key=lambda kv: kv[0].pos):
        n_events += len(events)
        label = pos_label(hand.pos)
        tolerance = _separation([e.time for e in events]) / 2
        pairs, orphans = _pair_rows(sorted(starts[hand]), sorted(ends[hand]), tolerance)
        if orphans:
            warnings.append(f"{label}: START と END の対が取れない行が {orphans} 行あります")
        if not pairs:
            warnings.append(f"{label}: ログCに動作がありません")
            continue
        matched, skipped = _match_events(events, pairs, tolerance)
        if skipped:
            breaks[hand.robot_id].extend(skipped)
            warnings.append(
                f"{label}: ログBのイベント {len(events)} 件とログCの動作 {len(pairs)} 件が順に対応しません．"
                f"時刻で区切り直し，{len(matched)} 件を対応させました"
            )
        n_matched += len(matched)
        for event, (start, end) in matched:
            actions[hand.robot_id].append(Action(
                hand, event.kind, event.material_id, event.pos, topology.stage_of[event.pos].index,
                event.time, start, end,
            ))

    lags: dict[str, Lag] = {}
    for robot_id, rows in actions.items():
        rows.sort(key=lambda a: (a.event_time, a.start))
        lags[robot_id] = _measure_lag(rows)
        # 欠けをまたぐ区間は，移動や待ちの計算に使わない
        times = sorted(breaks[robot_id])
        for prev, cur in zip(rows, rows[1:]):
            i = bisect_right(times, prev.event_time)
            cur.resync = i < len(times) and times[i] <= cur.event_time
    return Alignment(dict(actions), lags, n_events, n_matched, warnings)
