"""能力が変わっていないかの監視（algorithm.md 第8章）.

能力は1か月分をまとめて求めるので，期間の途中で変わると平均に埋もれる．
動作の長さと処理時間の中心を週ごとの時系列で出し，全体の値から離れた週を警告する．
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .actions import Alignment
from .service import ServiceTimes
from .stats import MIN_COUNT, center
from .topology import Topology, pos_label

WEEK_S = 7 * 86400
MEASURE_ACTION = "動作の長さ"
MEASURE_SERVICE = "処理時間"


@dataclass
class TrendPoint:
    week: int  # 0 始まり
    measure: str
    part: str
    process_id: str  # 動作の長さは空欄
    center_s: float
    n: int


def _points(
    measure: str, part: str, process_id: str, samples: list[tuple[float, float]], width: float,
) -> tuple[list[TrendPoint], list[str]]:
    by_week: dict[int, list[float]] = defaultdict(list)
    for t, value in samples:
        by_week[int(t // WEEK_S)].append(value)
    overall = center([v for _, v in samples], width)
    points = []
    warnings = []
    for week, values in sorted(by_week.items()):
        if len(values) < MIN_COUNT:
            continue
        point = TrendPoint(week, measure, part, process_id, center(values, width), len(values))
        points.append(point)
        if abs(point.center_s - overall) > width:
            warnings.append(
                f"{measure} {part} {process_id}: 第{week + 1}週の {point.center_s:.2f} s が，"
                f"全体の {overall:.2f} s から離れています（能力が変わった可能性があります）"
            )
    return points, warnings


def build_trend(
    alignment: Alignment, service: ServiceTimes, topology: Topology,
) -> tuple[list[TrendPoint], list[str]]:
    points: list[TrendPoint] = []
    warnings: list[str] = []
    default_width = max((alignment.width(r) for r in alignment.lags), default=1.0)
    for robot_id, actions in alignment.actions.items():
        groups: dict[tuple[str, tuple[str, str]], list[tuple[float, float]]] = defaultdict(list)
        for a in actions:
            groups[(a.kind, a.pos)].append((a.event_time, a.duration))
        for (kind, pos), samples in sorted(groups.items()):
            found, notes = _points(MEASURE_ACTION, f"{kind}@{pos_label(pos)}", "", samples, alignment.width(robot_id))
            points.extend(found)
            warnings.extend(notes)
    for (stage_index, pid), samples in sorted(service.samples.items()):
        if not samples.idle:
            continue
        pairs = list(zip(samples.idle_times, samples.idle))
        name = topology.stages[stage_index].name
        found, notes = _points(MEASURE_SERVICE, name, pid, pairs, default_width)
        points.extend(found)
        warnings.extend(notes)
    return points, warnings
