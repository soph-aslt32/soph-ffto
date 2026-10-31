"""部位の能力（algorithm.md 第6章）.

能力は1か月分をまとめて，(Pos/段, 処理ID) ごとに求める．部位は次の名前で区別する．

- Pos：`U5-P7`
- 段：Pos を `+` でつないだ名前（`U5-P7+U5-P8+U5-P9`）．単層の段は Pos と同じ名前になる
- ロボット：`R1`
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .actions import PICK, PLACE
from .logs import PosKey
from .motion import Motion
from .quality import BORROWED, ESTIMATED, Value, worse
from .service import ServiceTimes
from .topology import Stage, Topology


@dataclass
class Capacity:
    by_pos: dict[tuple[PosKey, str], Value]  # c_{p,k}
    by_stage: dict[tuple[int, str], Value]  # C_{S,k}
    by_robot: dict[str, Value]  # W_r
    process_ids: list[str]


def _pos_capacity(service: Value, pos: PosKey, stage: Stage, motion: Motion) -> Value:
    """処理時間 ＋ 取る動作 ＋ 置く動作．ロボットは処理層の前で待つので，処理完了後の移動は含まない."""
    value = service
    if stage.outbound is not None:
        value = value.plus(motion.duration(PICK, pos, stage.index))
    if stage.feed is not None:
        value = value.plus(motion.duration(PLACE, pos, stage.index))
    return value


def _mean(values: list[Value]) -> Value:
    """同じ段の層の平均．品質は最も悪いものを取る（品質の順は Value.plus に従う）."""
    def average(items: list[float | None]) -> float | None:
        return None if any(x is None for x in items) else statistics.fmean(items)

    total = values[0]
    for v in values[1:]:
        total = total.plus(v)
    return Value(
        average([v.value for v in values]),
        average([v.lo for v in values]),
        average([v.hi for v in values]),
        total.quality,
    )


def build_capacity(topology: Topology, motion: Motion, service: ServiceTimes, process_ids: list[str]) -> Capacity:
    by_pos: dict[tuple[PosKey, str], Value] = {}
    by_stage: dict[tuple[int, str], Value] = {}
    exchanging = {c.process: c.robot_id for cs in motion.cycles.values() for c in cs if c.process is not None}
    for stage in topology.stages:
        for pid in process_ids:
            # 同じ段の層は処理時間を共有する．観測の無い層は借用の印を付ける
            shared = service.by_stage.get((stage.index, pid))
            shared_value = shared.value if shared is not None else Value.unobserved()
            layers = []
            for pos in stage.positions:
                own = service.by_pos.get((pos, pid))
                source = shared_value
                if (own is None or own.n_total == 0) and source.value is not None:
                    source = Value(source.value, source.lo, source.hi, worse(source.quality, BORROWED))
                by_pos[(pos, pid)] = _pos_capacity(source, pos, stage, motion)
                layers.append(by_pos[(pos, pid)])
            value = _mean(layers).scaled(1 / len(stage.positions))
            if value.quality == BORROWED:
                # 段の値としては，使われていない層があっても借用ではない
                value = Value(value.value, value.lo, value.hi, ESTIMATED)
            robot_id = exchanging.get(stage.index)
            if robot_id is not None and len(stage.positions) == 1:
                value = _at_least(value, motion.capacity[robot_id])
            by_stage[(stage.index, pid)] = value
    return Capacity(by_pos, by_stage, dict(motion.capacity), process_ids)


def _at_least(value: Value, floor: Value) -> Value:
    """単層の段で同じロボットが前後を担うときは，C ≥ max(c, W_r)."""
    if value.value is None or floor.value is None or floor.value <= value.value:
        return value
    return floor
