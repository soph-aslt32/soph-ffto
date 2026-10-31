"""ログから各部位の能力・余裕・律速を求める一連の処理."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Alignment, align_actions
from .bottleneck import Calibration, CassetteJudgement, calibrate, judge_cassettes
from .capacity import Capacity, build_capacity
from .logs import Logs, load_logs
from .margin import Cassette, PartMargin, build_cassettes, build_margins
from .motion import BasicExchangeCycle, CycleRule, Motion, measure_motion
from .service import ServiceTimes, estimate_service
from .topology import Topology, build_topology
from .trend import TrendPoint, build_trend
from .waits import CycleWaits, collect_waits


@dataclass
class Estimate:
    topology: Topology
    alignment: Alignment
    motion: Motion
    service: ServiceTimes
    capacity: Capacity
    cassettes: list[Cassette]
    waits: dict[str, list[CycleWaits]]
    margins: list[PartMargin]
    judgements: list[CassetteJudgement]
    calibrations: list[Calibration]
    trend: list[TrendPoint]
    warnings: list[str]


def estimate(logs: Logs, rule: CycleRule | None = None) -> Estimate:
    topology = build_topology(logs)
    alignment = align_actions(logs, topology)
    motion = measure_motion(alignment, topology, rule or BasicExchangeCycle())
    pid_of = {row.material_id: row.process_id for row in logs.intakes}
    service = estimate_service(alignment, motion, topology, pid_of)
    process_ids = sorted(set(pid_of.values()))
    capacity = build_capacity(topology, motion, service, process_ids)
    cassettes = build_cassettes(logs)
    waits = collect_waits(alignment, motion, topology)
    margins = build_margins(cassettes, topology, motion, capacity, waits, alignment.actions)
    judgements = judge_cassettes(cassettes, topology, motion, capacity, waits, alignment, margins)
    calibrations = calibrate(judgements)
    trend, trend_warnings = build_trend(alignment, service, topology)
    warnings = [*topology.warnings, *alignment.warnings, *motion.warnings, *service.warnings, *trend_warnings]
    return Estimate(topology, alignment, motion, service, capacity, cassettes, waits, margins, judgements,
                    calibrations, trend, warnings)


def estimate_dir(log_dir) -> Estimate:
    return estimate(load_logs(log_dir))
