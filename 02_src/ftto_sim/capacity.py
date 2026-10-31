"""設定から解析的に求めた各部位の能力（推定結果の照合用の真値）."""

from __future__ import annotations

from typing import Any

from .config import RobotConfig, SimConfig, StageConfig


def _robot_cycle(robot: RobotConfig, upstream: str, process: str, downstream: str) -> list[dict[str, Any]]:
    """入れ替え手順に沿った，1枚あたりの動作の列."""
    return [
        {"action": "PICK", "at": upstream, "s": robot.pick_s},
        {"action": "MOVE", "at": f"{upstream}->{process}", "s": robot.move_time(upstream, process)},
        {"action": "PICK", "at": process, "s": robot.pick_s},
        {"action": "PLACE", "at": process, "s": robot.place_s},
        {"action": "MOVE", "at": f"{process}->{downstream}", "s": robot.move_time(process, downstream)},
        {"action": "PLACE", "at": downstream, "s": robot.place_s},
        {"action": "MOVE", "at": f"{downstream}->{upstream}", "s": robot.move_time(downstream, upstream)},
    ]


def _exchanges_at(stages: tuple[StageConfig, ...], i: int) -> bool:
    """段 i の前後を同じロボットの別々のハンドが担うか（入れ替えが成り立つか）."""
    if not 0 < i < len(stages) - 1:
        return False
    feed, outbound = stages[i].feed, stages[i + 1].feed
    return (feed is not None and outbound is not None and feed.kind == outbound.kind == "robot"
            and feed.robot_id == outbound.robot_id and feed.hand != outbound.hand)


def theoretical_capacity(config: SimConfig, waits_at_target: bool) -> dict[str, Any]:
    """ロボットの飽和サイクル W，Pos の能力 c，段の能力 C，処理IDごとのライン限界の下限を返す.

    ライン限界は全部位の max である．部位間の干渉は含まないので，実際の限界はこれ以上になる．
    """
    stages = config.stages
    robots = {r.robot_id: r for r in config.robots}
    pids = list(config.input.process_id_weights)

    cycles: dict[str, list[list[dict[str, Any]]]] = {r: [] for r in robots}
    for i, stage in enumerate(stages):
        if _exchanges_at(stages, i):
            cycles[stage.feed.robot_id].append(
                _robot_cycle(robots[stage.feed.robot_id], stages[i - 1].name, stage.name, stages[i + 1].name)
            )
    robot_rows = {
        r: {"W_s": sum(part["s"] for cycle in cs for part in cycle) if cs else None, "cycles": cs}
        for r, cs in cycles.items()
    }

    stage_rows: dict[str, dict[str, Any]] = {}
    for i, stage in enumerate(stages):
        c: dict[str, float] = {}
        outbound = stages[i + 1].feed if i + 1 < len(stages) else None
        for pid in pids:
            value = stage.service.base_time(pid)
            if outbound is not None and outbound.kind == "robot":
                value += robots[outbound.robot_id].pick_s
            if stage.feed is not None and stage.feed.kind == "robot":
                value += robots[stage.feed.robot_id].place_s
            if _exchanges_at(stages, i) and not waits_at_target:
                value += robots[stage.feed.robot_id].move_time(stages[i - 1].name, stage.name)
            c[pid] = value
        stage_rows[stage.name] = {
            "positions": [f"{u}-{p}" for u, p in stage.positions],
            "c_s": c,
            "C_s": {pid: value / len(stage.positions) for pid, value in c.items()},
        }

    line: dict[str, dict[str, Any]] = {}
    for pid in pids:
        limits = {name: row["C_s"][pid] for name, row in stage_rows.items()}
        limits.update({r: row["W_s"] for r, row in robot_rows.items() if row["W_s"] is not None})
        bottleneck = max(limits, key=lambda k: limits[k])
        line[pid] = {"T_min_s": limits[bottleneck], "bottleneck": bottleneck}

    return {"waits_at_target": waits_at_target, "robots": robot_rows, "stages": stage_rows, "line": line}
