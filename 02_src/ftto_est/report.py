"""推定結果の出力."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .bottleneck import STATUS_NONE, STATUS_STEADY, STATUS_TRANSIENT, limit_by_process, month_limit
from .margin import KIND_HAND, KIND_POS, KIND_ROBOT, KIND_STAGE, PartMargin
from .pipeline import Estimate
from .quality import UNCALIBRATED, Value
from .topology import pos_label


def _num(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 3)
    return v


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[_num(v) for v in row] for row in rows])


def _value_cells(v: Value) -> list[Any]:
    return [v.value, v.lo, v.hi, v.quality]


def capacity_rows(e: Estimate) -> list[list[Any]]:
    rows = []
    for (pos, pid), v in sorted(e.capacity.by_pos.items()):
        rows.append([KIND_POS, pos_label(pos), pid, *_value_cells(v)])
    for (index, pid), v in sorted(e.capacity.by_stage.items()):
        rows.append([KIND_STAGE, e.topology.stages[index].name, pid, *_value_cells(v)])
    for robot_id, v in sorted(e.capacity.by_robot.items()):
        rows.append([KIND_ROBOT, robot_id, "", *_value_cells(v)])
    for pos, hand in sorted(e.topology.hands.items()):
        rows.append([KIND_HAND, pos_label(pos), "", *_value_cells(e.capacity.by_robot[hand.robot_id])])
    return rows


def service_rows(e: Estimate) -> list[list[Any]]:
    rows = []
    for (index, pid), s in sorted(e.service.by_stage.items()):
        rows.append([KIND_STAGE, e.topology.stages[index].name, pid, *_value_cells(s.value),
                     s.n_idle, s.n_front, s.n_total, s.adoption])
    for (pos, pid), s in sorted(e.service.by_pos.items()):
        rows.append([KIND_POS, pos_label(pos), pid, *_value_cells(s.value), s.n_idle, s.n_front, s.n_total,
                     s.adoption])
    return rows


def motion_rows(e: Estimate) -> list[list[Any]]:
    rows = []
    for (kind, pos), v in sorted(e.motion.durations.items()):
        rows.append([kind, pos_label(pos), "", *_value_cells(v)])
    for robot_id, moves in sorted(e.motion.moves.items()):
        for (src, dst), v in sorted(moves.table.items()):
            stages = e.topology.stages
            rows.append([f"MOVE {robot_id}", stages[src].name, stages[dst].name, *_value_cells(v)])
    return rows


def margin_rows(e: Estimate) -> list[list[Any]]:
    return [
        [m.cassette_id, m.kind, m.part, *_value_cells(m.capacity), m.interval_s, m.margin_s, m.rho,
         m.wait_idle_s, m.wait_process_s, m.wait_hold_s, m.hold_target, m.hold_mean_s, m.hold_min_s,
         m.hold_excess_s]
        for m in e.margins
    ]


def month_margin_rows(e: Estimate) -> list[list[Any]]:
    """期間全体の部位ごとの余裕．枚数で重み付けした平均と最小．過渡のカセットは最小の計算に入れない."""
    n_of = {c.cassette_id: c.n for c in e.cassettes}
    pid_of = {c.cassette_id: c.process_id for c in e.cassettes}
    status_of = {j.cassette.cassette_id: j.status for j in e.judgements}
    groups: dict[tuple[str, str, str], list[PartMargin]] = defaultdict(list)
    for m in e.margins:
        if m.margin_s is None or m.rho is None:
            continue
        groups[(m.kind, m.part, pid_of[m.cassette_id])].append(m)
        groups[(m.kind, m.part, "全体")].append(m)

    rows = []
    for (kind, part, pid), items in sorted(groups.items()):
        weight = sum(n_of[m.cassette_id] for m in items)
        steady = [m for m in items if status_of.get(m.cassette_id) != STATUS_TRANSIENT] or items
        rows.append([
            kind, part, pid, len(items), weight,
            sum(m.margin_s * n_of[m.cassette_id] for m in items) / weight,
            min(m.margin_s for m in steady),
            sum(m.rho * n_of[m.cassette_id] for m in items) / weight,
            max(m.rho for m in steady),
        ])
    return rows


def bottleneck_rows(e: Estimate) -> list[list[Any]]:
    rows = []
    for j in e.judgements:
        limit = j.limit
        rows.append([
            j.cassette.cassette_id, j.cassette.process_id, j.cassette.n, j.interval_s,
            *_value_cells(limit.value), limit.part, "|".join(limit.candidates), limit.runner_up,
            limit.runner_up_gap_s, limit.tie, j.observed, j.status,
            ";".join(f"{k}={v:.2f}" for k, v in sorted(j.hold_share.items())),
        ])
    return rows


def summary(e: Estimate) -> dict[str, Any]:
    limits = limit_by_process(e.topology, e.motion, e.capacity)
    kappa = {(c.process_id, c.part): c for c in e.calibrations}
    by_process = {}
    for pid, limit in sorted(limits.items()):
        part = limit.part
        calibrated = next((kappa[(pid, c)] for c in limit.candidates if (pid, c) in kappa), None)
        by_process[pid] = {
            "T_line_min_s": limit.value.value,
            "lo_s": limit.value.lo,
            "hi_s": limit.value.hi,
            "quality": limit.value.quality,
            "bottleneck": part,
            "candidates": limit.candidates,
            "runner_up": limit.runner_up,
            "runner_up_gap_s": limit.runner_up_gap_s,
            "tie": limit.tie,
            "kappa": calibrated.kappa if calibrated else None,
            "n_calibration_cassettes": calibrated.n_cassettes if calibrated else 0,
            "calibration": "校正済み" if calibrated else UNCALIBRATED,
        }
    counts: dict[str, int] = defaultdict(int)
    for j in e.judgements:
        counts[j.status] += 1
    return {
        "n_cassettes": len(e.cassettes),
        "n_materials": sum(c.n for c in e.cassettes),
        "n_events": e.alignment.n_events,
        "n_matched_events": e.alignment.n_matched,
        "log_lag_s": {r: {"center": lag.center_s, "spread": lag.spread_s} for r, lag in e.alignment.lags.items()},
        "stages": [
            {"name": s.name, "kind": s.kind, "positions": [pos_label(p) for p in s.positions],
             "feed": s.feed.robot_id + "/" + s.feed.name if s.feed else None,
             "outbound": s.outbound.robot_id + "/" + s.outbound.name if s.outbound else None}
            for s in e.topology.stages
        ],
        "unused_layers": [pos_label(p) for p in e.topology.unused_layers],
        "service_delay_s": {
            e.topology.stages[i].name: {"lo": d.lo, "hi": d.hi} for i, d in e.service.delays.items() if d.hi > 0
        },
        "line_limit_by_process": by_process,
        "line_limit_month": month_limit(e.judgements),
        "cassette_status": dict(counts),
        "warnings": e.warnings,
    }


def write_report(e: Estimate, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "capacity.csv", ["part_kind", "part", "process_id", "value_s", "lo_s", "hi_s", "quality"],
               capacity_rows(e))
    _write_csv(out / "service_time.csv",
               ["part_kind", "part", "process_id", "value_s", "lo_s", "hi_s", "quality", "n_idle", "n_front",
                "n_total", "adoption"], service_rows(e))
    _write_csv(out / "motion.csv", ["type", "at", "to", "value_s", "lo_s", "hi_s", "quality"], motion_rows(e))
    _write_csv(out / "margin_cassette.csv",
               ["cassette_id", "part_kind", "part", "capacity_s", "capacity_lo_s", "capacity_hi_s", "quality",
                "interval_s", "margin_s", "rho", "wait_idle_s", "wait_process_s", "wait_hold_s", "hold_target",
                "hold_mean_s", "hold_min_s", "hold_excess_s"], margin_rows(e))
    _write_csv(out / "margin_month.csv",
               ["part_kind", "part", "process_id", "n_cassettes", "n_materials", "margin_mean_s", "margin_min_s",
                "rho_mean", "rho_max"], month_margin_rows(e))
    _write_csv(out / "bottleneck_cassette.csv",
               ["cassette_id", "process_id", "n", "interval_s", "T_line_min_s", "lo_s", "hi_s", "quality",
                "predicted", "candidates", "runner_up", "runner_up_gap_s", "tie", "observed", "status",
                "hold_share"], bottleneck_rows(e))
    _write_csv(out / "capacity_trend.csv", ["week", "measure", "part", "process_id", "center_s", "n"],
               [[p.week + 1, p.measure, p.part, p.process_id, p.center_s, p.n] for p in e.trend])
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary(e), f, ensure_ascii=False, indent=2)


def format_summary(e: Estimate) -> str:
    """標準出力に出す要約."""
    data = summary(e)
    matched = data["n_matched_events"] / max(data["n_events"], 1)
    lines = [f"カセット {data['n_cassettes']} 個，マテリアル {data['n_materials']} 枚，動作の対応 {matched:.1%}"]
    for pid, row in data["line_limit_by_process"].items():
        value = "未観測" if row["T_line_min_s"] is None else f"{row['T_line_min_s']:.2f} s"
        lines.append(f"  処理ID {pid}: ライン限界の下限 {value}（{row['quality']}），"
                     f"律速 {row['bottleneck']}，{row['calibration']}")
    status = data["cassette_status"]
    lines.append(f"  カセットの判定: 律速なし {status.get(STATUS_NONE, 0)}，律速 {status.get(STATUS_STEADY, 0)}，"
                 f"過渡 {status.get(STATUS_TRANSIENT, 0)}，その他 "
                 f"{data['n_cassettes'] - sum(status.get(k, 0) for k in (STATUS_NONE, STATUS_STEADY, STATUS_TRANSIENT))}")
    for warning in data["warnings"]:
        lines.append(f"  警告: {warning}")
    return "\n".join(lines)
