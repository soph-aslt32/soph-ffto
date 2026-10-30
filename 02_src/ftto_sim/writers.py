"""ログX・A・B・Cと真値の記録・出力.

ログ（X/A/B/C）は実ログを模した形式で，時刻は秒単位（切り捨て）．
真値（truth_*）は推定結果の照合用で，時刻はシミュレーション開始からの秒（小数）．
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import SimConfig
from .topology import Material, Position, Topology

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_X_HEADER = ["seq", "unit_id", "pos_id", "pos_kind", "robot_id", "hand"]
LOG_A_HEADER = ["time", "material_id", "process_id", "cassette_id"]
LOG_B_HEADER = ["time", "material_id", "unit_id", "pos_id", "event"]
LOG_C_HEADER = ["time", "unit_id", "pos_id", "event", "action", "target"]
TRUTH_OCCUPANCY_HEADER = [
    "material_id", "process_id", "unit_id", "pos_id", "pos_kind", "stage",
    "t_in", "t_service_end", "t_out", "service_s",
]
TRUTH_ROBOT_OPS_HEADER = ["robot_id", "hand", "unit_id", "pos_id", "action", "target", "t_start", "t_end"]

# ログCの action 名．ハンド以外の Pos（処理層・コンベアなど）の動作を記録する場合に使う
SERVICE_ACTIONS = {"process": "PROCESS", "conveyor": "CONVEY"}


@dataclass
class Recorder:
    config: SimConfig
    topology: Topology
    rng: random.Random  # 打刻ラグ・ログCの欠落用（シミュレーションの挙動とは独立）
    log_a: list[tuple[float, list[Any]]] = field(default_factory=list)
    log_b: list[tuple[float, list[Any]]] = field(default_factory=list)
    log_c: list[tuple[float, list[Any]]] = field(default_factory=list)
    occupancy: list[dict[str, Any]] = field(default_factory=list)
    robot_ops: list[dict[str, Any]] = field(default_factory=list)
    _open: dict[tuple[str, tuple[str, str]], dict[str, Any]] = field(default_factory=dict)

    # --- 打刻 ---

    @staticmethod
    def stamp(t: float) -> int:
        return math.floor(t + 1e-9)

    def _lag(self) -> float:
        lo, hi = self.config.log.lag_min_s, self.config.log.lag_max_s
        return self.rng.uniform(lo, hi) if hi > lo else lo

    # --- イベント ---

    def intake(self, t: float, material: Material) -> None:
        self.log_a.append((t, [self.stamp(t), material.material_id, material.process_id, material.cassette_id]))

    def transfer(self, t: float, material: Material, src: Position | None, dst: Position | None) -> None:
        """src から dst へのマテリアルの移動．src=None は取り込み，dst=None は払い出し."""
        if src is not None:
            self.log_b.append((t, [self.stamp(t), material.material_id, src.unit_id, src.pos_id, "OUT"]))
            row = self._open.pop((material.material_id, src.key))
            row["t_out"] = t
        if dst is not None:
            t_in = t if src is None else t + self._lag()
            self.log_b.append((t_in, [self.stamp(t_in), material.material_id, dst.unit_id, dst.pos_id, "IN"]))
            row = {
                "material_id": material.material_id,
                "process_id": material.process_id,
                "unit_id": dst.unit_id,
                "pos_id": dst.pos_id,
                "pos_kind": dst.kind,
                "stage": dst.stage.name if dst.stage else "",
                "t_in": t,
                "t_service_end": None,
                "t_out": None,
                "service_s": None,
            }
            self.occupancy.append(row)
            self._open[(material.material_id, dst.key)] = row

    def service(self, material: Material, pos: Position, t_start: float, duration: float) -> None:
        row = self._open[(material.material_id, pos.key)]
        row["service_s"] = duration
        row["t_service_end"] = t_start + duration
        if pos.kind in self.config.log.logc_kinds:
            action = SERVICE_ACTIONS.get(pos.kind, pos.kind.upper())
            self._action(pos, action, pos.label, t_start, t_start + duration)

    def robot_op(self, robot_id: str, hand: Position, hand_name: str, action: str, target: str,
                 t_start: float, t_end: float) -> None:
        self.robot_ops.append({
            "robot_id": robot_id, "hand": hand_name, "unit_id": hand.unit_id, "pos_id": hand.pos_id,
            "action": action, "target": target, "t_start": t_start, "t_end": t_end,
        })
        if hand.kind in self.config.log.logc_kinds:
            self._action(hand, action, target, t_start, t_end)

    def _action(self, pos: Position, action: str, target: str, t_start: float, t_end: float) -> None:
        if self.rng.random() < self.config.log.logc_missing_ratio:
            action, target = "", ""
        for t, event in ((t_start, "START"), (t_end, "END")):
            self.log_c.append((t, [self.stamp(t), pos.unit_id, pos.pos_id, event, action, target]))

    # --- 集計・出力 ---

    def summary(self, horizon: float, n_input: int, n_completed: int) -> dict[str, Any]:
        per_pos: dict[str, dict[str, Any]] = {}
        for p in self.topology.positions:
            per_pos[p.label] = {
                "kind": p.kind, "stage": p.stage.name if p.stage else "",
                "n_materials": 0, "occupied_s": 0.0, "service_s": 0.0,
            }
        for row in self.occupancy:
            s = per_pos[f"{row['unit_id']}-{row['pos_id']}"]
            s["n_materials"] += 1
            s["occupied_s"] += (row["t_out"] if row["t_out"] is not None else horizon) - row["t_in"]
            if row["service_s"] is not None:
                s["service_s"] += min(row["t_service_end"], horizon) - row["t_in"]
        per_robot: dict[str, float] = {r.robot_id: 0.0 for r in self.topology.robots}
        for op in self.robot_ops:
            per_robot[op["robot_id"]] += op["t_end"] - op["t_start"]
        return {
            "horizon_s": horizon,
            "n_input": n_input,
            "n_completed": n_completed,
            "positions": per_pos,
            "robots_busy_s": per_robot,
        }

    def write(self, out_dir: Path, summary: dict[str, Any]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        start = self.config.start

        def fmt(rows: list[tuple[float, list[Any]]]) -> list[list[Any]]:
            ordered = sorted(enumerate(rows), key=lambda x: (x[1][1][0], x[0]))
            return [[_fmt_time(start, r[0]), *r[1:]] for _, (_, r) in ordered]

        _write_csv(out_dir / "log_x.csv", LOG_X_HEADER, self._log_x_rows())
        _write_csv(out_dir / "log_a.csv", LOG_A_HEADER, fmt(self.log_a))
        _write_csv(out_dir / "log_b.csv", LOG_B_HEADER, fmt(self.log_b))
        _write_csv(out_dir / "log_c.csv", LOG_C_HEADER, fmt(self.log_c))
        _write_csv(out_dir / "truth_occupancy.csv", TRUTH_OCCUPANCY_HEADER,
                   [[_num(r[k]) for k in TRUTH_OCCUPANCY_HEADER] for r in self.occupancy])
        _write_csv(out_dir / "truth_robot_ops.csv", TRUTH_ROBOT_OPS_HEADER,
                   [[_num(r[k]) for k in TRUTH_ROBOT_OPS_HEADER] for r in self.robot_ops])
        with open(out_dir / "truth_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _log_x_rows(self) -> list[list[Any]]:
        hand_owner = {}
        for r in self.topology.robots:
            for name, pos in r.hands.items():
                hand_owner[pos.key] = (r.robot_id, name)
        rows = []
        for i, p in enumerate(self.topology.positions, start=1):
            robot_id, hand = hand_owner.get(p.key, ("", ""))
            rows.append([i, p.unit_id, p.pos_id, p.kind, robot_id, hand])
        return rows


def _fmt_time(start: datetime, seconds: int) -> str:
    return (start + timedelta(seconds=seconds)).strftime(TIME_FORMAT)


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
        w.writerows(rows)
