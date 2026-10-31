"""標準形式のログ（X/A/B/C）の読み込み.

時刻は，最も早い行を 0 とする整数の秒に直す．ログごとに別の時計だが，起点だけは共通にする．
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PosKey = tuple[str, str]  # (UnitID, PosID)

IN = "IN"
OUT = "OUT"
START = "START"
END = "END"


@dataclass(frozen=True, slots=True)
class PosInfo:
    seq: int
    key: PosKey
    kind: str
    robot_id: str  # ハンド以外は空欄
    hand: str  # ハンド以外は空欄


@dataclass(frozen=True, slots=True)
class Intake:
    time: int
    material_id: str
    process_id: str
    cassette_id: str


@dataclass(frozen=True, slots=True)
class Transition:
    time: int
    material_id: str
    pos: PosKey
    event: str


@dataclass(frozen=True, slots=True)
class ActionRow:
    time: int
    pos: PosKey
    event: str


@dataclass
class Logs:
    origin: datetime
    positions: list[PosInfo]
    intakes: list[Intake]
    transitions: list[Transition]
    actions: list[ActionRow]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _parse_times(rows: list[dict[str, str]]) -> dict[str, datetime]:
    # 同じ秒の行が多いので，文字列ごとに一度だけ解釈する
    return {s: datetime.fromisoformat(s) for s in {r["time"] for r in rows}}


def load_logs(log_dir: str | Path) -> Logs:
    """ディレクトリから log_x/a/b/c.csv を読み込む."""
    log_dir = Path(log_dir)
    raw_x = _read_rows(log_dir / "log_x.csv")
    raw_a = _read_rows(log_dir / "log_a.csv")
    raw_b = _read_rows(log_dir / "log_b.csv")
    raw_c = _read_rows(log_dir / "log_c.csv")

    parsed = _parse_times(raw_a) | _parse_times(raw_b) | _parse_times(raw_c)
    if not parsed:
        raise ValueError(f"{log_dir}: ログに行がありません")
    origin = min(parsed.values())
    seconds = {s: int((t - origin).total_seconds()) for s, t in parsed.items()}

    return Logs(
        origin=origin,
        positions=sorted(
            (
                PosInfo(int(r["seq"]), (r["unit_id"], r["pos_id"]), r["pos_kind"], r["robot_id"], r["hand"])
                for r in raw_x
            ),
            key=lambda p: p.seq,
        ),
        intakes=[Intake(seconds[r["time"]], r["material_id"], r["process_id"], r["cassette_id"]) for r in raw_a],
        transitions=[
            Transition(seconds[r["time"]], r["material_id"], (r["unit_id"], r["pos_id"]), r["event"]) for r in raw_b
        ],
        actions=[ActionRow(seconds[r["time"]], (r["unit_id"], r["pos_id"]), r["event"]) for r in raw_c],
    )
