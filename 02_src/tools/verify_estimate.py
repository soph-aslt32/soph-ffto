"""合成データで推定処理を検証する（algorithm.md 第12.3章）.

設定値から合成ログを生成し，ログだけから推定した値を真値と照合する．推定処理そのものは生成器に依存しない．

例: python tools/verify_estimate.py --days 30
    python tools/verify_estimate.py --only interval
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import tempfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ftto_est import Estimate, Settings, estimate_dir  # noqa: E402
from ftto_est.quality import ESTIMATED  # noqa: E402
from ftto_sim import SimConfig, SimResult, load_config, load_config_str, simulate  # noqa: E402

SAMPLE = ROOT / "configs" / "sample.toml"
MIX = {"A": 0.5, "B": 0.3, "C": 0.2}
TOLERANCE_S = 1e-9


# --- 条件 ---


@dataclass(frozen=True)
class Scenario:
    group: str
    name: str
    build: Callable[[SimConfig], SimConfig]
    config_text: str | None = None  # サンプル設定の代わりに使う構成
    use_settings: bool = True  # 信号の時間と原点からの移動時間を，推定処理に与えるか


def settings_of(config: SimConfig) -> Settings:
    """実機でエンジニアが調整している設定値を，推定処理に渡す形にする."""
    leads: dict[str, float] = {}
    homes: dict[str, float] = {}
    robots = {r.robot_id: r for r in config.robots}
    for i, stage in enumerate(config.stages[:-1]):
        outbound = config.stages[i + 1].feed
        for unit, pos in stage.positions:
            label = f"{unit}-{pos}"
            leads[label] = stage.service.accept_lead_s
            if outbound is None or outbound.kind != "robot":
                continue
            robot = robots[outbound.robot_id]
            homes[label] = robot.home_time(stage.name) if robot.has_home else 0.0
    return Settings(leads, homes)


def _input(**kw: Any) -> Callable[[SimConfig], SimConfig]:
    def build(c: SimConfig) -> SimConfig:
        return dataclasses.replace(c, input=dataclasses.replace(c.input, **kw))
    return build


def _log(interval: float, **kw: Any) -> Callable[[SimConfig], SimConfig]:
    def build(c: SimConfig) -> SimConfig:
        c = _input(interval_s=interval)(c)
        return dataclasses.replace(c, log=dataclasses.replace(c.log, **kw))
    return build


def _robots(interval: float, **kw: Any) -> Callable[[SimConfig], SimConfig]:
    def build(c: SimConfig) -> SimConfig:
        c = _input(interval_s=interval)(c)
        return dataclasses.replace(c, robots=tuple(dataclasses.replace(r, **kw) for r in c.robots))
    return build


def _accept(interval: float, lead: float, home: float | None = None, **input_kw: Any) -> Callable[[SimConfig], SimConfig]:
    def build(c: SimConfig) -> SimConfig:
        c = _input(interval_s=interval, **input_kw)(c)
        stages = tuple(
            dataclasses.replace(s, service=dataclasses.replace(s.service, accept_lead_s=lead))
            if s.service.jitter_s > 0 else s
            for s in c.stages
        )
        robots = tuple(dataclasses.replace(r, home_move_s=home) for r in c.robots)
        return dataclasses.replace(c, stages=stages, robots=robots)
    return build


HOME_BY_STAGE = {"load": 4.0, "proc1": 2.0, "buffer": 3.0, "proc2": 1.5, "unload": 4.5}


def _home_by_stage(interval: float, lead: float) -> Callable[[SimConfig], SimConfig]:
    """原点との移動時間が段ごとに違う構成."""
    def build(c: SimConfig) -> SimConfig:
        c = _accept(interval, lead)(c)
        robots = tuple(dataclasses.replace(r, home_move_s=3.0, home_overrides=HOME_BY_STAGE) for r in c.robots)
        return dataclasses.replace(c, robots=robots)
    return build


def _slow_robot(robot_id: str, interval: float) -> Callable[[SimConfig], SimConfig]:
    """処理を短くして，ロボット robot_id だけを遅くする（ロボットが律速する2台の構成）.

    処理が移動より短いと入れ替えが起きず，原点との往復の分だけ限界が W より長くなる．ここでは判定を単独で見るので，原点を持たせない．
    """
    def build(c: SimConfig) -> SimConfig:
        c = _input(interval_s=interval, process_id_weights={"A": 1.0})(c)
        stages = tuple(
            dataclasses.replace(s, service=dataclasses.replace(s.service, times_s={"A": 5.0}, accept_lead_s=0.0))
            if s.service.jitter_s > 0 else s
            for s in c.stages
        )
        robots = tuple(
            dataclasses.replace(r, pick_s=3.0 if r.robot_id == robot_id else r.pick_s, home_move_s=None)
            for r in c.robots
        )
        return dataclasses.replace(c, stages=stages, robots=robots)
    return build


SHORT_PROCESS = """
[simulation]
duration_s = 86400
[input]
interval_s = {interval}
cassette_size = 20
[input.process_id_weights]
A = 1.0
[log]
lag_max_s = 2.0
logc_lag_offset_s = 1.0
logc_lag_jitter_s = 0.5
[[units]]
id = "U1"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[units]]
id = "U2"
positions = [{{ id = "P1", kind = "hand" }}, {{ id = "P2", kind = "hand" }}]
[[units]]
id = "U3"
positions = [{{ id = "P1", kind = "process" }}]
[[units]]
id = "U4"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[robots]]
id = "R1"
upper = ["U2", "P1"]
lower = ["U2", "P2"]
move_s = 3.0
pick_s = 2.0
place_s = 2.0
[[stages]]
name = "load"
positions = [["U1", "P1"]]
service = {{ time_s = 1.0 }}
[[stages]]
name = "proc"
positions = [["U3", "P1"]]
service = {{ times_s = {{ A = {process} }} }}
feed = {{ robot = "R1", hand = "upper" }}
[[stages]]
name = "out"
positions = [["U4", "P1"]]
service = {{ time_s = 1.0 }}
feed = {{ robot = "R1", hand = "lower" }}
"""

THREE_ROBOTS = """
[simulation]
duration_s = 86400
[input]
interval_s = {interval}
cassette_size = 25
[input.process_id_weights]
A = 0.6
B = 0.4
[log]
lag_max_s = 3.0
logc_lag_offset_s = 2.0
logc_lag_jitter_s = 1.0
[[units]]
id = "U1"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[units]]
id = "U2"
positions = [{{ id = "P1", kind = "hand" }}, {{ id = "P2", kind = "hand" }}, {{ id = "P3", kind = "process" }}, {{ id = "P4", kind = "process" }}]
[[units]]
id = "U3"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[units]]
id = "U4"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[units]]
id = "U5"
positions = [{{ id = "P1", kind = "hand" }}, {{ id = "P2", kind = "hand" }}, {{ id = "P3", kind = "process" }}]
[[units]]
id = "U6"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[units]]
id = "U7"
positions = [{{ id = "P1", kind = "hand" }}, {{ id = "P2", kind = "hand" }}, {{ id = "P3", kind = "process" }}, {{ id = "P4", kind = "process" }}, {{ id = "P5", kind = "process" }}, {{ id = "P6", kind = "process" }}]
[[units]]
id = "U8"
positions = [{{ id = "P1", kind = "conveyor" }}]
[[robots]]
id = "RA"
upper = ["U2", "P1"]
lower = ["U2", "P2"]
move_s = 2.0
pick_s = 3.0
place_s = 2.0
[[robots]]
id = "RB"
upper = ["U5", "P1"]
lower = ["U5", "P2"]
move_s = 4.0
pick_s = 2.0
place_s = 3.0
[[robots]]
id = "RC"
upper = ["U7", "P1"]
lower = ["U7", "P2"]
move_s = 3.0
pick_s = 2.0
place_s = 2.0
move_overrides = [{{ a = "c3", b = "out", s = 5.0 }}]
[[stages]]
name = "in"
positions = [["U1", "P1"]]
service = {{ time_s = 4.0 }}
[[stages]]
name = "p1"
positions = [["U2", "P3"], ["U2", "P4"]]
service = {{ times_s = {{ A = 90.0, B = 150.0 }}, jitter_s = 0.5 }}
feed = {{ robot = "RA", hand = "upper" }}
[[stages]]
name = "c1"
positions = [["U3", "P1"]]
service = {{ time_s = 6.0 }}
feed = {{ robot = "RA", hand = "lower" }}
[[stages]]
name = "c2"
positions = [["U4", "P1"]]
service = {{ time_s = 6.0 }}
feed = "auto"
[[stages]]
name = "p2"
positions = [["U5", "P3"]]
service = {{ times_s = {{ A = 50.0, B = 40.0 }}, jitter_s = 0.5 }}
feed = {{ robot = "RB", hand = "upper" }}
[[stages]]
name = "c3"
positions = [["U6", "P1"]]
service = {{ time_s = 5.0 }}
feed = {{ robot = "RB", hand = "lower" }}
[[stages]]
name = "p3"
positions = [["U7", "P3"], ["U7", "P4"], ["U7", "P5"], ["U7", "P6"]]
service = {{ times_s = {{ A = 200.0, B = 300.0 }}, jitter_s = 1.0 }}
feed = {{ robot = "RC", hand = "upper" }}
[[stages]]
name = "out"
positions = [["U8", "P1"]]
service = {{ time_s = 3.0 }}
feed = {{ robot = "RC", hand = "lower" }}
"""


def _text(template: str, **kw: Any) -> str:
    return template.format(**kw)


def scenarios() -> list[Scenario]:
    rows = [
        Scenario("interval", f"間隔 {t}", _input(interval_s=t))
        for t in (55.37, 62.37, 70.37, 86.37, 120.37, 137.37, 200.37, 600.37, 60.0, 86.4, 120.0)
    ]
    for pid, intervals in (("A", (50.37, 51.07, 51.67, 52.37, 53.37, 54.37)), ("B", (66.37, 67.37, 68.37, 69.37)),
                           ("C", (82.37, 83.37, 84.37, 85.37))):
        rows += [
            Scenario("judge", f"{pid} のみ {t}", _input(interval_s=t, process_id_weights={pid: 1.0}))
            for t in intervals
        ]
    rows += [
        Scenario("mode", "1枚ごとの処理ID 70.37", _input(interval_s=70.37, process_id_mode="per_material")),
        Scenario("mode", "1枚ごとの処理ID 120.37", _input(interval_s=120.37, process_id_mode="per_material")),
        Scenario("gap", "空き60s 86.37", _input(interval_s=86.37, cassette_gap_s=60.0)),
        Scenario("gap", "空き60s 137.37", _input(interval_s=137.37, cassette_gap_s=60.0)),
        Scenario("lag", "ログCのラグ -2s", _log(137.37, logc_lag_offset_s=-2.0)),
        Scenario("lag", "ログCのラグ +3s", _log(137.37, logc_lag_offset_s=3.0)),
        Scenario("lag", "ログCの行ごと 0〜3s", _log(137.37, logc_lag_jitter_s=3.0)),
        Scenario("lag", "ログBのIN 0s", _log(137.37, lag_max_s=0.0)),
        Scenario("lag", "ログBのIN 0〜5s", _log(137.37, lag_max_s=5.0)),
        Scenario("robot", "動作のばらつき 0.3s", _robots(137.37, jitter_s=0.3)),
        Scenario("robot", "外乱 1%で+10s", _robots(137.37, disturbance_ratio=0.01, disturbance_s=10.0)),
        Scenario("rows", "行の欠落 0.1%", _log(137.37, logc_row_missing_ratio=0.001)),
        Scenario("rows", "行の順序の乱れ", _log(137.37, shuffle_within_second=True)),
        Scenario("load", "低負荷＋高負荷", _input(interval_s=120.37, interval_schedule=((200, 120.37), (50, 58.37)))),
        Scenario("load", "低負荷＋律速", _input(interval_s=120.37, interval_schedule=((100, 120.37), (100, 45.37)))),
    ]
    rows += [
        Scenario("accept", f"受け入れ信号 {lead:.0f}s前 間隔{t}", _accept(t, lead))
        for lead in (5.0, 15.0, 30.0) for t in (60.37, 70.37, 86.37, 137.37, 200.37)
    ]
    rows += [
        Scenario("home", f"原点{home:.0f}s 信号{lead:.0f}s前 間隔{t}", _accept(t, lead, home))
        for home, lead in ((3.0, 0.0), (3.0, 2.0), (3.0, 5.0), (5.0, 2.0), (5.0, 20.0))
        for t in (60.37, 86.37, 137.37, 600.37)
    ]
    rows += [
        Scenario("home", f"原点が段ごとに違う 信号{lead:.0f}s前 間隔{t}", _home_by_stage(t, lead))
        for lead in (0.0, 3.0) for t in (86.37, 137.37)
    ]
    rows += [
        Scenario("nosetting", f"設定なし 原点{home:.0f}s 信号{lead:.0f}s前 間隔{t}", _accept(t, lead, home),
                 use_settings=False)
        for home, lead in ((3.0, 0.0), (3.0, 5.0)) for t in (86.37, 137.37)
    ]
    rows += [
        Scenario("nosetting", "設定なし 原点3s 信号0s前 高負荷を含む",
                 _accept(120.37, 0.0, 3.0, interval_schedule=((100, 120.37), (100, 45.37))), use_settings=False),
        Scenario("nosetting", "設定なし 原点5s 信号2s前 高負荷を含む",
                 _accept(120.37, 2.0, 5.0, interval_schedule=((100, 120.37), (100, 45.37))), use_settings=False),
    ]
    rows += [
        Scenario("layout", f"{robot}が遅い 間隔{t}", _slow_robot(robot, t))
        for robot in ("R1", "R2") for t in (15.37, 30.37)
    ]
    for interval in (40.37, 100.37):
        rows.append(Scenario("layout", f"3台 {interval}", lambda c: c, _text(THREE_ROBOTS, interval=interval)))
    for process, interval in ((3.0, 40.37), (3.0, 18.37), (3.0, 15.37), (30.0, 60.37), (30.0, 25.37)):
        rows.append(Scenario("layout", f"処理{process:.0f}s 間隔{interval}", lambda c: c,
                             _text(SHORT_PROCESS, interval=interval, process=process)))
    return rows


def _config(scenario: Scenario, days: float) -> SimConfig:
    if scenario.config_text is not None:
        base = load_config_str(scenario.config_text)
    else:
        base = load_config(SAMPLE)
        base = _input(process_id_weights=MIX)(base)
    return scenario.build(dataclasses.replace(base, duration_s=days * 86400.0))


# --- 真値との照合 ---


def _stage_index(e: Estimate, positions: list[str]) -> int:
    for stage in e.topology.stages:
        if ["-".join(p) for p in stage.positions] == positions:
            return stage.index
    raise KeyError(positions)


def _true_stage_intervals(result: SimResult) -> dict[tuple[str, str], float]:
    """(カセット, 段) ごとの入場間隔（実時刻）."""
    pid_cassette = {row[1]: row[3] for _, row in result.recorder.log_a}
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in result.recorder.occupancy:
        if row["stage"]:
            groups[(pid_cassette[row["material_id"]], row["stage"])].append(row["t_in"])
    out = {}
    for key, times in groups.items():
        if len(times) >= 2:
            times.sort()
            gaps = [b - a for a, b in zip(times, times[1:])]
            out[key] = statistics.fmean(gaps)
    return out


def _voluntary_hold(result: SimResult, config: SimConfig):
    """保持待ちのうち，段の別の層が空いているものを見分ける関数を返す（実時刻で判定する）."""
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in result.recorder.occupancy:
        spans[f"{row['unit_id']}-{row['pos_id']}"].append((row["t_in"], row["t_out"] or float("inf")))
    layers = {}
    for stage in config.stages:
        if len(stage.positions) > 1:
            labels = [f"{u}-{p}" for u, p in stage.positions]
            layers.update(dict.fromkeys(labels, labels))
    starts = {k: [a for a, _ in sorted(v)] for k, v in spans.items()}
    ends = {k: [b for _, b in sorted(v)] for k, v in spans.items()}

    def is_free(label: str, t: float) -> bool:
        i = bisect_right(starts.get(label, []), t) - 1
        return i < 0 or ends[label][i] <= t

    def voluntary(wait: dict[str, Any]) -> bool:
        labels = layers.get(wait["next_target"])
        return labels is not None and any(is_free(label, wait["t_start"]) for label in labels)

    return voluntary


def _true_robot_margins(result: SimResult, config: SimConfig) -> dict[tuple[str, str], float]:
    """(カセット, ロボット) ごとの 1枚あたりの空いている時間の中央値.

    周期 − W − 層の不足による保持待ち．原点への往復は，要求が来れば省ける時間なので余裕に含まれる．
    """
    cassette_of = {row[1]: row[3] for _, row in result.recorder.log_a}
    voluntary = _voluntary_hold(result, config)
    holds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for w in result.recorder.robot_waits():
        if w["category"] == "hold" and not voluntary(w):
            holds[w["robot_id"]].append(w)
    entry_stage = {}
    for i, stage in enumerate(config.stages):
        if stage.feed is not None and stage.feed.kind == "robot":
            entry_stage.setdefault(stage.feed.robot_id, config.stages[i - 1])
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for robot_id, stage in entry_stage.items():
        cycle = result.capacity["robots"][robot_id]["W_s"]
        if cycle is None:
            continue
        labels = {f"{u}-{p}" for u, p in stage.positions}
        entries = [op for op in result.recorder.robot_ops
                   if op["robot_id"] == robot_id and op["action"] == "PICK" and op["target"] in labels]
        rows = sorted(holds[robot_id], key=lambda w: w["t_start"])
        i = 0
        for prev, op in zip(entries, entries[1:]):
            held = 0.0
            while i < len(rows) and rows[i]["t_start"] < op["t_start"]:
                held += rows[i]["t_end"] - rows[i]["t_start"]
                i += 1
            groups[(cassette_of[op["material_id"]], robot_id)].append(op["t_start"] - prev["t_start"] - cycle - held)
    return {k: statistics.median(v) for k, v in groups.items()}


def _true_stage_capacity(truth: dict[str, Any], config: SimConfig, stage: Any, pid: str) -> float:
    """真の C．単層の段で同じロボットが前後を担うときは，C ≥ max(c, W_r)."""
    value = truth["stages"][stage.name]["C_s"][pid]
    index = [s.name for s in config.stages].index(stage.name)
    if len(stage.positions) == 1 and 0 < index < len(config.stages) - 1:
        feed, outbound = stage.feed, config.stages[index + 1].feed
        if feed.kind == outbound.kind == "robot" and feed.robot_id == outbound.robot_id and feed.hand != outbound.hand:
            return max(value, truth["robots"][feed.robot_id]["W_s"])
    return value


def _interval_of(config: SimConfig, cassette_index: int) -> float:
    schedule = config.input.interval_schedule
    if not schedule:
        return config.input.interval_s
    k = cassette_index % sum(n for n, _ in schedule)
    for n, interval in schedule:
        if k < n:
            return interval
        k -= n
    raise AssertionError


def judge_agreement(result: SimResult, config: SimConfig, e: Estimate) -> tuple[float, int]:
    """定常のカセット（直前の2個も同じ処理ID・同じ間隔）で，律速の判定が真の律速と一致する割合.

    真の律速は，投入間隔が真の限界より短ければ真のボトルネック，長ければ「なし」．
    """
    truth = result.capacity["line"]
    agree = total = 0
    cassettes = e.judgements
    for k in range(2, len(cassettes)):
        recent = cassettes[k - 2:k + 1]
        pids = {j.cassette.process_id for j in recent}
        intervals = {_interval_of(config, i) for i in range(k - 2, k + 1)}
        if len(pids) != 1 or len(intervals) != 1 or next(iter(pids)) not in truth:
            continue
        line = truth[next(iter(pids))]
        interval = intervals.pop()
        if abs(interval - line["T_min_s"]) < 0.25:
            continue  # 限界そのものの近くは，どちらの判定もありうる
        expected = None
        if interval < line["T_min_s"]:
            name = line["bottleneck"]
            stages = result.capacity["stages"]
            expected = "+".join(stages[name]["positions"]) if name in stages else name
        total += 1
        agree += cassettes[k].observed == expected
    return (agree / total if total else 1.0), total


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)] if ordered else 0.0


def compare(result: SimResult, config: SimConfig, e: Estimate) -> dict[str, Any]:
    truth = result.capacity
    pids = list(config.input.process_id_weights)
    row: dict[str, Any] = {"match": e.alignment.n_matched / max(e.alignment.n_events, 1)}

    robots = {r.robot_id: r for r in config.robots}
    g_err = 0.0
    for robot_id, actions in e.alignment.actions.items():
        for (kind, pos), v in e.motion.durations.items():
            if e.topology.hands and any(a.kind == kind and a.pos == pos for a in actions[:400]):
                expected = robots[robot_id].pick_s if kind == "PICK" else robots[robot_id].place_s
                g_err = max(g_err, abs(v.value - expected))
    row["g_err"] = g_err

    w_err = 0.0
    for robot_id, truth_row in truth["robots"].items():
        estimate = e.capacity.by_robot[robot_id]
        if truth_row["W_s"] is not None and estimate.value is not None:
            w_err = max(w_err, abs(estimate.value - truth_row["W_s"]))
    row["W_err"] = w_err

    c_err = 0.0
    s_quality = {}
    outside = []
    bad = 0
    for name, stage_truth in truth["stages"].items():
        stage = config.stages[[s.name for s in config.stages].index(name)]
        index = _stage_index(e, stage_truth["positions"])
        for pid in pids:
            value = e.capacity.by_stage.get((index, pid))
            if value is None or value.value is None:
                continue
            expected = _true_stage_capacity(truth, config, stage, pid)
            if value.quality == ESTIMATED:
                c_err = max(c_err, abs(value.value - expected))
            elif e.topology.stages[index].kind == "process":
                s_quality[f"{name}/{pid}"] = value.quality
            lo = (value.lo if value.lo is not None else 0.0) - 0.25
            hi = (value.hi if value.hi is not None else float("inf")) + 0.25
            if not lo <= expected <= hi:
                outside.append(f"{name}/{pid} {expected:.2f} [{lo:.2f},{hi:.2f}]")
                bad += 1
    row["C_err"] = c_err
    row["flags"] = ",".join(f"{k}:{v}" for k, v in s_quality.items())
    row["outside"] = ";".join(outside)

    intervals = _true_stage_intervals(result)
    errors = []
    rho_errors = []
    pid_of_cassette = {c.cassette_id: c for c in e.cassettes}
    for m in e.margins:
        if m.kind != "stage" or m.margin_s is None or m.capacity.quality != ESTIMATED:
            continue
        stage_index = next(s.index for s in e.topology.stages if s.name == m.part)
        cfg_stage = config.stages[stage_index]
        interval = intervals.get((m.cassette_id, cfg_stage.name))
        cassette = pid_of_cassette[m.cassette_id]
        if interval is None:
            continue
        expected_c = sum(truth["stages"][cfg_stage.name]["C_s"][p] * n for p, n in cassette.process_mix.items())
        expected_c /= cassette.n
        errors.append(abs(m.margin_s - (interval - expected_c)))
        rho_errors.append(abs(m.rho - expected_c / interval))
    row["margin_err"] = _percentile(errors, 0.95)
    row["rho_err"] = _percentile(rho_errors, 0.95)

    robot_truth = _true_robot_margins(result, config)
    robot_errors = [
        m.margin_s - robot_truth[(m.cassette_id, m.part)]
        for m in e.margins
        if m.kind == "robot" and m.margin_s is not None and (m.cassette_id, m.part) in robot_truth
    ]
    row["robot_err_mean"] = statistics.fmean(robot_errors) if robot_errors else 0.0
    row["robot_err"] = _percentile([abs(x) for x in robot_errors], 0.95)

    limits = []
    from ftto_est.bottleneck import limit_by_process
    for pid, limit in limit_by_process(e.topology, e.motion, e.capacity).items():
        expected = truth["line"].get(pid)
        if expected is None:
            continue
        stage_name = expected["bottleneck"]
        expected_part = stage_name
        if stage_name in truth["stages"]:
            expected_part = "+".join(truth["stages"][stage_name]["positions"])
        agrees = "○" if expected_part in limit.candidates else "×"
        value = limit.value.value
        text = "未観測" if value is None else f"{value:.2f}"
        limits.append(f"{pid}:{text}/{expected['T_min_s']:.2f}{agrees}")
    row["limits"] = " ".join(limits)

    statuses: dict[str, int] = defaultdict(int)
    for j in e.judgements:
        statuses[j.status] += 1
    row["status"] = " ".join(f"{k}{v}" for k, v in sorted(statuses.items()))
    row["agreement"], row["n_agreement"] = judge_agreement(result, config, e)
    row["warnings"] = len(e.warnings)
    return row


def saturated_limit(config: SimConfig, pid: str, days: float = 2.0) -> float:
    """処理IDを固定して極小の間隔で流し，定常の搬出間隔を真のライン限界として測る（T3）."""
    config = dataclasses.replace(
        _input(interval_s=1.0, process_id_weights={pid: 1.0}, interval_schedule=())(config),
        duration_s=days * 86400.0,
    )
    result = simulate(config)
    last = config.stages[-1].positions
    times = sorted(r["t_in"] for r in result.recorder.occupancy if (r["unit_id"], r["pos_id"]) in last)
    times = times[len(times) // 5:]
    return (times[-1] - times[0]) / (len(times) - 1)


def verify_limits() -> None:
    """低負荷のログだけから求めたライン限界が，飽和実行の真の限界の下限になっているか."""
    print("| 構成 | 処理ID | 低負荷からの推定 | 印 | 律速部位 | 真の限界（飽和実行） | 推定÷真値 |")
    print("|---|---|---|---|---|---|---|")
    cases = [("サンプル", s) for s in scenarios() if s.name == "間隔 137.37"]
    cases += [("3台", s) for s in scenarios() if s.name == "3台 100.37"]
    cases += [("処理3s", s) for s in scenarios() if s.name == "処理3s 間隔40.37"]
    cases += [("処理30s", s) for s in scenarios() if s.name == "処理30s 間隔60.37"]
    from ftto_est.bottleneck import limit_by_process
    for label, scenario in cases:
        config = _config(scenario, 30)
        result = simulate(config)
        with tempfile.TemporaryDirectory() as d:
            result.write(d)
            e = estimate_dir(d, settings_of(config))
        for pid, limit in limit_by_process(e.topology, e.motion, e.capacity).items():
            true = saturated_limit(config, pid)
            value = limit.value.value
            print(f"| {label} | {pid} | {value:.2f} | {limit.value.quality} | {limit.part} | {true:.2f} "
                  f"| {value / true:.3f} |")


def run(scenario: Scenario, days: float) -> tuple[Scenario, dict[str, Any], list[str]]:
    config = _config(scenario, days)
    result = simulate(config)
    with tempfile.TemporaryDirectory() as d:
        result.write(d)
        e = estimate_dir(d, settings_of(config) if scenario.use_settings else None)
    return scenario, compare(result, config, e), e.warnings[:3]


COLUMNS = (
    ("対応", "match", "{:.1%}"), ("g誤差", "g_err", "{:.2f}"), ("W誤差", "W_err", "{:.2f}"),
    ("C誤差", "C_err", "{:.2f}"), ("Δ誤差", "margin_err", "{:.2f}"), ("ρ誤差", "rho_err", "{:.3f}"),
    ("Δ_r誤差", "robot_err", "{:.2f}"), ("Δ_r偏り", "robot_err_mean", "{:+.2f}"),
    ("律速一致", "agreement", "{:.1%}"), ("件数", "n_agreement", "{}"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="合成データで推定処理を検証する")
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--limits", action="store_true", help="飽和実行の真の限界と照合する")
    parser.add_argument("--only", help="条件のグループ（interval, judge, mode, gap, lag, robot, rows, load, accept, home, nosetting, layout）")
    args = parser.parse_args()

    if args.limits:
        verify_limits()
        return 0

    selected = [s for s in scenarios() if args.only is None or s.group == args.only]
    results = [run(s, args.days) for s in selected]

    print("| 条件 | " + " | ".join(c[0] for c in COLUMNS) + " | 限界（推定/真値） | 判定 | 区間外 | 印 |")
    print("|---|" + "---|" * (len(COLUMNS) + 4))
    for scenario, row, notes in results:
        cells = [fmt.format(row[key]) for _, key, fmt in COLUMNS]
        print(f"| {scenario.name} | " + " | ".join(cells)
              + f" | {row['limits']} | {row['status']} | {row['outside'] or '-'} | {row['flags'] or '-'} |")
        for note in notes:
            print(f"<!-- {scenario.name}: {note} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
