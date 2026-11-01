"""推定処理のテスト．合成ログから，真値を使わずに求めた値を真値と照合する."""

import dataclasses
import json
from datetime import datetime

import pytest

from ftto_est import Settings, estimate_dir, load_logs, load_settings
from ftto_est.__main__ import main
from ftto_est.actions import PICK, PLACE, _common_offset, _is_shifted, align_actions
from ftto_est.bottleneck import (
    STATUS_NONE,
    STATUS_STEADY,
    STATUS_TRANSIENT,
    limit_by_process,
    line_limit,
)
from ftto_est.logs import IN, OUT, ActionRow, Intake, Logs, PosInfo, Transition
from ftto_est.quality import (
    BIASED,
    ESTIMATED,
    INTERVAL_ONLY,
    LOWER_ONLY,
    UNOBSERVED,
    UPPER_ONLY,
    Value,
    worse,
)
from ftto_est.stats import center, low_center, lowest_common, mean_interval
from ftto_est.topology import build_topology, build_visits
from ftto_est.waits import _hold_target
from ftto_sim import load_config_str, simulate
from tools.verify_estimate import settings_of

from .helpers import sample_config, with_log

PROC1 = "U3-P1"
PROC2 = "U5-P7+U5-P8+U5-P9"
MIX = {"A": 0.5, "B": 0.3, "C": 0.2}
TRUE_C = {
    (PROC1, "A"): 44.0, (PROC1, "B"): 64.0, (PROC1, "C"): 84.0,
    (PROC2, "A"): 154 / 3, (PROC2, "B"): 204 / 3, (PROC2, "C"): 244 / 3,
}


def run_estimate(config, tmp_path_factory):
    """実機でエンジニアが調整している設定値を与えて推定する."""
    result = simulate(config)
    log_dir = tmp_path_factory.mktemp("logs")
    result.write(log_dir)
    return result, estimate_dir(log_dir, settings_of(config))


def config_for(interval, days=3, **input_overrides):
    config = sample_config(interval_s=interval, process_id_weights=MIX, **input_overrides)
    return dataclasses.replace(config, duration_s=days * 86400.0)


def stage_capacity(e, name, pid):
    index = next(s.index for s in e.topology.stages if s.name == name)
    return e.capacity.by_stage[(index, pid)]


@pytest.fixture(scope="module")
def low_load(tmp_path_factory):
    return run_estimate(config_for(137.37), tmp_path_factory)


@pytest.fixture(scope="module")
def very_low_load(tmp_path_factory):
    return run_estimate(config_for(600.37, days=30), tmp_path_factory)


# --- 統計 ---


def test_center_ignores_disturbed_values():
    assert center([2, 2, 3, 2, 3, 40], 1.0) == pytest.approx(2.4)


def test_lowest_common_skips_rare_values():
    assert lowest_common([1] + [3] * 5 + [9] * 8) == 3
    assert lowest_common([1, 2, 3]) is None


def test_low_center_ignores_waits():
    assert low_center([3] * 6 + [4] * 4 + [30, 50, 80], 2.0) == pytest.approx(3.4)


def test_mean_interval_is_not_fooled_by_bursts():
    # 3層の束：短い間隔が過半でも，平均は実際の間隔になる
    times = []
    t = 0.0
    for _ in range(10):
        for gap in (20.0, 20.0, 120.0):
            times.append(t)
            t += gap
    assert mean_interval(times, burst=3) == pytest.approx((times[-1] - times[0]) / (len(times) - 1))


def test_mean_interval_drops_a_stop():
    times = [0.0, 60.0, 120.0, 180.0, 3780.0, 3840.0]
    assert mean_interval(times) == pytest.approx(60.0)


def test_mean_interval_needs_two_times():
    assert mean_interval([10.0]) is None


# --- 品質の印 ---


def test_worse_keeps_the_weaker_quality():
    assert worse(ESTIMATED, BIASED) == BIASED
    assert worse(INTERVAL_ONLY, ESTIMATED) == INTERVAL_ONLY
    assert worse(UPPER_ONLY, UNOBSERVED) == UNOBSERVED


def test_sum_with_an_unobserved_value_has_no_value():
    assert Value.of(3.0).plus(Value.unobserved()).value is None


# --- 前処理 ---


def small_logs(transitions=None, actions=None):
    positions = [
        PosInfo(1, ("U1", "P1"), "conveyor", "", ""),
        PosInfo(2, ("U2", "P1"), "hand", "R", "upper"),
        PosInfo(3, ("U2", "P2"), "hand", "R", "lower"),
        PosInfo(4, ("U3", "P1"), "process", "", ""),
        PosInfo(5, ("U3", "P2"), "process", "", ""),
        PosInfo(6, ("U4", "P1"), "conveyor", "", ""),
    ]
    intakes = [Intake(0, f"M{i}", "A", "C1") for i in range(1, 8)]
    return Logs(datetime(2026, 1, 1), positions, intakes, transitions or [], actions or [])


def visit_rows(material, base, layer="P1", out_lag=0):
    """M: load → upper → 層 → lower → 出口 の IN/OUT の行と，取る動作・置く動作の END の時刻を作る."""
    load, upper, proc, lower, out = ("U1", "P1"), ("U2", "P1"), ("U3", layer), ("U2", "P2"), ("U4", "P1")
    rows = [
        (base, load, IN), (base + 5, load, OUT),
        (base + 6, upper, IN), (base + 10, upper, OUT),
        (base + 11, proc, IN), (base + 50, proc, OUT),
        (base + 51, lower, IN), (base + 55 + out_lag, lower, OUT),
        (base + 56 + out_lag, out, IN), (base + 60 + out_lag, out, OUT),
    ]
    return [Transition(t, material, pos, event) for t, pos, event in rows]


def action_rows(base, lag=1):
    upper, lower = ("U2", "P1"), ("U2", "P2")
    spans = [(upper, base + 3, base + 5), (upper, base + 8, base + 10),
             (lower, base + 48, base + 50), (lower, base + 53, base + 55)]
    rows = []
    for pos, start, end in spans:
        rows.append(ActionRow(start + lag, pos, "START"))
        rows.append(ActionRow(end + lag, pos, "END"))
    return rows


def test_visits_pair_in_and_out_when_in_is_stamped_after_out():
    # 滞在が打刻ラグより短いと，IN の打刻が OUT より後になる
    pos = ("U4", "P1")
    logs = small_logs([
        Transition(100, "M1", ("U2", "P2"), OUT),
        Transition(101, "M1", pos, OUT),
        Transition(102, "M1", pos, IN),
    ])
    visits = build_visits(logs, {p.key: p.seq for p in logs.positions})["M1"]
    last = next(v for v in visits if v.pos == pos)
    assert (last.t_in, last.t_out) == (102, 101)
    assert [v.pos for v in visits].count(pos) == 1


def test_visit_without_out_comes_last_even_on_a_tie():
    logs = small_logs([
        Transition(200, "M1", ("U3", "P1"), IN),
        Transition(250, "M1", ("U3", "P1"), OUT),
        Transition(250, "M1", ("U2", "P2"), IN),
    ])
    visits = build_visits(logs, {p.key: p.seq for p in logs.positions})["M1"]
    assert [v.pos for v in visits] == [("U3", "P1"), ("U2", "P2")]


def test_unused_layer_comes_from_log_x():
    # ログXが真の構成なので，使われなかった層も数える．警告にはしない
    transitions = [row for i in range(1, 8) for row in visit_rows(f"M{i}", i * 100)]
    topology = build_topology(small_logs(transitions))
    process = next(s for s in topology.stages if s.kind == "process")
    assert process.positions == (("U3", "P1"), ("U3", "P2"))
    assert topology.unused_layers == [("U3", "P2")]
    assert not topology.warnings


def test_feed_and_outbound_hands_are_found_from_the_route():
    transitions = [row for i in range(1, 8) for row in visit_rows(f"M{i}", i * 100)]
    topology = build_topology(small_logs(transitions))
    process = next(s for s in topology.stages if s.kind == "process")
    assert process.feed.name == "upper"
    assert process.outbound.name == "lower"
    assert topology.stages[0].feed is None
    assert topology.stages[-1].outbound is None


def aligned(transitions, actions):
    logs = small_logs(transitions, actions)
    return align_actions(logs, build_topology(logs))


def test_alignment_needs_no_action_names_and_tracks_each_hand():
    bases = [i * 100 for i in range(1, 8)]
    alignment = aligned([r for i, b in enumerate(bases, 1) for r in visit_rows(f"M{i}", b)],
                        [r for b in bases for r in action_rows(b)])
    assert alignment.n_matched == alignment.n_events == 28
    actions = alignment.actions["R"]
    assert [a.kind for a in actions[:4]] == [PICK, PLACE, PICK, PLACE]
    assert [a.hand.name for a in actions[:4]] == ["upper", "upper", "lower", "lower"]
    assert {a.duration for a in actions} == {2}
    assert alignment.lags["R"].center_s == pytest.approx(1.0)


def test_alignment_recovers_after_a_missing_row():
    bases = [i * 100 for i in range(1, 8)]
    transitions = [r for i, b in enumerate(bases, 1) for r in visit_rows(f"M{i}", b)]
    actions = [r for b in bases for r in action_rows(b)]
    actions.remove(ActionRow(300 + 53 + 1, ("U2", "P2"), "START"))  # M3 の出口へ置く動作の START が欠ける
    alignment = aligned(transitions, actions)
    assert alignment.warnings
    assert alignment.n_matched == 27
    for a in alignment.actions["R"]:
        assert a.duration == 2
        assert 0 <= a.end - a.event_time <= 2


def test_alignment_recovers_when_counts_agree_by_chance():
    bases = [i * 100 for i in range(1, 12)]
    logs_intakes = [Intake(0, f"M{i}", "A", "C1") for i in range(1, 12)]
    transitions = [r for i, b in enumerate(bases, 1) for r in visit_rows(f"M{i}", b)]
    actions = [r for b in bases for r in action_rows(b)]
    upper = ("U2", "P1")
    # END が1つ，その後の START が1つ欠けると，件数は合うが途中がずれる
    actions.remove(ActionRow(100 + 5 + 1, upper, "END"))
    actions.remove(ActionRow(1100 + 8 + 1, upper, "START"))
    logs = dataclasses.replace(small_logs(transitions, actions), intakes=logs_intakes)
    alignment = align_actions(logs, build_topology(logs))
    assert all(a.duration == 2 for a in alignment.actions["R"])


@pytest.mark.parametrize("period", [137, 19])
def test_offset_of_long_alternating_rows(period):
    # 取る・置くが交互に並ぶ長い列．隣のイベントとの差や，隣の周期との差を選んではいけない
    events = [t for k in range(30000) for t in (k * period, k * period + 5)]
    starts = [t + 1 for t in events]
    ends = [t + 3 for t in events]
    assert _common_offset(events, ends) == 3
    assert _common_offset(starts, ends) == 2
    assert not _is_shifted(starts, ends, 2.5)


def test_offset_larger_than_half_of_the_event_spacing():
    events = [t for k in range(1000) for t in (k * 137, k * 137 + 5)]
    assert _common_offset(events, [t + 4 for t in events]) == 4


def test_a_long_month_is_aligned_without_resyncing(tmp_path_factory):
    # 行数が多いと現れる不具合の回帰テスト（ずれを求める区間の取り方）
    config = with_log(config_for(120.37, days=30), logc_kinds=("hand",))
    _, e = run_estimate(config, tmp_path_factory)
    assert e.alignment.n_matched == e.alignment.n_events
    assert not [w for w in e.warnings if "順に対応しません" in w]


def test_single_stage_holds_target_of_the_next_action():
    assert _hold_target({"upper": 1}, _action(stage=1)) == 1


def test_hold_target_ignores_an_action_for_another_material():
    # Lower が下流の空きを待っている間に，上流へ取りに行っても，待ち先は下流のまま
    assert _hold_target({"lower": 2}, _action(stage=0)) == 2


def _action(stage):
    from ftto_est.actions import Action
    from ftto_est.topology import Hand

    return Action(Hand("R", "upper", ("U", "P")), PICK, "M", ("U", "P"), stage, 0, 0, 0)


# --- 能力：低負荷のログから求める ---


def test_action_lengths_and_moves(low_load):
    _, e = low_load
    for value in e.motion.durations.values():
        assert value.value == pytest.approx(2.0, abs=0.2)
    assert e.motion.moves["R1"].table[(0, 1)].value == pytest.approx(3.0, abs=0.2)
    assert e.motion.moves["R1"].table[(1, 2)].value == pytest.approx(3.0, abs=0.2)


def test_return_move_is_only_an_upper_bound(low_load):
    _, e = low_load
    value = e.motion.moves["R1"].table[(2, 0)]
    assert value.quality == UPPER_ONLY
    assert value.value >= 5.0 - 0.2  # 真値 5 s


def test_robot_cycle_is_the_sum_of_the_exchange_steps(low_load):
    result, e = low_load
    for robot_id in ("R1", "R2"):
        value = e.capacity.by_robot[robot_id]
        assert value.quality == UPPER_ONLY
        assert result.capacity["robots"][robot_id]["W_s"] - 0.3 <= value.value <= 20.5


@pytest.mark.parametrize(("part", "pid"), TRUE_C)
def test_stage_capacity_at_low_load(low_load, part, pid):
    _, e = low_load
    assert stage_capacity(e, part, pid).value == pytest.approx(TRUE_C[(part, pid)], abs=0.3)


@pytest.mark.parametrize(("part", "pid"), TRUE_C)
def test_capacity_does_not_follow_the_input_interval(low_load, very_low_load, part, pid):
    a = stage_capacity(low_load[1], part, pid).value
    b = stage_capacity(very_low_load[1], part, pid).value
    assert a == pytest.approx(b, abs=0.3)


def test_line_limit_is_recovered_from_a_very_low_load_month(very_low_load):
    result, e = very_low_load
    limits = limit_by_process(e.topology, e.motion, e.capacity)
    for pid, truth in result.capacity["line"].items():
        assert limits[pid].value.value == pytest.approx(truth["T_min_s"], abs=0.3)
    assert limits["A"].part == PROC2
    assert limits["B"].part == PROC2
    assert limits["C"].part == PROC1


def test_unused_layers_share_the_stage_capacity(very_low_load):
    _, e = very_low_load
    # 投入間隔が長いと P8, P9 は使われないが，段の能力は3層で求まる
    assert ("U5", "P9") in e.topology.unused_layers
    assert stage_capacity(e, PROC2, "A").value == pytest.approx(154 / 3, abs=0.3)
    assert e.capacity.by_pos[(("U5", "P9"), "A")].value == pytest.approx(154.0, abs=1.0)


def test_final_stage_uses_log_b_dwell(low_load):
    _, e = low_load
    name = e.topology.stages[-1].name
    value = stage_capacity(e, name, "A")
    assert value.quality == ESTIMATED
    assert value.value == pytest.approx(7.0, abs=0.3)


def test_conveyors_are_upper_bounds(low_load):
    _, e = low_load
    for index, true_value in ((0, 7.0), (2, 14.0)):
        value = e.capacity.by_stage[(index, "A")]
        assert value.quality == UPPER_ONLY
        assert value.lo - 0.3 <= true_value <= value.hi + 0.3


def test_service_without_idle_samples_is_only_an_interval(tmp_path_factory):
    # 処理の完了がいつも次のマテリアルを取りに行く途中に重なる
    _, e = run_estimate(with_home(config_for(86.37), None, 0.0), tmp_path_factory)
    value = stage_capacity(e, PROC1, "C")
    assert value.quality == INTERVAL_ONLY
    assert value.lo <= 84.0 <= value.hi
    limit = limit_by_process(e.topology, e.motion, e.capacity)["C"]
    assert limit.value.quality == INTERVAL_ONLY
    assert limit.part == "未確定"
    assert set(limit.candidates) == {PROC1, PROC2}


# --- 余裕 ---


def test_margin_of_a_stage_uses_its_own_entry_interval(low_load):
    _, e = low_load
    rows = [m for m in e.margins if m.kind == "stage" and m.part == PROC1 and m.interval_s is not None]
    assert rows
    pid_of = {c.cassette_id: c.process_id for c in e.cassettes}
    for m in rows[1:-1]:
        pid = pid_of[m.cassette_id]
        assert m.interval_s == pytest.approx(137.37, abs=0.5)
        assert m.margin_s == pytest.approx(137.37 - TRUE_C[(PROC1, pid)], abs=0.8)
        assert m.rho == pytest.approx(TRUE_C[(PROC1, pid)] / 137.37, abs=0.02)


def test_robot_margin_is_idle_plus_process_wait(low_load):
    _, e = low_load
    rows = [m for m in e.margins if m.kind == "robot" and m.part == "R1" and m.margin_s is not None]
    for m in rows[1:-1]:
        assert m.margin_s == pytest.approx(m.wait_idle_s + m.wait_process_s, abs=1.0)
        assert m.interval_s - 20.5 <= m.margin_s <= m.interval_s - 18.0
        assert m.wait_hold_s == 0.0


def test_margin_uses_the_upper_bound_of_an_interval(tmp_path_factory):
    _, e = run_estimate(with_home(config_for(86.37), None, 0.0), tmp_path_factory)
    pid_of = {c.cassette_id: c.process_id for c in e.cassettes}
    row = next(m for m in e.margins[100:] if m.kind == "stage" and m.part == PROC1 and pid_of[m.cassette_id] == "C")
    assert row.margin_s == pytest.approx(row.interval_s - row.capacity.hi)


def test_hand_shares_robot_margin_and_reports_hold_time(low_load):
    _, e = low_load
    robot = next(m for m in e.margins if m.kind == "robot" and m.part == "R1" and m.margin_s is not None)
    hand = next(m for m in e.margins if m.kind == "hand" and m.part == "U2-P1" and m.cassette_id == robot.cassette_id)
    assert hand.margin_s == robot.margin_s
    assert hand.hold_min_s == pytest.approx(7.0, abs=0.3)
    assert hand.hold_excess_s < 1.0


# --- 律速の判定 ---


@pytest.mark.parametrize(("pid", "below", "above", "part"), [
    ("A", 51.07, 51.67, PROC2),
    ("B", 67.37, 68.37, PROC2),
    ("C", 83.37, 84.37, PROC1),
])
def test_judgement_switches_at_the_true_limit(tmp_path_factory, pid, below, above, part):
    def observed(interval):
        config = dataclasses.replace(
            sample_config(interval_s=interval, process_id_weights={pid: 1.0}), duration_s=2 * 86400.0)
        _, e = run_estimate(config, tmp_path_factory)
        return [j.observed for j in e.judgements[5:-1]]

    limited = observed(below)
    assert limited.count(part) >= 0.9 * len(limited)
    # 限界のごく近くでは，信号で先に動いたロボットの都合で一時的に保持待ちが出ることがある
    free = observed(above)
    assert free.count(None) >= 0.9 * len(free)


def test_carried_over_congestion_is_transient(tmp_path_factory):
    _, e = run_estimate(config_for(60.37, days=6), tmp_path_factory)
    previous = None
    transient = 0
    for j in e.judgements:
        # 処理ID A は限界 51.33 s より長い間隔なので，判定が出るのは前のカセットの詰まりだけ
        if j.cassette.process_id == "A" and j.observed is not None and previous.observed is not None:
            assert j.status == STATUS_TRANSIENT
            transient += 1
        previous = j
    assert transient > 0


def test_no_bottleneck_at_low_load(low_load):
    _, e = low_load
    assert {j.status for j in e.judgements} == {STATUS_NONE}
    assert not e.calibrations


SHORT_PROCESS = """
[simulation]
duration_s = 86400
[input]
interval_s = 15.37
cassette_size = 20
[input.process_id_weights]
A = 1.0
[log]
lag_max_s = 2.0
logc_lag_offset_s = 1.0
logc_lag_jitter_s = 0.5
[[units]]
id = "U1"
positions = [{ id = "P1", kind = "conveyor" }]
[[units]]
id = "U2"
positions = [{ id = "P1", kind = "hand" }, { id = "P2", kind = "hand" }]
[[units]]
id = "U3"
positions = [{ id = "P1", kind = "process" }]
[[units]]
id = "U4"
positions = [{ id = "P1", kind = "conveyor" }]
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
service = { time_s = 1.0 }
[[stages]]
name = "proc"
positions = [["U3", "P1"]]
service = { times_s = { A = 3.0 } }
feed = { robot = "R1", hand = "upper" }
[[stages]]
name = "out"
positions = [["U4", "P1"]]
service = { time_s = 1.0 }
feed = { robot = "R1", hand = "lower" }
"""


def test_robot_is_the_bottleneck_when_it_never_waits(tmp_path_factory):
    _, e = run_estimate(load_config_str(SHORT_PROCESS), tmp_path_factory)
    observed = {j.observed for j in e.judgements[1:-1]}
    assert observed == {"R1"}
    assert e.capacity.by_robot["R1"].value == pytest.approx(17.0, abs=0.3)


def test_robot_with_margin_left_is_not_the_bottleneck(tmp_path_factory):
    # 投入間隔 18.37 s は W = 17 s より長い．余裕が小さく待ちは測れないが，律速ではない
    config = load_config_str(SHORT_PROCESS.replace("interval_s = 15.37", "interval_s = 18.37"))
    _, e = run_estimate(config, tmp_path_factory)
    assert {j.observed for j in e.judgements[1:-1]} == {None}


@pytest.mark.parametrize(("robot", "interval", "expected"), [
    ("R1", 15.37, "R1"),  # R2 との干渉で入場間隔が W より長くなる
    ("R2", 15.37, "R2"),
    ("R1", 30.37, None),
    ("R2", 30.37, None),
])
def test_one_of_two_robots_is_the_bottleneck(tmp_path_factory, robot, interval, expected):
    config = sample_config(interval_s=interval, process_id_weights={"A": 1.0})
    stages = tuple(
        dataclasses.replace(s, service=dataclasses.replace(s.service, times_s={"A": 5.0}, accept_lead_s=0.0))
        if s.service.jitter_s > 0 else s
        for s in config.stages
    )
    robots = tuple(dataclasses.replace(r, pick_s=3.0) if r.robot_id == robot else r for r in config.robots)
    config = dataclasses.replace(config, stages=stages, robots=robots, duration_s=86400.0)
    _, e = run_estimate(config, tmp_path_factory)
    observed = [j.observed for j in e.judgements[3:-1]]
    assert observed.count(expected) >= 0.95 * len(observed)


def with_accept_lead(config, lead):
    stages = tuple(
        dataclasses.replace(s, service=dataclasses.replace(s.service, accept_lead_s=lead)) if s.service.jitter_s > 0
        else s
        for s in config.stages
    )
    return dataclasses.replace(config, stages=stages)


@pytest.mark.parametrize(("pid", "interval", "held"), [("A", 70.37, False), ("B", 200.37, True)])
def test_waiting_at_a_signalled_layer_is_not_a_bottleneck(tmp_path_factory, pid, interval, held):
    # もうすぐ終わる層の前で待っても，別の層が空いていれば層の不足ではない
    config = with_accept_lead(
        dataclasses.replace(sample_config(interval_s=interval, process_id_weights={pid: 1.0}), duration_s=3 * 86400.0),
        30.0,
    )
    result, e = run_estimate(config, tmp_path_factory)
    assert any(w["category"] == "hold" for w in result.recorder.robot_waits()) == held
    assert all(j.observed is None for j in e.judgements[1:-1])


def test_accept_signal_does_not_hide_a_real_bottleneck(tmp_path_factory):
    config = with_accept_lead(
        dataclasses.replace(sample_config(interval_s=45.37, process_id_weights={"A": 1.0}), duration_s=86400.0), 30.0)
    _, e = run_estimate(config, tmp_path_factory)
    observed = [j.observed for j in e.judgements[3:-1]]
    assert observed.count(PROC2) >= 0.9 * len(observed)


def test_accept_signal_does_not_change_the_capacity(tmp_path_factory):
    _, e = run_estimate(with_accept_lead(config_for(137.37), 30.0), tmp_path_factory)
    for (part, pid), truth in TRUE_C.items():
        assert stage_capacity(e, part, pid).value == pytest.approx(truth, abs=0.3)


def with_home(config, home, lead):
    robots = tuple(dataclasses.replace(r, home_move_s=home) for r in config.robots)
    return with_accept_lead(dataclasses.replace(config, robots=robots), lead)


def test_settings_file(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[accept_lead_s]\n"*" = 5.0\n"U5-P7" = 8.0\n[home_move_s]\n"U3-P1" = 2.0\n', encoding="utf-8")
    settings = load_settings(path)
    assert (settings.lead("U5-P7"), settings.lead("U3-P1")) == (8.0, 5.0)
    assert (settings.home("U3-P1"), settings.home("U5-P7")) == (2.0, None)


def test_late_signal_adds_a_delay_that_the_settings_remove(tmp_path_factory):
    # 原点から 3 s，信号は処理の完了と同時．着くのが 3 s 遅れる
    config = with_home(config_for(137.37), 3.0, 0.0)
    _, e = run_estimate(config, tmp_path_factory)
    assert stage_capacity(e, PROC1, "A").value == pytest.approx(44.0, abs=0.3)
    assert stage_capacity(e, PROC1, "A").quality == ESTIMATED


def test_without_settings_the_delay_makes_an_upper_bound(tmp_path_factory):
    config = with_home(config_for(137.37), 3.0, 0.0)
    log_dir = tmp_path_factory.mktemp("logs")
    simulate(config).write(log_dir)
    value = stage_capacity(estimate_dir(log_dir), PROC1, "A")
    assert value.quality == UPPER_ONLY
    assert value.value == pytest.approx(47.0, abs=0.3)
    assert value.lo <= 44.0 <= value.hi


def test_long_signal_needs_no_home_time(tmp_path_factory):
    # 信号がどの移動よりも長ければ，ロボットは必ず先に着くので，原点からの移動時間が分からなくても正確に求まる
    config = with_home(config_for(137.37), 3.0, 10.0)
    log_dir = tmp_path_factory.mktemp("logs")
    simulate(config).write(log_dir)
    value = stage_capacity(estimate_dir(log_dir, Settings({"*": 10.0}, {})), PROC1, "A")
    assert value.quality == ESTIMATED
    assert value.value == pytest.approx(44.0, abs=0.3)


def test_delay_is_measured_when_the_log_has_high_load(tmp_path_factory):
    # 高負荷のカセットでは層の前で完了を待つので，取り始めた時刻が処理の完了そのものになる．設定が無くても遅れが分かる
    config = with_home(
        sample_config(interval_s=120.37, interval_schedule=((100, 120.37), (100, 45.37)), process_id_weights=MIX),
        3.0, 0.0,
    )
    log_dir = tmp_path_factory.mktemp("logs")
    simulate(dataclasses.replace(config, duration_s=10 * 86400.0)).write(log_dir)
    e = estimate_dir(log_dir)
    assert e.service.delays[1].lo == pytest.approx(3.0, abs=0.3)
    for (part, pid), truth in TRUE_C.items():
        assert stage_capacity(e, part, pid).value == pytest.approx(truth, abs=0.3)


def test_home_trips_count_as_robot_margin(tmp_path_factory):
    # 原点への往復は，要求が来れば省ける時間なので余裕に含まれる．入場間隔 − W に一致する
    _, e = run_estimate(config_for(137.37), tmp_path_factory)
    rows = [m for m in e.margins if m.kind == "robot" and m.part == "R1" and m.margin_s is not None]
    for m in rows[1:-1]:
        assert m.margin_s == pytest.approx(m.interval_s - 19.0, abs=2.0)


def test_calibration_uses_only_steady_cassettes(tmp_path_factory):
    config = dataclasses.replace(
        sample_config(interval_s=45.37, process_id_weights={"A": 1.0}), duration_s=86400.0)
    _, e = run_estimate(config, tmp_path_factory)
    assert any(j.status == STATUS_STEADY for j in e.judgements)
    assert len(e.calibrations) == 1
    calibration = e.calibrations[0]
    assert (calibration.process_id, calibration.part) == ("A", PROC2)
    assert calibration.kappa == pytest.approx(1.0, abs=0.02)


def test_line_limit_keeps_parts_without_a_value():
    parts = {"a": Value.of(10.0), "b": Value.unobserved()}
    limit = line_limit(parts, _fake_topology(), _fake_motion())
    assert limit.value.value == 10.0
    assert limit.value.quality == LOWER_ONLY


def test_line_limit_lists_every_candidate_of_an_interval():
    parts = {"a": Value(86.0, 75.0, 86.0, INTERVAL_ONLY), "b": Value.of(81.0), "c": Value.of(20.0)}
    limit = line_limit(parts, _fake_topology(), _fake_motion())
    assert limit.candidates == ["a", "b"]
    assert (limit.value.lo, limit.value.hi) == (81.0, 86.0)
    assert limit.value.quality == INTERVAL_ONLY


def test_line_limit_uses_the_upper_bound_when_the_bottleneck_is_open():
    parts = {"a": Value(80.0, 70.0, 85.0, BIASED), "b": Value.of(82.0)}
    limit = line_limit(parts, _fake_topology(), _fake_motion())
    assert limit.value.value == 85.0


def _fake_topology():
    from ftto_est.topology import Topology

    return Topology({}, [], {}, {}, {})


def _fake_motion():
    from ftto_est.motion import Motion

    return Motion({}, {}, {}, {}, {})


# --- 頑健さ ---


@pytest.mark.parametrize("overrides", [
    {"logc_lag_offset_s": -2.0},
    {"logc_lag_offset_s": 3.0},
    {"logc_lag_jitter_s": 3.0},
    {"lag_max_s": 5.0},
    {"shuffle_within_second": True},
])
def test_log_lag_and_row_order_do_not_change_the_result(tmp_path_factory, overrides):
    _, e = run_estimate(with_log(config_for(137.37), **overrides), tmp_path_factory)
    assert e.alignment.n_matched == e.alignment.n_events
    assert stage_capacity(e, PROC1, "A").value == pytest.approx(44.0, abs=0.3)
    assert stage_capacity(e, PROC2, "B").value == pytest.approx(68.0, abs=0.3)


def test_missing_rows_lose_only_the_affected_actions(tmp_path_factory):
    _, e = run_estimate(with_log(config_for(137.37), logc_row_missing_ratio=0.001), tmp_path_factory)
    assert e.warnings
    assert e.alignment.n_matched / e.alignment.n_events > 0.99
    assert stage_capacity(e, PROC1, "A").value == pytest.approx(44.0, abs=0.3)
    assert stage_capacity(e, PROC2, "B").value == pytest.approx(68.0, abs=0.3)


def test_disturbed_actions_do_not_shift_the_centers(tmp_path_factory):
    config = config_for(137.37)
    robots = tuple(dataclasses.replace(r, disturbance_ratio=0.01, disturbance_s=10.0) for r in config.robots)
    _, e = run_estimate(dataclasses.replace(config, robots=robots), tmp_path_factory)
    assert e.alignment.n_matched == e.alignment.n_events
    for value in e.motion.durations.values():
        assert value.value == pytest.approx(2.0, abs=0.2)
    assert stage_capacity(e, PROC1, "A").value == pytest.approx(44.0, abs=0.5)


def test_biased_quality_comes_with_the_true_value_in_range(tmp_path_factory):
    _, e = run_estimate(config_for(120.37), tmp_path_factory)
    for (index, pid), value in e.capacity.by_stage.items():
        if value.quality == BIASED:
            truth = TRUE_C.get((e.topology.stages[index].name, pid))
            assert truth is None or value.lo - 0.3 <= truth <= value.hi + 0.3


# --- 出力 ---


def test_cli_writes_every_output(tmp_path, capsys):
    logs, out = tmp_path / "logs", tmp_path / "out"
    simulate(config_for(137.37, days=1)).write(logs)
    assert main(["--logs", str(logs), "--out", str(out)]) == 0
    assert "ライン限界の下限" in capsys.readouterr().out
    for name in ("capacity.csv", "service_time.csv", "motion.csv", "margin_cassette.csv", "margin_month.csv",
                 "bottleneck_cassette.csv", "capacity_trend.csv", "summary.json"):
        assert (out / name).stat().st_size > 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["line_limit_by_process"]["A"]["calibration"] == "未校正"
    assert summary["n_matched_events"] == summary["n_events"]


def test_cli_reports_a_missing_directory(tmp_path, capsys):
    assert main(["--logs", str(tmp_path / "none"), "--out", str(tmp_path / "out")]) == 2
    assert "失敗" in capsys.readouterr().err


def test_estimator_reads_only_the_logs(tmp_path):
    # 真値のファイルが無くても動く
    simulate(config_for(137.37, days=1)).write(tmp_path)
    for path in tmp_path.glob("truth_*"):
        path.unlink()
    assert load_logs(tmp_path).intakes
    assert estimate_dir(tmp_path).capacity.by_robot["R1"].value is not None
