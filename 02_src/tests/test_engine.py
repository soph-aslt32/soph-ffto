from collections import defaultdict

import pytest

from ftto_sim import DeadlockError, load_config_str

from .helpers import (
    layers_in_order,
    log_rows,
    occupancy_by_material,
    run,
    sample_config,
    with_log,
    without_jitter,
)

PROC2 = {"P7", "P8", "P9"}
EXPECTED_PATH = [
    ("U1", "P1"), ("U2", "P1"), ("U3", "P1"), ("U2", "P2"), ("U4", "P1"),
    ("U5", "P1"), ("U5", "proc2"), ("U5", "P2"), ("U6", "P1"),
]


@pytest.fixture(scope="module")
def normal():
    return run(sample_config())


@pytest.fixture(scope="module")
def congested():
    # 装置の能力（proc2 で約 61 s/枚）を超える間隔で取り込む
    return run(sample_config(interval_s=40.0))


def test_each_position_holds_one_material(normal, congested):
    for result in (normal, congested):
        by_pos = defaultdict(list)
        for row in result.recorder.occupancy:
            by_pos[(row["unit_id"], row["pos_id"])].append(row)
        for rows in by_pos.values():
            rows.sort(key=lambda r: r["t_in"])
            for prev, nxt in zip(rows, rows[1:]):
                assert prev["t_out"] is not None
                assert prev["t_out"] <= nxt["t_in"]


def test_material_path_follows_stages(normal):
    completed = 0
    for rows in occupancy_by_material(normal).values():
        if rows[-1]["t_out"] is None:
            continue  # 期間終了時に装置内に残っている
        path = [(r["unit_id"], "proc2" if r["pos_id"] in PROC2 else r["pos_id"]) for r in rows]
        assert path == EXPECTED_PATH
        assert all(a["t_out"] <= b["t_in"] for a, b in zip(rows, rows[1:]))
        completed += 1
    assert completed == normal.summary["n_completed"] > 0


def test_log_b_out_is_not_after_next_in(congested):
    events = defaultdict(list)
    for t, mid, unit, pos, ev in log_rows(congested.recorder.log_b):
        events[mid].append((t, (unit, pos), ev))
    for seq in events.values():
        assert [e[2] for e in seq[::2]] == ["IN"] * len(seq[::2])
        assert [e[2] for e in seq[1::2]] == ["OUT"] * len(seq[1::2])
        for (t_in, p_in, _), (t_out, p_out, _) in zip(seq[::2], seq[1::2]):
            assert p_in == p_out and t_in <= t_out
        for (t_out, _, _), (t_in, _, _) in zip(seq[1::2], seq[2::2]):
            assert t_out <= t_in <= t_out + 3  # ラグ 0〜2 s と秒への切り捨て


def test_log_a_time_is_actual_intake(congested):
    first_in = {}
    for t, mid, _, _, ev in log_rows(congested.recorder.log_b):
        if ev == "IN":
            first_in.setdefault(mid, t)
    rows = log_rows(congested.recorder.log_a)
    assert all(first_in[mid] == t for t, mid, _, _ in rows)
    times = [r[0] for r in rows]
    # 実際の取り込み時刻から，指定間隔（40 s）以上空く
    assert all(b - a >= 40 for a, b in zip(times, times[1:]))
    # 能力を超える取り込み要求なので，先頭の取り込みが待たされて平均間隔は指定より長くなる
    assert (times[-1] - times[0]) / (len(times) - 1) > 40.0 * 1.2


def test_intake_interval_without_congestion(normal):
    times = [r[0] for r in log_rows(normal.recorder.log_a)]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= 86 and max(gaps) <= 87  # 指定 86.4 s を秒に切り捨てた値


def test_process_id_per_cassette(normal):
    by_cassette = defaultdict(set)
    for _, _, pid, cassette in log_rows(normal.recorder.log_a):
        by_cassette[cassette].add(pid)
    assert all(len(pids) == 1 for pids in by_cassette.values())
    assert len(set().union(*by_cassette.values())) > 1


def test_process_id_per_material():
    result = run(sample_config(process_id_mode="per_material"))
    by_cassette = defaultdict(set)
    for _, _, pid, cassette in log_rows(result.recorder.log_a):
        by_cassette[cassette].add(pid)
    assert any(len(pids) > 1 for pids in by_cassette.values())


def test_service_time_is_constant_but_dwell_varies():
    result = run(without_jitter(sample_config(interval_s=40.0)))
    base = {"A": 150.0, "B": 200.0, "C": 240.0}
    dwell_extra = []
    for row in result.recorder.occupancy:
        if row["pos_id"] in PROC2 and row["t_out"] is not None:
            assert row["service_s"] == base[row["process_id"]]
            extra = row["t_out"] - row["t_in"] - row["service_s"]
            assert extra >= 0
            dwell_extra.append(extra)
    assert max(dwell_extra) > 0  # ロボット待ちなどで，滞在時間は処理時間より長くなりうる


def test_long_interval_uses_only_first_layer():
    result = run(sample_config(interval_s=600.0))
    assert set(layers_in_order(result, "U5", PROC2)) == {"P7"}


def test_short_interval_rotates_layers():
    # 処理IDが混ざると処理時間の差で順序が入れ替わるため，1種類に絞って確認する
    result = run(sample_config(interval_s=40.0, process_id_weights={"A": 1.0}))
    order = layers_in_order(result, "U5", PROC2)
    rotation = {"P7": "P8", "P8": "P9", "P9": "P7"}
    follows = sum(rotation[a] == b for a, b in zip(order, order[1:]))
    assert follows / (len(order) - 1) > 0.95


def _exchanges(result):
    """同じロボットで「Lower で取る → Upper で同じ Pos に置く」が連続した回数."""
    count = 0
    by_robot = defaultdict(list)
    for op in result.recorder.robot_ops:
        by_robot[op["robot_id"]].append(op)
    for ops in by_robot.values():
        for a, b in zip(ops, ops[1:]):
            if (a["hand"], a["action"], b["hand"], b["action"]) == ("lower", "PICK", "upper", "PLACE") \
                    and a["target"] == b["target"]:
                count += 1
    return count


def test_basic_exchange_happens_under_load(congested):
    assert _exchanges(congested) > 100


def test_robot_does_one_operation_at_a_time(congested):
    by_robot = defaultdict(list)
    for op in congested.recorder.robot_ops:
        by_robot[op["robot_id"]].append(op)
    for ops in by_robot.values():
        for a, b in zip(ops, ops[1:]):
            assert a["t_end"] <= b["t_start"]


def test_log_c_start_end_alternate(congested):
    by_pos = defaultdict(list)
    for row in log_rows(congested.recorder.log_c):
        by_pos[(row[1], row[2])].append(row)
    assert set(by_pos) == {("U2", "P1"), ("U2", "P2"), ("U5", "P1"), ("U5", "P2")}  # 既定はハンドのみ
    for rows in by_pos.values():
        assert [r[3] for r in rows] == ["START", "END"] * (len(rows) // 2)


def test_log_c_missing_ratio(congested):
    rows = log_rows(congested.recorder.log_c)
    missing = sum(r[4] == "" for r in rows) / len(rows)
    assert 0.4 < missing < 0.6  # サンプル設定は 0.5


def test_log_c_can_include_process_positions():
    result = run(with_log(sample_config(), logc_kinds=("hand", "process"), logc_missing_ratio=0.0))
    actions = {(r[1], r[2], r[4]) for r in log_rows(result.recorder.log_c)}
    assert ("U5", "P7", "PROCESS") in actions
    assert ("U3", "P1", "PROCESS") in actions


def test_same_seed_is_deterministic():
    a, b = run(sample_config()), run(sample_config())
    assert log_rows(a.recorder.log_b) == log_rows(b.recorder.log_b)
    assert log_rows(a.recorder.log_c) == log_rows(b.recorder.log_c)


def test_log_settings_do_not_change_dynamics(normal):
    other = run(with_log(sample_config(), lag_max_s=0.0, logc_missing_ratio=0.0))
    assert other.recorder.occupancy == normal.recorder.occupancy


def test_zero_lag_stamps_out_and_in_at_same_second():
    result = run(with_log(sample_config(), lag_max_s=0.0))
    events = defaultdict(list)
    for t, mid, _, _, ev in log_rows(result.recorder.log_b):
        events[mid].append(t)
    for times in events.values():
        assert all(a == b for a, b in zip(times[1::2], times[2::2]))


DEADLOCK = """
[simulation]
duration_s = 3600
[input]
interval_s = 10
[input.process_id_weights]
A = 1.0
[[units]]
id = "U1"
positions = [
  { id = "P1", kind = "conveyor" }, { id = "P2", kind = "hand" }, { id = "P3", kind = "hand" },
  { id = "P4", kind = "process" }, { id = "P5", kind = "process" }, { id = "P6", kind = "conveyor" },
]
[[robots]]
id = "R1"
upper = ["U1", "P2"]
lower = ["U1", "P3"]
move_s = 1.0
pick_s = 1.0
place_s = 1.0
[[stages]]
name = "s0"
positions = [["U1", "P1"]]
[[stages]]
name = "s1"
positions = [["U1", "P4"]]
service = { time_s = 30.0 }
feed = { robot = "R1", hand = "upper" }
[[stages]]
name = "s2"
positions = [["U1", "P5"]]
service = { time_s = 5.0 }
feed = { robot = "R1", hand = "lower" }
[[stages]]
name = "s3"
positions = [["U1", "P6"]]
feed = { robot = "R1", hand = "upper" }
"""


def test_deadlock_is_detected():
    # 1台のロボットが Upper で s0→s1 と s2→s3 を担うため，入れ替え後に s2 の空きを待ったまま止まる
    with pytest.raises(DeadlockError, match="デッドロック"):
        run(load_config_str(DEADLOCK))
