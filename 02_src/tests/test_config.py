import dataclasses

import pytest

from ftto_sim import ConfigError, Simulator, load_config, load_config_str
from ftto_sim.config import DEFAULT_PROCESS_KEY, TASK_ORDER, _validate

from .helpers import SAMPLE, sample_config, with_policies

MINIMAL = """
[simulation]
duration_s = 100
[input]
interval_s = 10
[input.process_id_weights]
A = 1.0
[[units]]
id = "U1"
positions = [{ id = "P1", kind = "conveyor" }, { id = "P2", kind = "hand" }, { id = "P3", kind = "hand" }]
[[units]]
id = "U2"
positions = [{ id = "P1", kind = "process" }]
[[robots]]
id = "R1"
upper = ["U1", "P2"]
lower = ["U1", "P3"]
[[stages]]
name = "s0"
positions = [["U1", "P1"]]
[[stages]]
name = "s1"
positions = [["U2", "P1"]]
service = { times_s = { A = 5.0 } }
feed = { robot = "R1", hand = "upper" }
"""


def test_sample_config_loads():
    config = load_config(SAMPLE)
    assert [s.name for s in config.stages] == ["load", "proc1", "buffer", "proc2", "unload"]
    assert config.stages[3].positions == (("U5", "P7"), ("U5", "P8"), ("U5", "P9"))
    assert config.robots[0].hand_key("upper") == ("U2", "P1")
    assert config.stages[0].service.times_s == {DEFAULT_PROCESS_KEY: 5.0}
    assert config.stages[1].service.base_time("B") == 60.0


def test_minimal_config_loads():
    config = load_config_str(MINIMAL)
    assert config.input.cassette_gap_s == 0.0
    assert config.log.logc_kinds == ("hand",)
    assert config.policies.dispatch == "priority"


def test_bom_is_accepted(tmp_path):
    path = tmp_path / "bom.toml"
    path.write_text(MINIMAL, encoding="utf-8-sig")
    assert load_config(path).duration_s == 100


def test_syntax_error_is_config_error():
    with pytest.raises(ConfigError, match="構文エラー"):
        load_config_str("[simulation\n")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('positions = [["U2", "P1"]]', 'positions = [["U1", "P2"]]', "ハンドの Pos"),
        ('robot = "R1", hand = "upper"', 'robot = "R9", hand = "upper"', "未定義"),
        ("times_s = { A = 5.0 }", "times_s = { B = 5.0 }", "処理ID A"),
        ('upper = ["U1", "P2"]', 'upper = ["U1", "P1"]', "kind=hand"),
        ('interval_s = 10', 'interval_s = 0', "正の値"),
    ],
)
def test_invalid_config(old, new, message):
    assert old in MINIMAL
    with pytest.raises(ConfigError, match=message):
        load_config_str(MINIMAL.replace(old, new, 1))


def test_log_c_defaults_match_real_logs():
    config = load_config_str(MINIMAL)
    assert config.log.logc_moves is False
    assert config.log.logc_lag_offset_s == 0.0
    assert config.log.logc_lag_jitter_s == 0.0


def test_negative_jitter_is_rejected():
    text = MINIMAL.replace("[simulation]", "[log]\nlogc_lag_jitter_s = -1.0\n[simulation]", 1)
    with pytest.raises(ConfigError, match="logc_lag_jitter_s"):
        load_config_str(text)


def test_unknown_policy():
    with pytest.raises(ConfigError, match="policies.exchange"):
        Simulator(with_policies(sample_config(), exchange="unknown"))


def test_rejects_invalid_interval_schedule():
    config = load_config(SAMPLE)
    bad = dataclasses.replace(config, input=dataclasses.replace(config.input, interval_schedule=((0, 10.0),)))
    with pytest.raises(ConfigError):
        _validate(bad)


def test_rejects_invalid_robot_disturbance():
    config = load_config(SAMPLE)
    robots = tuple(dataclasses.replace(r, disturbance_ratio=1.5) for r in config.robots)
    with pytest.raises(ConfigError):
        _validate(dataclasses.replace(config, robots=robots))


def test_rejects_invalid_row_missing_ratio():
    config = load_config(SAMPLE)
    bad = dataclasses.replace(config, log=dataclasses.replace(config.log, logc_row_missing_ratio=-0.1))
    with pytest.raises(ConfigError):
        _validate(bad)


def test_rejects_home_moves_without_a_default():
    config = load_config(SAMPLE)
    robots = tuple(dataclasses.replace(r, home_move_s=None, home_overrides={"load": 2.0}) for r in config.robots)
    with pytest.raises(ConfigError):
        _validate(dataclasses.replace(config, robots=robots))


def test_home_time_can_be_set_per_stage():
    robot = dataclasses.replace(load_config(SAMPLE).robots[0], home_move_s=3.0, home_overrides={"proc1": 1.5})
    assert (robot.home_time("proc1"), robot.home_time("load")) == (1.5, 3.0)


@pytest.mark.parametrize("order", [
    TASK_ORDER[:3],  # 足りない
    (*TASK_ORDER[:3], TASK_ORDER[0]),  # 重複
    (*TASK_ORDER[:3], "fetch_middle"),  # 未定義
])
def test_rejects_invalid_task_order(order):
    config = load_config(SAMPLE)
    bad = dataclasses.replace(config, policies=dataclasses.replace(config.policies, task_order=order))
    with pytest.raises(ConfigError):
        _validate(bad)


def test_rejects_unknown_priority_setting():
    text = SAMPLE.read_text(encoding="utf-8").replace("commit_delivery = true", "commit_delivary = true")
    with pytest.raises(ConfigError):
        load_config_str(text)


def test_priority_is_read_from_the_settings():
    text = SAMPLE.read_text(encoding="utf-8").replace("commit_on_signal = true", "commit_on_signal = false")
    assert load_config_str(text).policies.commit_on_signal is False
    assert load_config(SAMPLE).policies.task_order == TASK_ORDER
