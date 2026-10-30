import pytest

from ftto_sim import ConfigError, Simulator, load_config, load_config_str
from ftto_sim.config import DEFAULT_PROCESS_KEY

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
    assert config.policies.dispatch == "upstream_first"


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


def test_unknown_policy():
    with pytest.raises(ConfigError, match="policies.exchange"):
        Simulator(with_policies(sample_config(), exchange="unknown"))
