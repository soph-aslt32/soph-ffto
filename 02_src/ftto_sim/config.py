"""合成データ生成器の設定（TOML）の読み込みと検証."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

HANDS = ("upper", "lower")
DEFAULT_PROCESS_KEY = "*"


class ConfigError(ValueError):
    """設定内容の不備."""


@dataclass(frozen=True)
class PositionConfig:
    unit_id: str
    pos_id: str
    kind: str  # "process" | "conveyor" | "hand" など


@dataclass(frozen=True)
class RobotConfig:
    robot_id: str
    upper: tuple[str, str]
    lower: tuple[str, str]
    move_s: float
    pick_s: float
    place_s: float
    # (段名, 段名) -> 移動時間．順不同で引く
    move_overrides: dict[frozenset[str], float]

    def hand_key(self, hand: str) -> tuple[str, str]:
        return self.upper if hand == "upper" else self.lower


@dataclass(frozen=True)
class ServiceConfig:
    # 処理ID -> 処理時間．DEFAULT_PROCESS_KEY は処理IDによらない既定値
    times_s: dict[str, float]
    jitter_s: float

    def base_time(self, process_id: str) -> float:
        if process_id in self.times_s:
            return self.times_s[process_id]
        return self.times_s[DEFAULT_PROCESS_KEY]


@dataclass(frozen=True)
class FeedConfig:
    """前段からこの段へマテリアルを運ぶ手段."""

    kind: str  # "robot" | "auto"
    robot_id: str | None = None
    hand: str | None = None


@dataclass(frozen=True)
class StageConfig:
    name: str
    positions: tuple[tuple[str, str], ...]
    service: ServiceConfig
    feed: FeedConfig | None  # 先頭段のみ None（装置への取り込み）


@dataclass(frozen=True)
class InputConfig:
    interval_s: float
    cassette_size: int
    cassette_gap_s: float
    process_id_mode: str  # "per_cassette" | "per_material"
    process_id_weights: dict[str, float]


@dataclass(frozen=True)
class LogConfig:
    lag_min_s: float
    lag_max_s: float
    logc_missing_ratio: float
    logc_kinds: tuple[str, ...]


@dataclass(frozen=True)
class PolicyConfig:
    dispatch: str
    exchange: str
    layer_select: str


@dataclass(frozen=True)
class SimConfig:
    seed: int
    start: datetime
    duration_s: float
    input: InputConfig
    log: LogConfig
    policies: PolicyConfig
    positions: tuple[PositionConfig, ...]  # 上流から順に並ぶ
    robots: tuple[RobotConfig, ...]
    stages: tuple[StageConfig, ...]  # 上流から順に並ぶ


def load_config(path: str | Path) -> SimConfig:
    # Windows のエディタで保存された BOM 付き UTF-8 も読めるようにする
    return load_config_str(Path(path).read_text(encoding="utf-8-sig"))


def load_config_str(text: str) -> SimConfig:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"TOML の構文エラー: {e}") from e
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> SimConfig:
    sim = raw.get("simulation", {})
    config = SimConfig(
        seed=int(sim.get("seed", 0)),
        start=datetime.fromisoformat(str(sim.get("start", "2026-01-01T00:00:00"))),
        duration_s=float(_require(sim, "duration_s", "simulation")),
        input=_parse_input(_require(raw, "input", "root")),
        log=_parse_log(raw.get("log", {})),
        policies=_parse_policies(raw.get("policies", {})),
        positions=_parse_positions(_require(raw, "units", "root")),
        robots=tuple(_parse_robot(r) for r in raw.get("robots", [])),
        stages=tuple(_parse_stage(s, i) for i, s in enumerate(_require(raw, "stages", "root"))),
    )
    _validate(config)
    return config


def _require(table: dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"[{where}] に {key} がありません")
    return table[key]


def _pos_key(value: Any, where: str) -> tuple[str, str]:
    if not (isinstance(value, list | tuple) and len(value) == 2):
        raise ConfigError(f"{where}: Pos は [UnitID, PosID] の形で指定してください: {value!r}")
    return (str(value[0]), str(value[1]))


def _parse_input(t: dict[str, Any]) -> InputConfig:
    weights = {str(k): float(v) for k, v in _require(t, "process_id_weights", "input").items()}
    return InputConfig(
        interval_s=float(_require(t, "interval_s", "input")),
        cassette_size=int(t.get("cassette_size", 20)),
        cassette_gap_s=float(t.get("cassette_gap_s", 0.0)),
        process_id_mode=str(t.get("process_id_mode", "per_cassette")),
        process_id_weights=weights,
    )


def _parse_log(t: dict[str, Any]) -> LogConfig:
    return LogConfig(
        lag_min_s=float(t.get("lag_min_s", 0.0)),
        lag_max_s=float(t.get("lag_max_s", 0.0)),
        logc_missing_ratio=float(t.get("logc_missing_ratio", 0.0)),
        logc_kinds=tuple(str(k) for k in t.get("logc_kinds", ["hand"])),
    )


def _parse_policies(t: dict[str, Any]) -> PolicyConfig:
    return PolicyConfig(
        dispatch=str(t.get("dispatch", "upstream_first")),
        exchange=str(t.get("exchange", "basic")),
        layer_select=str(t.get("layer_select", "lowest_index")),
    )


def _parse_positions(units: list[dict[str, Any]]) -> tuple[PositionConfig, ...]:
    result = []
    for u in units:
        unit_id = str(_require(u, "id", "units"))
        for p in _require(u, "positions", f"units.{unit_id}"):
            result.append(
                PositionConfig(
                    unit_id=unit_id,
                    pos_id=str(_require(p, "id", f"units.{unit_id}.positions")),
                    kind=str(_require(p, "kind", f"units.{unit_id}.positions")),
                )
            )
    return tuple(result)


def _parse_robot(t: dict[str, Any]) -> RobotConfig:
    robot_id = str(_require(t, "id", "robots"))
    overrides = {}
    for o in t.get("move_overrides", []):
        overrides[frozenset((str(o["a"]), str(o["b"])))] = float(o["s"])
    return RobotConfig(
        robot_id=robot_id,
        upper=_pos_key(_require(t, "upper", f"robots.{robot_id}"), f"robots.{robot_id}.upper"),
        lower=_pos_key(_require(t, "lower", f"robots.{robot_id}"), f"robots.{robot_id}.lower"),
        move_s=float(t.get("move_s", 0.0)),
        pick_s=float(t.get("pick_s", 0.0)),
        place_s=float(t.get("place_s", 0.0)),
        move_overrides=overrides,
    )


def _parse_service(t: dict[str, Any], where: str) -> ServiceConfig:
    if "time_s" in t:
        times = {DEFAULT_PROCESS_KEY: float(t["time_s"])}
    else:
        times = {str(k): float(v) for k, v in _require(t, "times_s", where).items()}
    return ServiceConfig(times_s=times, jitter_s=float(t.get("jitter_s", 0.0)))


def _parse_stage(t: dict[str, Any], index: int) -> StageConfig:
    name = str(_require(t, "name", f"stages[{index}]"))
    feed_raw = t.get("feed")
    if feed_raw is None:
        feed = None
    elif feed_raw == "auto":
        feed = FeedConfig(kind="auto")
    else:
        feed = FeedConfig(
            kind="robot",
            robot_id=str(_require(feed_raw, "robot", f"stages.{name}.feed")),
            hand=str(_require(feed_raw, "hand", f"stages.{name}.feed")),
        )
    return StageConfig(
        name=name,
        positions=tuple(_pos_key(p, f"stages.{name}") for p in _require(t, "positions", f"stages.{name}")),
        service=_parse_service(t.get("service", {"time_s": 0.0}), f"stages.{name}.service"),
        feed=feed,
    )


def _validate(c: SimConfig) -> None:
    kinds = {(p.unit_id, p.pos_id): p.kind for p in c.positions}
    if len(kinds) != len(c.positions):
        raise ConfigError("同じ (UnitID, PosID) が複数回定義されています")

    robots = {r.robot_id: r for r in c.robots}
    hand_keys: set[tuple[str, str]] = set()
    for r in c.robots:
        for hand in HANDS:
            key = r.hand_key(hand)
            if kinds.get(key) != "hand":
                raise ConfigError(f"ロボット {r.robot_id} の {hand} {key} は kind=hand の Pos ではありません")
            if key in hand_keys:
                raise ConfigError(f"ハンド {key} が複数のロボットに割り当てられています")
            hand_keys.add(key)

    if not c.stages:
        raise ConfigError("stages が空です")
    used: set[tuple[str, str]] = set()
    names: set[str] = set()
    for i, s in enumerate(c.stages):
        if s.name in names:
            raise ConfigError(f"段名 {s.name} が重複しています")
        names.add(s.name)
        if not s.positions:
            raise ConfigError(f"段 {s.name} に Pos がありません")
        for key in s.positions:
            if key not in kinds:
                raise ConfigError(f"段 {s.name} の Pos {key} が units に定義されていません")
            if kinds[key] == "hand":
                raise ConfigError(f"段 {s.name} にハンドの Pos {key} は置けません")
            if key in used:
                raise ConfigError(f"Pos {key} が複数の段に含まれています")
            used.add(key)
        if i == 0:
            if s.feed is not None:
                raise ConfigError("先頭段に feed は指定できません（装置への取り込み段）")
        elif s.feed is None:
            raise ConfigError(f"段 {s.name} に feed がありません")
        elif s.feed.kind == "robot":
            if s.feed.robot_id not in robots:
                raise ConfigError(f"段 {s.name} の feed のロボット {s.feed.robot_id} が未定義です")
            if s.feed.hand not in HANDS:
                raise ConfigError(f"段 {s.name} の feed の hand は upper/lower のいずれかです")
        for pid in c.input.process_id_weights:
            if pid not in s.service.times_s and DEFAULT_PROCESS_KEY not in s.service.times_s:
                raise ConfigError(f"段 {s.name} に処理ID {pid} の処理時間がありません")

    inp = c.input
    if inp.interval_s <= 0 or inp.cassette_size <= 0:
        raise ConfigError("input.interval_s と input.cassette_size は正の値にしてください")
    if inp.process_id_mode not in ("per_cassette", "per_material"):
        raise ConfigError("input.process_id_mode は per_cassette / per_material のいずれかです")
    if not inp.process_id_weights or any(w < 0 for w in inp.process_id_weights.values()):
        raise ConfigError("input.process_id_weights には非負の重みを1つ以上指定してください")
    if not 0 <= c.log.lag_min_s <= c.log.lag_max_s:
        raise ConfigError("log の打刻ラグは 0 <= lag_min_s <= lag_max_s にしてください")
    if not 0 <= c.log.logc_missing_ratio <= 1:
        raise ConfigError("log.logc_missing_ratio は 0〜1 にしてください")
