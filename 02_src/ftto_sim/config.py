"""合成データ生成器の設定（TOML）の読み込みと検証."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOWER = "lower"  # 処理済みのマテリアルを下流へ運ぶハンド
HANDS = ("upper", LOWER)
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
    home_move_s: float | None = None  # 原点と各段の間の移動時間（既定）．None なら原点へ戻らず，その場で待つ
    home_overrides: dict[str, float] = field(default_factory=dict)  # 段名 -> 原点との移動時間
    jitter_s: float = 0.0  # 動作時間のばらつき（標準偏差）
    disturbance_ratio: float = 0.0  # 動作が外乱で伸びる割合
    disturbance_s: float = 0.0  # 外乱で伸びる時間

    def hand_key(self, hand: str) -> tuple[str, str]:
        return self.upper if hand == "upper" else self.lower

    def move_time(self, src: str, dst: str) -> float:
        """段 src から段 dst への移動時間."""
        if src == dst:
            return 0.0
        return self.move_overrides.get(frozenset((src, dst)), self.move_s)

    @property
    def has_home(self) -> bool:
        return self.home_move_s is not None

    def home_time(self, stage: str) -> float:
        """原点と段 stage の間の移動時間."""
        assert self.home_move_s is not None
        return self.home_overrides.get(stage, self.home_move_s)


@dataclass(frozen=True)
class ServiceConfig:
    # 処理ID -> 処理時間．DEFAULT_PROCESS_KEY は処理IDによらない既定値
    times_s: dict[str, float]
    jitter_s: float
    # 処理が終わるこの時間前から，受け入れ可能の信号を出す（ユーザー確認済み．値は未確認）
    accept_lead_s: float = 0.0

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
    # (カセット数, 取り込み間隔) の列．順に繰り返す．空なら interval_s を使う
    interval_schedule: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class LogConfig:
    lag_min_s: float
    lag_max_s: float
    logc_missing_ratio: float
    logc_kinds: tuple[str, ...]
    logc_moves: bool = False  # MOVE の行を出すか．実ログには記録されない
    logc_lag_offset_s: float = 0.0  # ログBに対するログCの打刻のずれ（一定分．負も可）
    logc_lag_jitter_s: float = 0.0  # ログCの打刻の行ごとのばらつき（0〜この値の一様分布）
    logc_row_missing_ratio: float = 0.0  # ログCの行そのものが欠ける割合
    shuffle_within_second: bool = False  # 同じ秒の中で行の順序を乱す


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
        interval_schedule=tuple((int(n), float(v)) for n, v in t.get("interval_schedule", [])),
    )


def _parse_log(t: dict[str, Any]) -> LogConfig:
    return LogConfig(
        lag_min_s=float(t.get("lag_min_s", 0.0)),
        lag_max_s=float(t.get("lag_max_s", 0.0)),
        logc_missing_ratio=float(t.get("logc_missing_ratio", 0.0)),
        logc_kinds=tuple(str(k) for k in t.get("logc_kinds", ["hand"])),
        logc_moves=bool(t.get("logc_moves", False)),
        logc_lag_offset_s=float(t.get("logc_lag_offset_s", 0.0)),
        logc_lag_jitter_s=float(t.get("logc_lag_jitter_s", 0.0)),
        logc_row_missing_ratio=float(t.get("logc_row_missing_ratio", 0.0)),
        shuffle_within_second=bool(t.get("shuffle_within_second", False)),
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
        home_move_s=float(t["home_move_s"]) if "home_move_s" in t else None,
        home_overrides={str(k): float(v) for k, v in t.get("home_moves", {}).items()},
        jitter_s=float(t.get("jitter_s", 0.0)),
        disturbance_ratio=float(t.get("disturbance_ratio", 0.0)),
        disturbance_s=float(t.get("disturbance_s", 0.0)),
    )


def _parse_service(t: dict[str, Any], where: str) -> ServiceConfig:
    if "time_s" in t:
        times = {DEFAULT_PROCESS_KEY: float(t["time_s"])}
    else:
        times = {str(k): float(v) for k, v in _require(t, "times_s", where).items()}
    return ServiceConfig(
        times_s=times, jitter_s=float(t.get("jitter_s", 0.0)), accept_lead_s=float(t.get("accept_lead_s", 0.0)),
    )


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
        if r.home_overrides and r.home_move_s is None:
            raise ConfigError(f"ロボット {r.robot_id} の home_moves には home_move_s も指定してください")
        if (r.home_move_s is not None and r.home_move_s < 0) or any(v < 0 for v in r.home_overrides.values()):
            raise ConfigError(f"ロボット {r.robot_id} の原点との移動時間は 0 以上にしてください")
        if r.jitter_s < 0 or r.disturbance_s < 0 or not 0 <= r.disturbance_ratio <= 1:
            raise ConfigError(f"ロボット {r.robot_id} の jitter_s / disturbance_s / disturbance_ratio が不正です")

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
        if s.service.accept_lead_s < 0:
            raise ConfigError(f"段 {s.name} の accept_lead_s は 0 以上にしてください")
        for pid in c.input.process_id_weights:
            if pid not in s.service.times_s and DEFAULT_PROCESS_KEY not in s.service.times_s:
                raise ConfigError(f"段 {s.name} に処理ID {pid} の処理時間がありません")

    inp = c.input
    if inp.interval_s <= 0 or inp.cassette_size <= 0:
        raise ConfigError("input.interval_s と input.cassette_size は正の値にしてください")
    if any(n <= 0 or v <= 0 for n, v in inp.interval_schedule):
        raise ConfigError("input.interval_schedule は [カセット数, 取り込み間隔] の正の値の列にしてください")
    if inp.process_id_mode not in ("per_cassette", "per_material"):
        raise ConfigError("input.process_id_mode は per_cassette / per_material のいずれかです")
    if not inp.process_id_weights or any(w < 0 for w in inp.process_id_weights.values()):
        raise ConfigError("input.process_id_weights には非負の重みを1つ以上指定してください")
    if not 0 <= c.log.lag_min_s <= c.log.lag_max_s:
        raise ConfigError("log の打刻ラグは 0 <= lag_min_s <= lag_max_s にしてください")
    if not 0 <= c.log.logc_missing_ratio <= 1:
        raise ConfigError("log.logc_missing_ratio は 0〜1 にしてください")
    if not 0 <= c.log.logc_row_missing_ratio <= 1:
        raise ConfigError("log.logc_row_missing_ratio は 0〜1 にしてください")
    if c.log.logc_lag_jitter_s < 0:
        raise ConfigError("log.logc_lag_jitter_s は 0 以上にしてください")
