"""シミュレーション中の装置構成（Pos・段・ロボット）とマテリアルの状態."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import HANDS, RobotConfig, ServiceConfig, SimConfig

if TYPE_CHECKING:
    from .ops import Op


@dataclass(eq=False)
class Material:
    material_id: str
    process_id: str
    cassette_id: str
    stage_index: int = -1  # 最後に置かれた段．ハンドで保持中は搬送元の段
    position: Position | None = None


@dataclass(eq=False)
class Position:
    unit_id: str
    pos_id: str
    kind: str
    stage: Stage | None = None  # ハンドは None
    material: Material | None = None
    finished: bool = False  # 段での処理（コンベアは搬送）が終わっているか
    finished_at: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.unit_id, self.pos_id)

    @property
    def label(self) -> str:
        return f"{self.unit_id}-{self.pos_id}"

    @property
    def is_empty(self) -> bool:
        return self.material is None


@dataclass(eq=False)
class Feed:
    """段へマテリアルを運ぶ手段．robot が None なら自動搬送（またはラインからの払い出し）."""

    robot: Robot | None
    hand: str | None


@dataclass(eq=False)
class Stage:
    index: int
    name: str
    positions: list[Position]
    service: ServiceConfig
    feed: Feed | None  # 先頭段は None
    outbound: Feed | None = None  # 次段の feed．最終段は None（自動で払い出し）


@dataclass(eq=False)
class Robot:
    config: RobotConfig
    hands: dict[str, Position]
    location: int = 0  # ロボットがいる段の index
    busy: bool = False
    plan: deque[Op] = field(default_factory=deque)

    @property
    def robot_id(self) -> str:
        return self.config.robot_id

    def other_hand(self, hand: str) -> str:
        return "lower" if hand == "upper" else "upper"

    def move_time(self, stages: list[Stage], src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        pair = frozenset((stages[src].name, stages[dst].name))
        return self.config.move_overrides.get(pair, self.config.move_s)


@dataclass
class Topology:
    positions: list[Position]  # 上流から順
    stages: list[Stage]
    robots: list[Robot]

    @classmethod
    def build(cls, config: SimConfig) -> Topology:
        positions = [Position(p.unit_id, p.pos_id, p.kind) for p in config.positions]
        by_key = {p.key: p for p in positions}

        robots = []
        for rc in config.robots:
            robots.append(Robot(config=rc, hands={h: by_key[rc.hand_key(h)] for h in HANDS}))
        robot_by_id = {r.robot_id: r for r in robots}

        stages: list[Stage] = []
        for i, sc in enumerate(config.stages):
            feed = None
            if sc.feed is not None:
                robot = robot_by_id[sc.feed.robot_id] if sc.feed.kind == "robot" else None
                feed = Feed(robot=robot, hand=sc.feed.hand)
            stage = Stage(
                index=i,
                name=sc.name,
                positions=[by_key[k] for k in sc.positions],
                service=sc.service,
                feed=feed,
            )
            for p in stage.positions:
                p.stage = stage
            stages.append(stage)
        for prev, nxt in zip(stages, stages[1:]):
            prev.outbound = nxt.feed

        # ロボットの初期位置は，そのロボットが取り出しを担う最も上流の段
        for r in robots:
            sources = [s.index for s in stages if s.outbound and s.outbound.robot is r]
            r.location = min(sources) if sources else 0
        return cls(positions=positions, stages=stages, robots=robots)
