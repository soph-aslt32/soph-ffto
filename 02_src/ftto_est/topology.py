"""ログX とログB から，段・ハンド・経路を組み立てる."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import zip_longest

from .logs import IN, Logs, PosInfo, PosKey


@dataclass(frozen=True, slots=True)
class Hand:
    robot_id: str
    name: str  # "upper" | "lower"
    pos: PosKey


@dataclass(slots=True)
class Visit:
    pos: PosKey
    t_in: int | None
    t_out: int | None


@dataclass
class Stage:
    index: int
    positions: tuple[PosKey, ...]
    kind: str
    feed: Hand | None = None  # この段へ運ぶハンド．先頭段と自動搬送は None
    outbound: Hand | None = None  # この段から運ぶハンド．最終段と自動搬送は None

    @property
    def name(self) -> str:
        return "+".join(pos_label(p) for p in self.positions)


@dataclass
class Topology:
    positions: dict[PosKey, PosInfo]
    stages: list[Stage]
    hands: dict[PosKey, Hand]
    stage_of: dict[PosKey, Stage]
    visits: dict[str, list[Visit]]
    warnings: list[str] = field(default_factory=list)
    unused_layers: list[PosKey] = field(default_factory=list)  # ログXにあるが，ログBに現れなかった層

    @property
    def robots(self) -> list[str]:
        return sorted({h.robot_id for h in self.hands.values()})


def pos_label(pos: PosKey) -> str:
    return f"{pos[0]}-{pos[1]}"


def build_visits(logs: Logs, seq: dict[PosKey, int]) -> dict[str, list[Visit]]:
    """マテリアルごとの Pos の列.

    IN と OUT は Pos ごとに対にする．同じ秒の行順に依存せず，滞在が打刻ラグより短くて IN が OUT より後でもよい．
    上流の OUT は下流の IN より前なので，並びは OUT の時刻で決める（IN にはラグが乗る）．
    """
    rows: dict[str, dict[PosKey, tuple[list[int], list[int]]]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for row in logs.transitions:
        rows[row.material_id][row.pos][row.event != IN].append(row.time)

    visits: dict[str, list[Visit]] = {}
    for material_id, by_pos in rows.items():
        path = [
            Visit(pos, t_in, t_out)
            for pos, (ins, outs) in by_pos.items()
            for t_in, t_out in zip_longest(sorted(ins), sorted(outs))
        ]
        # 出ていない Pos は，ログの終わりにいる場所なので最後に置く
        path.sort(key=lambda v: (v.t_out is None, v.t_out if v.t_out is not None else v.t_in, seq.get(v.pos, 0)))
        visits[material_id] = path
    return visits


def build_topology(logs: Logs) -> Topology:
    positions = {p.key: p for p in logs.positions}
    hands = {p.key: Hand(p.robot_id, p.hand, p.key) for p in logs.positions if p.robot_id}
    seq = {p.key: p.seq for p in logs.positions}
    visits = build_visits(logs, seq)
    warnings: list[str] = []

    # 段は，経路の同じ位置に現れる Pos の組．ログAにあるマテリアルは経路が先頭からそろっている
    known = {row.material_id for row in logs.intakes}
    routes = [[v.pos for v in visits[m] if v.pos not in hands] for m in known if m in visits]
    if not routes:
        raise ValueError("ログAとログBに共通のマテリアルがありません")
    length = max(len(r) for r in routes)
    members: list[Counter[PosKey]] = [Counter() for _ in range(length)]
    for route in routes:
        for i, pos in enumerate(route):
            members[i][pos] += 1

    stages = []
    stage_of: dict[PosKey, Stage] = {}
    for i, counter in enumerate(members):
        keys = tuple(sorted(counter, key=lambda p: seq.get(p, 0)))
        repeated = [p for p in keys if p in stage_of]
        if repeated:
            warnings.append(f"経路の複数の位置に現れる Pos があります（直列ではありません）: {repeated}")
        stage = Stage(i, keys, positions[keys[0]].kind if keys[0] in positions else "unknown")
        stages.append(stage)
        for pos in keys:
            stage_of.setdefault(pos, stage)

    # ログXが真の構成である．投入間隔が長いと使われない層はログBに現れないが，同じ Unit の同じ種類の Pos は並列の層
    unused_layers: list[PosKey] = []
    for info in logs.positions:
        if info.key in hands or info.key in stage_of:
            continue
        peers = [s for s in stages if s.kind == info.kind and any(p[0] == info.key[0] for p in s.positions)]
        if len(peers) != 1:
            warnings.append(f"{pos_label(info.key)} はログBに現れず，属する段が決まりません．推定から除きます")
            continue
        stage = peers[0]
        stage.positions = tuple(sorted((*stage.positions, info.key), key=lambda p: seq[p]))
        stage_of[info.key] = stage
        unused_layers.append(info.key)

    # 段の間を担うハンド
    feeds: list[Counter[PosKey]] = [Counter() for _ in stages]
    for material_id in known:
        path = visits.get(material_id, [])
        for cur, nxt in zip(path, path[1:]):
            if cur.pos in hands and nxt.pos in stage_of:
                feeds[stage_of[nxt.pos].index][cur.pos] += 1
    for stage, counter in zip(stages, feeds):
        if not counter:
            continue
        if len(counter) > 1:
            warnings.append(f"段 {stage.name} へ運ぶハンドが複数あります: {sorted(counter)}．最も多いものを使います")
        hand = hands[counter.most_common(1)[0][0]]
        stage.feed = hand
        if stage.index > 0:
            stages[stage.index - 1].outbound = hand

    return Topology(positions, stages, hands, stage_of, visits, warnings, unused_layers)
