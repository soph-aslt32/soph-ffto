from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

from ftto_sim import SimConfig, SimResult, load_config, simulate

SAMPLE = Path(__file__).resolve().parent.parent / "configs" / "sample.toml"


def sample_config(**input_overrides: Any) -> SimConfig:
    """サンプル設定（6時間分）．input の項目を上書きできる."""
    config = load_config(SAMPLE)
    config = dataclasses.replace(config, duration_s=6 * 3600)
    if input_overrides:
        config = dataclasses.replace(config, input=dataclasses.replace(config.input, **input_overrides))
    return config


def with_log(config: SimConfig, **log_overrides: Any) -> SimConfig:
    return dataclasses.replace(config, log=dataclasses.replace(config.log, **log_overrides))


def with_policies(config: SimConfig, **policy_overrides: Any) -> SimConfig:
    return dataclasses.replace(config, policies=dataclasses.replace(config.policies, **policy_overrides))


def without_jitter(config: SimConfig) -> SimConfig:
    stages = tuple(
        dataclasses.replace(s, service=dataclasses.replace(s.service, jitter_s=0.0)) for s in config.stages
    )
    return dataclasses.replace(config, stages=stages)


def log_rows(rows: list[tuple[float, list[Any]]]) -> list[list[Any]]:
    """Recorder のログ行を，出力時と同じ順（打刻秒 → 記録順）に並べる."""
    ordered = sorted(enumerate(rows), key=lambda x: (x[1][1][0], x[0]))
    return [r for _, (_, r) in ordered]


def occupancy_by_material(result: SimResult) -> dict[str, list[dict[str, Any]]]:
    paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result.recorder.occupancy:
        paths[row["material_id"]].append(row)
    return paths


def layers_in_order(result: SimResult, unit_id: str, pos_ids: set[str]) -> list[str]:
    rows = [r for r in result.recorder.occupancy if r["unit_id"] == unit_id and r["pos_id"] in pos_ids]
    return [r["pos_id"] for r in sorted(rows, key=lambda r: r["t_in"])]


def run(config: SimConfig) -> SimResult:
    return simulate(config)
