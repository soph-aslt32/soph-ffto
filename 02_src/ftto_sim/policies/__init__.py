"""差し替え可能なルール群．設定ファイルの [policies] で名前を指定して選ぶ."""

from __future__ import annotations

from ..config import ConfigError, PolicyConfig
from .dispatch import DispatchPolicy, UpstreamFirst
from .exchange import BasicExchange, ExchangeRule
from .layer_select import LayerSelectPolicy, LowestIndex

# 新しいルールは，クラスを実装してここに名前を登録すると設定ファイルから選べる
DISPATCH_POLICIES: dict[str, type] = {"upstream_first": UpstreamFirst}
EXCHANGE_RULES: dict[str, type] = {"basic": BasicExchange}
LAYER_SELECT_POLICIES: dict[str, type] = {"lowest_index": LowestIndex}


def _lookup(registry: dict[str, type], name: str, kind: str) -> type:
    if name not in registry:
        raise ConfigError(f"policies.{kind} = {name!r} は未定義です（候補: {', '.join(registry)}）")
    return registry[name]


def build_policies(config: PolicyConfig) -> tuple[DispatchPolicy, ExchangeRule, LayerSelectPolicy]:
    return (
        _lookup(DISPATCH_POLICIES, config.dispatch, "dispatch")(),
        _lookup(EXCHANGE_RULES, config.exchange, "exchange")(),
        _lookup(LAYER_SELECT_POLICIES, config.layer_select, "layer_select")(),
    )
