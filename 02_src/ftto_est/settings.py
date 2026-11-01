"""推定処理の設定（すべて省略可）.

実機ではエンジニアが受け入れ可能の信号のタイミングを調整するので，設定値として与える．ログには記録されない．

例（TOML）．キーは Pos の名前．"*" はその他の Pos の既定値．

    [accept_lead_s]        # 処理が終わる何秒前に受け入れ可能の信号を出すか
    "*" = 5.0
    "U5-P7" = 8.0

    [home_move_s]          # ロボットの原点から Pos の前までの移動時間
    "*" = 3.0
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KEY = "*"


@dataclass(frozen=True)
class Settings:
    accept_lead_s: dict[str, float] = field(default_factory=dict)
    home_move_s: dict[str, float] = field(default_factory=dict)

    def lead(self, pos_label: str) -> float:
        """信号の時間．未設定なら 0（処理完了で信号が出る）."""
        return self.accept_lead_s.get(pos_label, self.accept_lead_s.get(DEFAULT_KEY, 0.0))

    def home(self, pos_label: str) -> float | None:
        """原点からの移動時間．未設定なら None."""
        return self.home_move_s.get(pos_label, self.home_move_s.get(DEFAULT_KEY))


def load_settings(path: str | Path) -> Settings:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Settings(
        accept_lead_s={str(k): float(v) for k, v in raw.get("accept_lead_s", {}).items()},
        home_move_s={str(k): float(v) for k, v in raw.get("home_move_s", {}).items()},
    )
