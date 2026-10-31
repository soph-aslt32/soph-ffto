# 02_src

## 合成データ生成器（ftto_sim）

ディスプレイ製造装置の動作ログ（ログX・A・B・C）を模した合成データと，照合用の真値を生成する．
仕様と仮定は `../01_doc/design.md` を参照．

### 実行

```powershell
uv run python -m ftto_sim --config configs/sample.toml --out <出力先ディレクトリ>
```

- `--duration-s`：シミュレーション期間（秒）を上書きする（例：1か月分は `2592000`）
- `--seed`：乱数シードを上書きする

## 推定処理（ftto_est）

ログX・A・B・Cだけから，各部位の潜在能力，余裕，カセットごとの律速を求める．方針は `../01_doc/algorithm.md` を参照．

### 実行

```powershell
uv run python -m ftto_est --logs <ログのディレクトリ> --out <出力先ディレクトリ>
```

| 出力 | 内容 |
|---|---|
| `summary.json` | 処理IDごとのライン限界，律速部位，校正，ログ間のラグ，警告 |
| `capacity.csv` | 部位（Pos，段，ロボット，ハンド）× 処理ID の能力，区間，品質の印 |
| `service_time.csv` | 処理時間，手空きサンプルの採用率 |
| `motion.csv` | 動作の長さと移動時間 |
| `margin_month.csv` | 期間全体の部位ごとの余裕（平均と最小） |
| `margin_cassette.csv` | カセットごとの入場間隔，余裕，稼働率，待ちの内訳 |
| `bottleneck_cassette.csv` | カセットごとのライン限界，予測した律速，観測した律速，判定 |
| `capacity_trend.csv` | 動作の長さと処理時間の週ごとの推移 |

### 検証

設定値から合成ログを生成し，ログだけから推定した値を真値と照合する．

```powershell
uv run python tools/verify_estimate.py --days 30           # 全条件（--only でグループを指定）
uv run python tools/verify_estimate.py --limits            # 低負荷のログと，飽和実行の真の限界を比べる
```

## テスト

```powershell
uv run pytest
```
