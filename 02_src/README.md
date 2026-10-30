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

### テスト

```powershell
uv run pytest
```
