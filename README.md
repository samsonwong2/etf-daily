# etf-daily

日终八条命令。需要本机已有的 qlib 基金数据。Python 3.12。

```bash
pip install qlib pandas numpy plotly scipy skfolio
cp config.env.example config.env
cp configs/production_regime_switch_ewma_shrink.json.example configs/production_regime_switch_ewma_shrink.json
```

改两份副本。`config.env` 的 `PY` 指向解释器，`PLOTLY_ROOT` 指向 HTML 目录，`TEMP_DIR` 与 json 里的 `paths.temp_dir` 相同。json 的 `paths.provider_uri` 指向 qlib 基金数据，`paths.qlib_scripts_dir` 指向 qlib 的 `scripts`（`run.py etl` 需要它）。公开缺省是 `~/.qlib/qlib_data/all_fund_data` 和 `~/etf-daily-output`。

续跑分片用链接，不提交：

```bash
OLD_DECISION_PACK_ROOT=/path/to/decision_packs ./scripts/link_decision_packs.sh
```

脚本只链接 `20260720` 和 `regime_transition_model_cache`。

## 日终

在仓库根目录。四支 shell 自己读取 `config.env`。下面前三条直接调用 `$PY`，先导出同一份配置：

```bash
set -a
source config.env
set +a
AS_OF=YYYY-MM-DD
NEXT_DAY=YYYY-MM-DD
AS_OF_TAG=YYYYMMDD
```

```bash
"$PY" run.py etl
"$PY" run.py pool
"$PY" fund_pool_builder/每日审查cluster_mapping_最短命令清单.py \
  --future-end "$AS_OF" \
  --selected-csv "$TEMP_DIR/cluster_mapping_selected.csv" \
  --mapping-csv "$TEMP_DIR/cluster_mapping.csv"
AS_OF=$AS_OF JOBS=8 PY=$PY SKIP_REBUILD=1 SKIP_HTML=1 \
  ./scripts/daily_regime_transition_validation.sh
AS_OF=$AS_OF JOBS=8 PY=$PY \
  ./scripts/daily_adaptive_stage_html.sh
AS_OF=$AS_OF JOBS=8 PY=$PY \
  ./scripts/daily_adaptive_from_listing.sh
"$PY" workspace/scripts/generate_hrp_dendrogram_html.py \
  --lookback-days 252 \
  --asof-date "$AS_OF" \
  --dist-t 0.8 \
  --output "$HRP_OUTPUT_DIR/$AS_OF_TAG/hrp_dendrogram_${AS_OF_TAG}_d080.html"
PYTHONPATH=. "$PY" decision_pack/scripts/scan_next_day_trigger_prices.py \
  --listing-dir "$PLOTLY_ROOT/${AS_OF_TAG}_from_listing" \
  --as-of "$AS_OF" \
  --next-day "$NEXT_DAY" \
  --jobs 8
```

`from_listing` 默认增量，不要加 `INCREMENTAL=0`。`HRP_MEMBERSHIP_CSV` 为空时，adaptive 脚本不传 `--hrp-membership-csv`。

同一八条也可以交给验收脚本。两个日期都必填：

```bash
./scripts/accept_eod.sh --as-of YYYY-MM-DD --next-day YYYY-MM-DD
```

缺 `config.env` 或缺本地 json 时，脚本退出并打印上面两条 `cp`，不会启动 Python。退出码为 0 之前，不要把这个目录当成日终入口。

盘中三条和池更新整月重跑不在这八条里。
