# etf-daily

本地 ETF 日更命令：刷新基金数据、生成候选池、更新 regime 信号、从上市日画到当天的 HTML、HRP 树状图，以及 rules7 清单。

这些页面是研究输出，不会下单。

Qlib 基金数据需要单独准备，`pip` 不会下载。先按本机已有方式装好 Qlib，再安装本仓库。

## 安装

```bash
pip install -e ".[dev]"
cp config.env.example config.env
cp configs/paths.example.json configs/paths.json
```

`config.env` 填写 `PY`、`PLOTLY_ROOT`、`HRP_OUTPUT_DIR`、`TEMP_DIR`。`configs/paths.json` 填写数据路径，其中 `paths.qlib_scripts_dir` 给 `etl` 用。没有 `paths.json` 时仍会读旧文件名 `configs/production_regime_switch_ewma_shrink.json`。

regime 分片不要放进 git，用符号链接：

```bash
scripts/link_decision_packs.sh /path/to/decision_packs
```

## 每日顺序

```bash
etf-daily etl
etf-daily pool
etf-daily cluster-review
etf-daily regime --skip-rebuild --skip-html
etf-daily listing --as-of 2026-11-04
etf-daily hrp --as-of 2026-11-04
etf-daily hrp --dist-t 0.8 --as-of 2026-11-04
etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20261104_from_listing"
```

`listing` 默认增量，只补新的交易日。图 12 默认打开，`FIG12=0` 可关。

各命令的参数，以及可选的 `adaptive`、`triggers`、`b1235`、`morning` 见 [README.md](README.md)。图 7–图 11 的读法见 [docs/fair-path.md](docs/fair-path.md)。rules7 的规则见 [docs/rules7.md](docs/rules7.md)。

## 日中

交易时段跑。这时 Qlib 还没有今天这根 K 线。`--as-of` 默认今天，而且必须是本机当天，快照会拒绝其他日期。目录名用启动时刻，不用手写时间戳。

```bash
etf-daily regime --skip-html --intraday
etf-daily listing --intraday
etf-daily hrp --dist-t 0.8
etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20261105_from_listing"
```

按这个顺序。`listing` 和收盘一样读 `$PLOTLY_ROOT/_train_cache`，不用先跑 `adaptive --intraday`。

`regime --skip-html --intraday` 抓一份冻结的 AkShare 快照，把当天信号和反转标签写到 `runtime/decision_packs/intraday/{YYYYMMDD}/{HHMMSS}/`。`HHMMSS` 是命令启动时刻。正式分片 `runtime/decision_packs/20260720` 不动。不要加 `--skip-rebuild`，反转标签算在这份包里。`--skip-html` 只跳过 regime 的 HTML。

`listing --intraday` 用当天最新一份同时有 `signals_oos.csv` 和 `live_snapshot.csv` 的盘中包，把这根实时 bar 叠到上市日图上。HTML 在 `$PLOTLY_ROOT/{YYYYMMDD}_from_listing`。没有盘中包时会退出，并提示先跑上面的 `regime`。

`adaptive --intraday` 可选，出图到 `$PLOTLY_ROOT/{YYYYMMDD}all_adaptive`。它复用日期早于今天的最新一份 `all_adaptive` 配置，默认不重训，并另抓一份快照。`listing` 不读这个目录。

## 测试

```bash
ruff check src tests
pytest -q -m "not qlib"
bash tests/test_daily_env.sh
```

没有 Qlib 时，依赖本地数据的测试会跳过。
