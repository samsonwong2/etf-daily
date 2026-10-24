# etf-daily

Daily ETF commands: data refresh, candidate pool, cluster review, regime validation, adaptive HTML, from-listing HTML, HRP dendrogram, next-day triggers, and the B1235 checklist.

Qlib fund data is required and is not downloaded by `pip`. If `pip install` cannot find a package named `qlib`, install Qlib the way this machine already has it, then install this checkout.

## Install

```bash
pip install -e .
cp config.env.example config.env
cp configs/production_regime_switch_ewma_shrink.json.example configs/production_regime_switch_ewma_shrink.json
```

Edit the two copies. `config.env` needs `PY` and `PLOTLY_ROOT`. The json holds data paths, including `paths.qlib_scripts_dir` for `etl`. A missing file exits and names the example.

Point two symlinks at your regime shard directory and model cache. Do not copy those CSVs into git:

```bash
mkdir -p runtime/decision_packs
ln -sfn /path/to/20260720 runtime/decision_packs/20260720
ln -sfn /path/to/regime_transition_model_cache runtime/decision_packs/regime_transition_model_cache
```

## Daily order

Run these in order. `listing` reads frozen hold_up configs from `$PLOTLY_ROOT/_train_cache` and trains any pool name that cache does not have yet. It does not read `{YYYYMMDD}all_adaptive`.

Only `listing` writes the HTML under `$PLOTLY_ROOT/20260924_from_listing`. The date comes from `--as-of 2026-09-24`. The other commands write fund data, a candidate pool, cluster tables, a regime signal shard, an HRP page, or a trigger table.

```bash
etf-daily etl
etf-daily pool
etf-daily cluster-review
etf-daily regime --skip-rebuild --skip-html
etf-daily listing --as-of 2026-10-23
etf-daily hrp --as-of 2026-10-23
etf-daily hrp --dist-t 0.8 --as-of 2026-10-23
etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20261023_from_listing"
etf-daily triggers --listing-dir "$PLOTLY_ROOT/20260930_from_listing"
etf-daily b1235 --listing-dir "$PLOTLY_ROOT/20260930_from_listing"
etf-daily b1235-backtest --listing-dir "$PLOTLY_ROOT/20260930_from_listing"

```

`listing` is incremental unless the shell is run with `INCREMENTAL=0`. Figure 12 runs unless `FIG12=0`.

### `etf-daily etl`

Refreshes the fund list and qlib binary data from the sina ETL script. This needs network and `paths.qlib_scripts_dir` in the local json. This does not draw HTML. Prices written to CSV are unadjusted; `adaptive` applies post-hoc qfq when it loads qlib bars.

Args after `--` go to the underlying script.

```bash
etf-daily etl
etf-daily etl -- --full
etf-daily etl -- --as-of 2026-09-30 --skip-dump
```

- `etf-daily etl` — daily default. Skips funds whose local CSV max date is already ≥ today. Uses `dump_update` when qlib `calendars/day.txt` exists, otherwise `dump_all`. No per-fund sleep unless `--polite-delay`.
- `etf-daily etl -- --full` — re-fetch every fund, overwrite CSVs, always `dump_all`. Use when the store looks dirty or you want a full rebuild.
- `etf-daily etl -- --as-of 2026-09-30 --skip-dump` — treat a fund as fresh when its CSV max date ≥ `2026-09-30`, and skip `dump_bin` (CSV / fund-list only). Handy to verify skip behavior or refresh CSVs without touching qlib binaries.

### `etf-daily pool`

Builds the candidate pool directly. There is no CSI800/CSI1000 merge. It writes `cluster_mapping.csv`, `cluster_mapping_selected.csv`, and `cluster_mapping_selected.txt` under the local json `temp_dir`, not under `PLOTLY_ROOT`.

### `etf-daily cluster-review`

Audits `cluster_mapping_selected.csv` against the full `cluster_mapping.csv` from the same select run. `--future-end` defaults to today. This writes a review. It does not refresh the listing HTML.

### `etf-daily regime --skip-rebuild --skip-html`

Updates the regime-transition signal shard for the month of `--as-of`. With no `--as-of`, that date is today, not `2026-09-24`. `--skip-rebuild` skips the reversal-metric rebuild. `--skip-html` skips the regime HTML, so nothing is written to `$PLOTLY_ROOT/{YYYYMMDD}all`. The shard lives under `runtime/decision_packs/20260720`. `listing` reads it. Optional `adaptive` reads it too.

### `etf-daily adaptive --as-of 2026-09-24` (optional)

Not part of the daily order. Draws the adaptive-stage HTML for the cluster-selected names, ending 2026-09-24. HTML goes to `$PLOTLY_ROOT/20260924all_adaptive`. Training reuses `$PLOTLY_ROOT/_train_cache`. A ticket card is written into the same directory. `listing` does not read this directory. Run this when you want that HTML.

### `etf-daily listing --as-of 2026-09-24`

Draws one HTML per name from listing date through 2026-09-24. Frozen methods come from `$PLOTLY_ROOT/_train_cache`. A name missing from that cache is trained and added to it; names already in the cache keep their method. HTML goes to `$PLOTLY_ROOT/20260924_from_listing`. Incremental mode appends each missing session after the newest older `*_from_listing` directory under `PLOTLY_ROOT`. Figure 12 is added unless `FIG12=0`.

### `etf-daily hrp`

Builds the HRP dendrogram for the cluster pool. Lookback is 252 days. Distance stays at the production default. With no `--asof-date`, the end date is today. The page is `$HRP_OUTPUT_DIR/{YYYYMMDD}/hrp_dendrogram_{YYYYMMDD}.html`. Pass `--asof-date 2026-09-24` when the page should match that listing day.

### `etf-daily hrp --dist-t 0.8`

Same dendrogram with a coarser distance cut of `0.8`. The page is `$HRP_OUTPUT_DIR/{YYYYMMDD}/hrp_dendrogram_{YYYYMMDD}_d080.html`, so it does not replace the default page.

### `etf-daily triggers --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing"`

Scans the from-listing HTML and writes next-session trigger prices back into that same directory. `YYYYMMDD` in the line above is a placeholder. For the 2026-09-24 batch, replace it with `20260924`. When `--as-of` and `--next-day` are omitted, the date in the directory name is T, and T+1 is the next weekday. This skip does not know exchange holidays. Pass `--next-day` yourself when the next session is not the next weekday.

### `etf-daily b1235 --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing"`

Scores the four auxiliary buy conditions on that day's from-listing HTML and writes `b1235_checklist_{YYYYMMDD}.csv` back into the same directory. Run it after `listing`. `YYYYMMDD` is a placeholder. For the 2026-09-28 batch, the directory is `$PLOTLY_ROOT/20260928_from_listing`. When `--as-of` is omitted, the date in the directory name is T, the same way `triggers` reads it. Bond ETFs are skipped: their HTML has no fig7 candlestick, so the four conditions cannot be scored.

The close is the fig7 qfq candlestick in each HTML. The four flags match `evaluate_checklist_row` (deep discount −15%, reward/risk at least 2, stop = close − ATR20). They only change auxiliary-watch confidence. They do not change the production `hold_up` triangles.

- **B1** deep discount or lower-rail touch: fig8 gap ≤ −15%, or the close touches the fig9 or fig10 lower rail.
- **B2** long scale still intact: fig7 annualized slope > 0, or fig8 slope > 0, or the fig7/fig8 path rose over the last 20 days. Fig7 is ignored when the name has been listed for less than about 1.9 years.
- **B3** short scale has stopped falling: the close is back above the fig10 or fig11 path, or fig10 touched its lower rail yesterday and reclaimed it today.
- **B5** reward/risk ≥ 2: the target is the fig8 or fig10 path (a path below the close is not a target), and the stop is close − ATR20.

`B1235达成` is true only when all four pass. The CSV lists the raw reading for each condition and a yes/blank verdict. Names that pass all four sort first; the rest sort by fig8 discount, deepest first.

Two opt-in flags change the verdict; the defaults keep the rules above. `--b3-mode fig11_or_reclaim` drops the "back above the fig10 path" branch of B3. `--min-listing-years 1` also requires one year since listing.

### `etf-daily b1235-backtest --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing"`

Replays the B1235 scoring on every historical bar of every equity HTML in that directory and writes `b1235_ablation.{md,csv}`, `b1235_yearly.csv`, and `b1235_basket.csv` under `{listing-dir}/b1235_backtest`. Entry is the next bar's open. It reports forward 5/10/20/40-day returns, hit rate, excess over the same-date pool mean, 20-day max adverse excursion, and an ATR-stop / fig8-target exit simulation. It also reports an equal-weight basket that holds each signal for 20 days. Results are split in-sample / out-of-sample at `--split-date` (default 2022-01-01). The variants are leave-one-condition-out, single conditions, each branch of B1/B2/B3, and sweeps of deep gap, min R:R, stop ATR multiple, and B5 target. `--ohlcv-dir DIR` reads `CODE.csv` files (`date,open,high,low,close`) instead of HTML. The first run caches the panel in `panel_cache.pkl`; pass `--no-cache` after the data changes.

### `etf-daily rules7 --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing"`

Applies the buy/sell rules in `docs/7只ETF_买卖规律对比.md` to every equity HTML in that directory and writes `rules7_checklist_{YYYYMMDD}.csv` and `rules7_checklist_{YYYYMMDD}.md` back into it. Names containing 债 and the bond list are skipped. The 7 documented codes use their own rules. Every other code is auto-typed from its own history (平稳震荡 / 慢趋势 / 高波动主题 / 长期下跌或周期, or 历史不足 under one year) and uses that type's rules.

The output describes day T only. Each rule is split into its components: fig9 near-lower and its rising edge, the aux buy and sell legs, B1 discount, fast pattern, fig9 and fig8 near-upper, the mid-band stretch legs, the G1 legs, and the trailing stop. Each component gets its reading and a pass mark, and a summary column lists the key indicators that fire. Position state is replayed from the first bar, but only today's result is written: 持仓 / 空仓, peak close, trailing-stop price and distance, and whether G1 has fired in the open trade. Today's action is 买 / 卖 / 持有 / 观望, filled at the T+1 close. Rows are annotated from `decision_packs/{asof}/cluster_representatives_*_d080.csv` (HRP coarse clusters): 聚类 / 是否簇代表 / 推荐买入. Same-day buy signals in one cluster keep only one 推荐买入 (prefer the cluster representative). The md lists recommended buys, raw buys, sells, an HRP cluster table, open positions, near-trigger names, and thresholds. `--cluster-csv` overrides the pack path; `--rewrite-from-csv` re-annotates an existing checklist without rescanning HTML. `--jobs` (default 8) sets the worker count. Research only; it does not change production.

### `etf-daily morning --as-of YYYY-MM-DD --next-day YYYY-MM-DD`

Writes `morning.html` and `morning_receipt.json` into `$PLOTLY_ROOT/{YYYYMMDD}_from_listing`. The page is a receipt. `STOP` means a landed file is missing or the last bar is not `--as-of`, and the short list is hidden. `WARN` still shows the list: `go_nogo` pass is model calibration, and the reversal file is often older because the daily run skips rebuilding it. `scripts/accept_eod.sh` runs this after triggers. Re-run this command alone when the receipt failed and the earlier files are already on disk.

## Intraday

Run these during the session, while Qlib still has no bar for today. `--as-of` defaults to today and must be that machine date, because the AkShare snapshot rejects any other day. The clock names the pack directory. Do not pass a time stamp.

```bash
etf-daily regime --skip-html --intraday
etf-daily listing --intraday
etf-daily hrp --dist-t 0.8 --as-of 2026-10-23
etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20261023_from_listing"
```

Run them in that order. `listing` uses the same shared train cache as the end-of-day command. It does not need `adaptive --intraday`.

### `etf-daily regime --skip-html --intraday`

Fetches a frozen AkShare snapshot, then writes today's signal shard and reversal labels into `runtime/decision_packs/intraday/{YYYYMMDD}/{HHMMSS}/`. `HHMMSS` is the time the command starts. The formal shard under `runtime/decision_packs/20260720` stays as it was. `--skip-html` skips the regime HTML. Leave off `--skip-rebuild`: the reversal labels are part of this pack. `listing` later picks the newest directory under that day which contains both `signals_oos.csv` and `live_snapshot.csv`.

### `etf-daily adaptive --intraday` (optional)

Not required before `listing --intraday`. Draws the adaptive-stage HTML through today. It copies configs from the newest `$PLOTLY_ROOT/{YYYYMMDD}all_adaptive` whose date is before today, and it does not retrain. It takes a new AkShare snapshot for the chart. HTML goes to `$PLOTLY_ROOT/{YYYYMMDD}all_adaptive`.

### `etf-daily listing --intraday`

Draws one HTML per name from listing date through today, and appends the live bar from the regime pack above. HTML goes to `$PLOTLY_ROOT/{YYYYMMDD}_from_listing`. Incremental mode appends each missing session after the newest older `*_from_listing` directory. Figure 12 is added unless `FIG12=0`. If today's regime pack is missing, the command exits and tells you to run `etf-daily regime --skip-html --intraday` first.

## 图7–图11 路径

[docs/路径说明.md](docs/路径说明.md) 解释 `listing` HTML 里图7到图11的公平路径、因果 P95 轨和 EWMA 轨：窗口怎么取、两套轨分歧时怎么读、为什么只适合做过滤和仓位调节。正式买卖仍以主图 `hold_up` 为准。

## Tests

```bash
pip install -e .
python -m pytest tests/test_public_tree.py
bash tests/test_daily_env.sh
```

These checks do not call qlib and do not run the daily commands.
