# etf-daily

Daily commands for a local Chinese ETF universe: refresh fund data, build a candidate pool, update the regime-transition signal, draw one HTML page per fund from its listing date, and write an HRP dendrogram plus a rules checklist.

The pages are research output. They do not send orders.

Qlib fund data is required and is not downloaded by `pip`. Install Qlib the way you already do on this machine, then install this checkout. `pyqlib` is an optional extra for environments where that wheel installs cleanly.

## Install

```bash
pip install -e ".[dev]"
cp config.env.example config.env
cp configs/paths.example.json configs/paths.json
```

Edit the two copies. `config.env` sets `PY`, `PLOTLY_ROOT`, `HRP_OUTPUT_DIR`, and `TEMP_DIR`. `configs/paths.json` holds data paths, including `paths.qlib_scripts_dir` for `etl`. A missing file exits and names the example.

`configs/production_regime_switch_ewma_shrink.json` is still read when `configs/paths.json` is absent.

Point two symlinks at your regime shard directory and model cache. Do not copy those CSVs into git:

```bash
scripts/link_decision_packs.sh /path/to/decision_packs
```

That links `runtime/decision_packs/20260720` and `runtime/decision_packs/regime_transition_model_cache`.

## Daily order

Run these in order. `listing` reads frozen hold-up configs from `$PLOTLY_ROOT/_train_cache` and trains any pool name that cache does not have yet. It does not read `{YYYYMMDD}all_adaptive`.

Only `listing` writes the HTML under `$PLOTLY_ROOT/20261026_from_listing`. The date comes from `--as-of`. The other commands write fund data, a candidate pool, cluster tables, a regime signal shard, an HRP page, or a checklist.

```bash
etf-daily etl
etf-daily pool
etf-daily cluster-review
etf-daily regime --skip-rebuild --skip-html
etf-daily listing --as-of 2026-10-26
etf-daily hrp --as-of 2026-10-26
etf-daily hrp --dist-t 0.8 --as-of 2026-10-26
etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20261026_from_listing"
etf-daily forward --listing-dir "$PLOTLY_ROOT/20261026_from_listing"
etf-daily forward-page --listing-dir "$PLOTLY_ROOT/20261026_from_listing"
```

`listing` is incremental unless the shell is run with `INCREMENTAL=0`. Figure 12 runs unless `FIG12=0`.

`scripts/accept_eod.sh --as-of YYYY-MM-DD --next-day YYYY-MM-DD` runs etl, pool, cluster-review for that date, regime, listing, `hrp --dist-t 0.8`, triggers, and the morning receipt. It exits first if `config.env` or a paths json is missing.

### `etf-daily etl`

Refreshes the fund list and Qlib binary data. This needs network access and `paths.qlib_scripts_dir`. Prices written to CSV are unadjusted. Later commands apply a forward adjustment when they load Qlib bars.

```bash
etf-daily etl
etf-daily etl -- --full
etf-daily etl -- --as-of 2026-10-26 --skip-dump
```

- `etf-daily etl` skips funds whose local CSV is already at least today. It uses `dump_update` when `calendars/day.txt` exists, otherwise `dump_all`.
- `etf-daily etl -- --full` re-fetches every fund and always runs `dump_all`.
- `--skip-dump` writes CSVs and the fund list only.

### `etf-daily pool`

Builds the candidate pool and writes `cluster_mapping.csv`, `cluster_mapping_selected.csv`, and `cluster_mapping_selected.txt` under `paths.temp_dir`. See [docs/pool.md](docs/pool.md).

### `etf-daily cluster-review`

Audits `cluster_mapping_selected.csv` against the full `cluster_mapping.csv` from the same select run. `--future-end` defaults to today.

### `etf-daily regime --skip-rebuild --skip-html`

Updates the regime-transition signal shard for the month of `--as-of` (today when omitted). `--skip-rebuild` skips the reversal-metric rebuild. `--skip-html` skips the regime HTML. The shard lives under `runtime/decision_packs/20260720`. `listing` reads it.

### `etf-daily listing --as-of 2026-10-26`

Draws one HTML per name from its listing date through that day. Frozen methods come from `$PLOTLY_ROOT/_train_cache`. A missing name is trained and added. HTML goes to `$PLOTLY_ROOT/20261026_from_listing`. Incremental mode appends each missing session after the newest older `*_from_listing` directory. Figure 12 is added unless `FIG12=0`.

How to read figures 7–11: [docs/fair-path.md](docs/fair-path.md).

### `etf-daily hrp`

Builds the HRP dendrogram for the cluster pool. Lookback is 252 days. The page is `$HRP_OUTPUT_DIR/{YYYYMMDD}/hrp_dendrogram_{YYYYMMDD}.html`.

### `etf-daily hrp --dist-t 0.8`

Same dendrogram with a coarser distance cut. The page is `$HRP_OUTPUT_DIR/{YYYYMMDD}/hrp_dendrogram_{YYYYMMDD}_d080.html`, so it does not replace the default page.

### `etf-daily rules7 --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing"`

Applies the buy/sell rules in [docs/rules7.md](docs/rules7.md) to every equity HTML in that directory and writes `rules7_checklist_{YYYYMMDD}.csv` and `.md` back into it. Bond names are skipped. The 7 documented codes use their own rules. Every other code is typed from its own history.

The output describes day T only. Rows are annotated from the HRP coarse clusters for that day: one recommended buy per cluster. Research only. It does not change production `hold_up`.

### `etf-daily forward` and `etf-daily forward-page`

`forward` replays the frozen rules7 spec on the listing HTML and writes `$PLOTLY_ROOT/rules7_forward/` (a trade ledger with the next-session fill, a one-day strip, and a pass/fail report). `forward-page` writes a separate HTML per name to `$PLOTLY_ROOT/{YYYYMMDD}_forward/`. The default set is the 7 documented ETFs. `--all` or `--codes SH512530,SH518880` selects others. Neither command rewrites `*_from_listing`. Failure tests and the freeze date are in [docs/rules7.md](docs/rules7.md).

## Optional commands

### `etf-daily adaptive --as-of YYYY-MM-DD`

Not part of the daily order. Draws the adaptive-stage HTML for the cluster-selected names into `$PLOTLY_ROOT/{YYYYMMDD}all_adaptive`. `listing` does not read that directory.

### `etf-daily triggers --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing"`

Writes next-session trigger prices into that directory. When `--as-of` and `--next-day` are omitted, the date in the directory name is T and T+1 is the next weekday. Pass `--next-day` when the next session is not the next weekday.

### `etf-daily b1235` and `etf-daily b1235-backtest`

Scores four auxiliary buy conditions on that day's from-listing HTML, or replays them on every historical bar. Notes: [docs/b1235.md](docs/b1235.md). Bond ETFs are skipped. The flags do not change production `hold_up`.

### `etf-daily morning --as-of YYYY-MM-DD --next-day YYYY-MM-DD`

Writes `morning.html` and `morning_receipt.json` into `$PLOTLY_ROOT/{YYYYMMDD}_from_listing`. `STOP` means a landed file is missing or the last bar is not `--as-of`. `WARN` still shows the list.

## Intraday

Run these during the session, while Qlib still has no bar for today. `--as-of` defaults to today and must be that machine date, because the snapshot rejects any other day. The clock names the pack directory.

```bash
etf-daily regime --skip-html --intraday
etf-daily listing --intraday
etf-daily hrp --dist-t 0.8
etf-daily rules7 --listing-dir "$PLOTLY_ROOT/2026103_from_listing"
```

`listing` uses the newest directory under `runtime/decision_packs/intraday/{YYYYMMDD}/` that contains both `signals_oos.csv` and `live_snapshot.csv`. The formal shard under `runtime/decision_packs/20260720` stays as it was.

## Layout

```
src/etf_daily/cli.py          command dispatch
src/etf_daily/etl/            Sina fund refresh
src/etf_daily/pool/builder/   candidate pool
src/etf_daily/regime/         signal shard
src/etf_daily/plots/          listing HTML, plus the separate rules7 forward page
src/etf_daily/hrp/            dendrogram
src/etf_daily/data/           Qlib load and forward adjustment
scripts/                      shells used by regime, adaptive, and listing
configs/paths.example.json    data paths
```

## Tests

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q -m "not qlib"
bash tests/test_daily_env.sh
```

`pytest` without the marker also runs the tests that read a local Qlib store. Those are skipped when Qlib is not installed.

## License

MIT. See [LICENSE](LICENSE).
