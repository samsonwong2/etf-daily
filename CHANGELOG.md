# Changelog

## Unreleased

`scripts/accept_eod.sh --print-plan` prints the eight commands the close actually runs. `rules7` and the default-distance HRP page stay out of that script. The README research list still includes both. `--next-day` is the next trading session. The script does not rewrite it to the next weekday.

Developer notes for the pstack Cursor plugin live in `docs/pstack.md`. The plugin is not a package dependency.

## 0.2.0

`rules7` buy status now says `快形态挡住` only when a real aux_edge or B1 buy was blocked today. A name in the fast pattern with no buy trigger reads `快形态中，未到买点`. The checklist adds `快形态剩余K线`, the bars until the fast pattern clears if no new 20% run appears.

`rules7` checklist adds an observation-only flat-buy table sorted by the fig9 EWMA lower rail (`EWMA买点价` = that rail × 1.01). It does not change today's action, the buy signal, or fills. The existing table stays on the causal P95 rail.

The repository now contains the daily command path only: `etl`, `pool`, `cluster-review`, `regime`, `adaptive`, `listing`, `hrp`, `triggers`, `b1235`, `b1235-backtest`, `rules7`, and `morning`.

Research backtests, the portfolio pipeline (`pipeline`, `health`, `all`), and the triangle ops scan are no longer in the tree. The tag `pre-cleanup` points at the tree from before this cut.

Local paths now live in `configs/paths.json` (copy `configs/paths.example.json`). `configs/production_regime_switch_ewma_shrink.json` is still read when `paths.json` is absent.

## 0.1.0

Initial daily ETF commands.
