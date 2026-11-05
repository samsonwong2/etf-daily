# Changelog

## 0.2.0

The repository now contains the daily command path only: `etl`, `pool`, `cluster-review`, `regime`, `adaptive`, `listing`, `hrp`, `triggers`, `b1235`, `b1235-backtest`, `rules7`, and `morning`.

Research backtests, the portfolio pipeline (`pipeline`, `health`, `all`), and the triangle ops scan are no longer in the tree. The tag `pre-cleanup` points at the tree from before this cut.

Local paths now live in `configs/paths.json` (copy `configs/paths.example.json`). `configs/production_regime_switch_ewma_shrink.json` is still read when `paths.json` is absent.

## 0.1.0

Initial daily ETF commands.
