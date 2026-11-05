# Candidate pool

`etf-daily pool` builds the candidate pool from the local Qlib fund store and `paths.fund_list_csv`. It writes three files under `paths.temp_dir`:

- `cluster_mapping.csv` — every name considered, with its cluster
- `cluster_mapping_selected.csv` — the names that passed the filters
- `cluster_mapping_selected.txt` — the same selection in Qlib instrument format

`etf-daily cluster-review` then checks that each selected name is still a reasonable representative of its cluster. `--future-end` defaults to today.

Filters and clustering thresholds live in `src/etf_daily/pool/shared_filter_config.json`. The implementation is `src/etf_daily/pool/builder/` (`cli.py` runs the build, `daily_review.py` runs the review). `src/etf_daily/pool/build_pool.py` is a thin wrapper around the same build.

There is no CSI800/CSI1000 merge step. Edit the fund list CSV to add or remove a fund, then run `etf-daily pool` again.
