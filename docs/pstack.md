# Use pstack on a change to this repo

pstack is a Cursor plugin for how an agent plans, edits, and checks a change. It is not a Python package. `pip install` does not install it, and this repo does not vendor its skills.

Installing the plugin on one Cursor account does not enable it on every machine. A cloud agent, a second laptop, or a fresh checkout still needs the plugin on that machine. If it is missing, say so and work from the public skill files. Do not claim the account install covered the machine.

Official tree: <https://github.com/cursor/plugins/tree/main/pstack>

## Install

In Cursor, run:

```text
/add-plugin pstack
```

Do not copy the upstream skills into this repo. Do not add pstack to `pyproject.toml`.

This repo does not store model choices. Do not run `/setup-pstack` as part of a change here, and do not commit a model rule.

## Start the skill

Type `/poteto-mode` and choose it before the rest of the request, so Cursor attaches the skill. A sentence that only mentions the name does not start it.

## A task that fits

Use it for a small change to the daily commands, the docs, or a test that does not need a private Qlib store.

```text
/poteto-mode
scripts/accept_eod.sh --print-plan should keep the --next-day I pass, even when that day is not the next weekday.
Show the current plan first. Do not run etl, and do not change hold_up or rules7 thresholds.
Done when bash tests/test_daily_env.sh passes and the printed triggers line still contains the date I passed.
```

The done check is something you can rerun. `ruff check src tests`, `pytest -q -m "not qlib"`, and `bash tests/test_daily_env.sh` are the checks that do not need the fund store. A full `etf-daily etl` is not one of those checks. It needs network access and a local Qlib store.

## Leave these alone

Do not place orders, connect a broker, or edit `hold_up`. Do not commit `config.env`, `configs/paths.json`, `runtime/`, or `temp/`. Do not change the rules7 thresholds or the research numbers in `docs/rules7.md` and `docs/b1235.md` as a side effect of a workflow edit.

`scripts/accept_eod.sh` runs eight commands. `rules7` is not one of them. The research list in `README.md` is a different list.
