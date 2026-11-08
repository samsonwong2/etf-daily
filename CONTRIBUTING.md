# Contributing

Bug fixes and small improvements to the daily commands are welcome.

1. Fork the repository and create a branch from `master`.
2. Install the package and the dev tools: `pip install -e ".[dev]"`.
3. Qlib is not installed by that command. The daily commands need a local Qlib fund store. Tests that read it are marked `qlib` and are skipped when Qlib is absent.
4. Run `ruff check src tests` and `pytest -q -m "not qlib"` before opening a pull request. Run the full `pytest` when you have Qlib data.
5. Do not commit `config.env`, `configs/paths.json`, `runtime/`, or `temp/`. Those are machine-local.
6. For an agent-assisted change, follow [docs/pstack.md](docs/pstack.md). pstack is a Cursor plugin. It is not a dependency of `etf-daily`.

The daily command names and flags are the public interface. A change that alters their output needs a note in the pull request.
