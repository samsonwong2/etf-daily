"""One-shot simple-pulse mean pipeline orchestrator.

Production entry is now ``run.py pipeline``. See the project README for
operator-facing documentation.

Module layout:

    constants     - defaults + mutable _DRY_RUN / _LOG_DIR flags
    infra         - file lock, stage filename sanitizer, _run_command
    paths         - _resolve_paths + filename helpers
    data_prep     - code list, qlib raw panel, AkShare spot append
    route_a       - Route A full / rolling-refresh + manifest
    bridge        - bridge.py invocation
    risk_controls - apply_risk_controls (L2 cap + L3 regime/vol/cluster + anchor)
    stages        - run_backtest, run_monitor
    cli           - build_parser, validate_args, main
"""
