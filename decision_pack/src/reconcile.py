"""Live vs pipeline holdings reconciliation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config
from decision_pack.src.portfolio import PortfolioSnapshot


@dataclass(frozen=True)
class ReconcileResult:
    warning: bool
    per_symbol_threshold: float
    total_exposure_threshold: float
    max_symbol_diff: float
    total_equity_diff: float
    mismatches: tuple[dict[str, Any], ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning": self.warning,
            "per_symbol_threshold": self.per_symbol_threshold,
            "total_exposure_threshold": self.total_exposure_threshold,
            "max_symbol_diff": self.max_symbol_diff,
            "total_equity_diff": self.total_equity_diff,
            "mismatches": list(self.mismatches),
            "message": self.message,
        }


def reconcile_portfolios(
    live: PortfolioSnapshot,
    pipeline: PortfolioSnapshot,
    config: DecisionPackConfig | None = None,
) -> ReconcileResult:
    cfg = config or load_decision_pack_config()
    reconcile_raw = cfg.raw.get("reconcile") or {}
    per_symbol_threshold = float(reconcile_raw.get("per_symbol_weight_diff", 0.02))
    total_threshold = float(reconcile_raw.get("total_equity_exposure_diff", 0.03))

    live_by_code = {row.code: row.weight for row in live.holdings}
    pipe_by_code = {row.code: row.weight for row in pipeline.holdings}
    all_codes = sorted(set(live_by_code) | set(pipe_by_code))

    mismatches: list[dict[str, Any]] = []
    max_symbol_diff = 0.0
    for code in all_codes:
        live_w = live_by_code.get(code, 0.0)
        pipe_w = pipe_by_code.get(code, 0.0)
        diff = abs(live_w - pipe_w)
        max_symbol_diff = max(max_symbol_diff, diff)
        if diff > per_symbol_threshold:
            mismatches.append(
                {
                    "code": code,
                    "live_weight": live_w,
                    "pipeline_weight": pipe_w,
                    "diff": diff,
                }
            )

    total_equity_diff = abs(live.equity_exposure - pipeline.equity_exposure)
    warning = bool(mismatches) or total_equity_diff > total_threshold
    if warning:
        message = (
            "RECONCILE_WARNING: live holdings differ from pipeline weights; "
            "tier and orders use live snapshot."
        )
    else:
        message = "Reconcile OK: live and pipeline exposures align within thresholds."

    return ReconcileResult(
        warning=warning,
        per_symbol_threshold=per_symbol_threshold,
        total_exposure_threshold=total_threshold,
        max_symbol_diff=max_symbol_diff,
        total_equity_diff=total_equity_diff,
        mismatches=tuple(mismatches),
        message=message,
    )


def render_reconcile_report(result: ReconcileResult) -> str:
    lines = [
        "# Reconcile Report",
        "",
        f"- warning: **{result.warning}**",
        f"- max_symbol_diff: {result.max_symbol_diff:.4f}",
        f"- total_equity_diff: {result.total_equity_diff:.4f}",
        f"- per_symbol_threshold: {result.per_symbol_threshold:.4f}",
        f"- total_exposure_threshold: {result.total_exposure_threshold:.4f}",
        "",
        result.message,
        "",
    ]
    if result.mismatches:
        lines.append("## Symbol mismatches")
        lines.append("")
        lines.append("| code | live_weight | pipeline_weight | diff |")
        lines.append("|------|-------------|-----------------|------|")
        for item in result.mismatches:
            lines.append(
                f"| {item['code']} | {item['live_weight']:.4f} | "
                f"{item['pipeline_weight']:.4f} | {item['diff']:.4f} |"
            )
    return "\n".join(lines) + "\n"


__all__ = ["ReconcileResult", "reconcile_portfolios", "render_reconcile_report"]
