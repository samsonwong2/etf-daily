#!/usr/bin/env python3
"""Generate fat-tail risk analysis markdown from decision_pack artifacts.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/generate_fat_tail_risk_report.py \\
    --pack-dir ~/temp/decision_packs/20260615
  # -> FAT_TAIL_RISK_REPORT_756.md, barbell_proposal_756.csv, ...

  $PY src/etf_daily/scripts/generate_fat_tail_risk_report.py \\
    --pack-dir ~/temp/decision_packs/20260615 --lookback-days 252
  # -> *_252.*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_portfolio import (
    BarbellProposal,
    build_mean_signal_table,
    concavity_flags,
    hard_stop_k,
    kelly_cap,
    build_exit_tier_table,
    name_exit_plan,
    name_risk_action,
    propose_from_holdings,
    propose_from_pool,
)
from etf_daily.lib.fat_tail_regime import enrich_regime_columns
from etf_daily.lib.fat_tail_risk import (
    FatTailConfig,
    build_fat_tail_frame,
    fat_tail_config_from_raw,
    portfolio_weighted_summary,
)
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, FUND_LIST_CSV, QLIB_PROVIDER_URI
from etf_daily.lib.generate_daily_mu_position_report import load_fund_name_map, normalize_code


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_as_of(pack_dir: Path) -> str:
    tier_path = pack_dir / "tier_decision.json"
    if tier_path.exists():
        return str(_load_json(tier_path).get("as_of", ""))
    market_path = pack_dir / "market_snapshot.json"
    if market_path.exists():
        return str(_load_json(market_path).get("as_of", ""))
    raise FileNotFoundError(f"Could not resolve as_of from {pack_dir}")


def load_holdings_from_pack(pack_dir: Path) -> pd.DataFrame:
    path = pack_dir / "portfolio_snapshot.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing portfolio snapshot: {path}")
    frame = pd.read_csv(path)
    if "code" not in frame.columns:
        raise ValueError(f"No code column in {path}")
    frame["code"] = frame["code"].map(normalize_code)
    return frame


def warmup_start(as_of: str, cfg: FatTailConfig) -> str:
    end = pd.Timestamp(as_of)
    days = cfg.lookback_days + cfg.warmup_buffer_days
    return (end - pd.tseries.offsets.BDay(days)).strftime("%Y-%m-%d")


def build_universe_codes(
    holdings: pd.DataFrame,
    cluster_codes: tuple[str, ...],
    defensive_codes: frozenset[str] | set[str] | None = None,
) -> list[str]:
    codes = [normalize_code(c) for c in holdings["code"].tolist() if normalize_code(c) != "CASH"]
    for code in cluster_codes:
        norm = normalize_code(code)
        if norm and norm not in codes:
            codes.append(norm)
    for code in defensive_codes or ():
        norm = normalize_code(code)
        if norm and norm != "CASH" and norm not in codes:
            codes.append(norm)
    return codes


def load_fat_tail_data(
    pack_dir: Path,
    *,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    defensive_codes: frozenset[str] | set[str] | None = None,
    cfg: FatTailConfig | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or FatTailConfig()
    defensive = {normalize_code(c) for c in (defensive_codes or set())}
    as_of = load_as_of(pack_dir)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(cluster_mapping_path or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    universe = build_universe_codes(holdings, cluster_codes, defensive)

    name_map = load_fund_name_map(FUND_LIST_CSV) if FUND_LIST_CSV.exists() else {}
    weights = {
        normalize_code(str(row["code"])): float(row.get("weight_total", row.get("weight", 0.0)))
        for _, row in holdings.iterrows()
    }
    names = {code: name_map.get(code, code) for code in universe}
    for _, row in holdings.iterrows():
        code = normalize_code(str(row["code"]))
        if row.get("name"):
            names[code] = str(row["name"])

    uri = str(provider_uri or QLIB_PROVIDER_URI)
    start = warmup_start(as_of, cfg)
    panel = load_qlib_close(uri, universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    full_frame = build_fat_tail_frame(
        universe,
        panel,
        as_of,
        weights=weights,
        names=names,
        defensive_codes=defensive,
        cfg=cfg,
    )
    # Attach cost basis / current price (§12 stop/take-profit) for held names.
    cost_map: dict[str, float] = {}
    price_map: dict[str, float] = {}
    for _, row in holdings.iterrows():
        code = normalize_code(str(row["code"]))
        cost = pd.to_numeric(row.get("cost_price"), errors="coerce")
        close = pd.to_numeric(row.get("close_price"), errors="coerce")
        if pd.notna(cost):
            cost_map[code] = float(cost)
        if pd.notna(close):
            price_map[code] = float(close)
    full_frame["cost_price"] = full_frame["code"].map(cost_map)
    full_frame["close_price"] = full_frame["code"].map(price_map)
    full_frame = enrich_regime_columns(full_frame, panel, as_of, cfg)
    held_codes = [
        normalize_code(c)
        for c in holdings["code"].tolist()
        if normalize_code(c) != "CASH"
    ]
    # Defensive leg belongs to the holdings/safe board even when not currently held,
    # so the barbell safe leg is visible.
    holdings_mask = full_frame["code"].isin(held_codes) | full_frame["is_defensive"]
    holdings_frame = full_frame.loc[holdings_mask].copy()
    cluster_frame = full_frame.copy()
    return as_of, holdings_frame, cluster_frame, full_frame


def _pct(value: float | None, *, digits: int = 2, scale: float = 100.0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * scale:.{digits}f}%"


def _num(value: float | None, *, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _flags_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value) if value else "—"
    text = str(value).strip()
    if text in {"", "()", "[]"}:
        return "—"
    return text


def _sort_risk_board(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["_alpha_sort"] = pd.to_numeric(out["alpha_left"], errors="coerce").fillna(999)
    out["_cvar_sort"] = pd.to_numeric(out["cvar_left"], errors="coerce").fillna(0)
    out = out.sort_values(
        ["_alpha_sort", "_cvar_sort"],
        ascending=[True, True],
    ).drop(columns=["_alpha_sort", "_cvar_sort"])
    return out.reset_index(drop=True)


def _render_metrics_table(frame: pd.DataFrame, *, empty_message: str) -> list[str]:
    lines = [
        "| code | name | wt | n | MAD_ann | α_L | α_R | κ | shadow μ | shrink | shrunk μ | VaR99 | CVaR99 | leg | flags |",
        "|------|------|----|---|---------|-----|-----|---|----------|--------|----------|-------|--------|-----|-------|",
    ]
    if frame.empty:
        lines.append(f"| — | {empty_message} | — | — | — | — | — | — | — | — | — | — | — | — | — |")
        return lines

    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row["name"]),
                    f"{float(row['weight']):.4f}",
                    str(int(row["n_samples"])),
                    _pct(row.get("mad_ann")),
                    _num(row.get("alpha_left"), digits=2),
                    _num(row.get("alpha_right"), digits=2),
                    _num(row.get("kappa"), digits=3),
                    _pct(row.get("shadow_mean")),
                    _num(row.get("shrink_factor"), digits=2),
                    _pct(row.get("shrunk_mean")),
                    _pct(row.get("var_left")),
                    _pct(row.get("cvar_left")),
                    str(row.get("barbell_leg") or "—"),
                    _flags_str(row.get("flags")),
                ]
            )
            + " |"
        )
    return lines


def _alert_rows(frame: pd.DataFrame, cfg: FatTailConfig) -> pd.DataFrame:
    if frame.empty:
        return frame
    alpha = pd.to_numeric(frame["alpha_left"], errors="coerce")
    cvar = pd.to_numeric(frame["cvar_left"], errors="coerce")
    reliable = ~frame["flags"].astype(str).str.contains(
        "low_n|alpha_unreliable|missing_price"
    )
    # red = barbell avoid (genuinely fat / extreme left tail).
    red = frame["barbell_leg"].eq("avoid")
    # yellow = reliable risk_leg with a moderately fat left tail or sizable CVaR.
    yellow = (
        frame["barbell_leg"].eq("risk_leg")
        & reliable
        & ((alpha < cfg.alpha_thin) | (cvar <= -cfg.cvar_safe_max))
    ) & ~red
    out = frame.loc[red | yellow].copy()
    out["alert_level"] = ["red" if r else "yellow" for r in red.loc[out.index]]
    return out


def _render_mean_signal_section(mean_signal: pd.DataFrame) -> list[str]:
    lines = [
        "## 均值预测（弱信号 · §3.1/§11.2）",
        "",
        "> 肥尾下样本均值收敛慢且**系统性偏低**——偏低的质量落在**尾部=风险**，"
        "不是可安全赚取的收益。故均值仅作弱信号：用 plug-in/shadow mean 估计并强力 shrink，"
        "**不**作为优化器核心输入。",
        "",
    ]
    if mean_signal.empty:
        lines.append("- (no holdings)")
        return lines
    lines.extend(
        [
            "| code | name | wt | 样本μ | shadowμ | shadow/样本 | shrink | shrunkμ |",
            "|------|------|----|-------|---------|-------------|--------|---------|",
        ]
    )
    for _, row in mean_signal.iterrows():
        ratio = row.get("shadow_to_sample")
        ratio_str = _num(ratio, digits=2) + "×" if ratio is not None and pd.notna(ratio) else "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row["name"]),
                    f"{float(row['weight']):.4f}",
                    _pct(row.get("sample_mu")),
                    _pct(row.get("shadow_mu")),
                    ratio_str,
                    _num(row.get("shrink_factor"), digits=2),
                    _pct(row.get("shrunk_mu")),
                ]
            )
            + " |"
        )
    return lines


def _render_proposal_section(proposal: BarbellProposal) -> list[str]:
    n_note = ""
    if proposal.n_risk and proposal.n_risk < proposal.n_suggested:
        n_note = (
            f"（当前 {proposal.n_risk} 只 < κ 建议 {proposal.n_suggested} 只，"
            "宜进一步分散）"
        )
    if proposal.objective == "return_max":
        breach = proposal.portfolio_crash_cvar > proposal.k_budget + 1e-9
        lines = [
            f"### {proposal.label}",
            "",
            f"- 满仓进攻：risk 100% / safe 0%（**无安全端缓冲**，{proposal.n_risk} 只按收益/尾部 Kelly 倾斜）",
            f"- κ 定股数（广度）: 建议 ≈ {proposal.n_suggested} 只"
            f"（κ_eff={_num(proposal.kappa_eff, digits=3)}）{n_note}",
            f"- 组合危机CVaR（相关性→1，无分散）: -{proposal.portfolio_crash_cvar:.2%}"
            f" vs 预算 K -{proposal.k_budget:.2%} → "
            + ("**超预算 ⚠（满仓无缓冲，须靠总额硬止损执行）**" if breach else "在预算内 ✅"),
        ]
        if proposal.no_edge:
            lines.append(
                "- ⚠ 候选无正预期 edge（shrunkμ≤0），已回退等权；收益最大化无依据。"
            )
    else:
        lines = [
            f"### {proposal.label}",
            "",
            f"- κ 定股数（广度）: 建议 ≈ {proposal.n_suggested} 只"
            f"（κ_eff={_num(proposal.kappa_eff, digits=3)}）{n_note}",
            f"- barbell 两端比例：safe={proposal.w_safe:.2%} / risk={proposal.w_risk:.2%}"
            f"（进攻端 {proposal.n_risk} 只等权、安全端 {proposal.n_safe} 只）",
            f"- 进攻端危机CVaR（相关性→1，无分散）: -{proposal.cvar_risk_crash:.2%}"
            if proposal.cvar_risk_crash
            else "- 进攻端危机CVaR：—",
            f"- 预算 K（可承受左尾损失）: -{proposal.k_budget:.2%} → "
            f"组合危机CVaR≈ -{proposal.portfolio_crash_cvar:.2%}",
        ]
    if proposal.fill_fraction > 1e-6:
        lines.append(
            f"- ⚠ 约 {proposal.fill_fraction:.0%} 权重 CVaR 不可估，已用最坏可得值保守填充。"
        )
    if proposal.excluded_avoid:
        lines.append(
            f"- 已剔除 avoid（左尾过重）: {', '.join(proposal.excluded_avoid)}"
        )
    lines.append("")
    if proposal.table.empty:
        lines.append("- (empty proposal)")
        return lines
    lines.extend(
        [
            "| leg | code | name | 目标权重 | α_L | κ | CVaR99 |",
            "|-----|------|------|----------|-----|---|--------|",
        ]
    )
    for _, row in proposal.table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["leg"]),
                    str(row["code"]),
                    str(row["name"]),
                    f"{float(row['target_weight']):.2%}",
                    _num(row.get("alpha_left"), digits=2),
                    _num(row.get("kappa"), digits=3),
                    _pct(row.get("cvar_left")),
                ]
            )
            + " |"
        )
    return lines


def _render_risk_control_section(
    proposal_a: BarbellProposal,
    summary: dict[str, Any],
    concavity: pd.DataFrame,
    kelly: pd.DataFrame,
    cfg: FatTailConfig,
) -> list[str]:
    stop = hard_stop_k(proposal_a, cfg)
    weighted_cvar = summary.get("weighted_cvar_left")
    return_max = proposal_a.objective == "return_max"
    lines = [
        "## 组合风险控制（§9 + §10.1）",
        "",
        "> 风控不是“测组合方差再调仓”，而是**改变暴露形状、避开吸收壁**：在组合**总额**上设"
        "不可越过的硬止损 K；用 Kelly 上限防 ruin。",
        "",
        "### 硬止损 K（作用于组合总额 · §25.5）",
        "",
        f"- 总额硬止损 K = -{stop['k_total']:.2%}（日度，作用于组合总市值，非单只个股）。",
    ]
    if return_max:
        lines.extend(
            [
                "- **满仓进攻：w_safe=0，无 barbell 安全端**，左尾不再被结构性封顶。",
                f"- 组合危机CVaR≈ -{stop['portfolio_crash_cvar']:.2%}，"
                + (
                    "在预算内 ✅。"
                    if stop["within_budget"]
                    else f"**超预算 ⚠（> K -{stop['k_total']:.2%}）：满仓无缓冲，"
                    "尾部纪律全靠总额硬止损 + Kelly 上限强制执行**。"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- 由 barbell 安全端 w_safe={stop['enforced_by_safe_w']:.2%} 把该比例左尾封为≈0 质量。",
                f"- 建议组合危机CVaR≈ -{stop['portfolio_crash_cvar']:.2%}，"
                + ("在预算内 ✅。" if stop["within_budget"] else "**超预算 ⚠，需提高 w_safe 或降进攻端**。"),
                "",
            ]
        )
    lines.extend([
        "### 危机相关性→1 压力测试（§10.1：分散在崩盘时失效）",
        "",
        "> 平时低相关 ≠ 安全；肥尾下多只同时暴跌概率被低估。安全端封顶按**全组合共振**设定。",
        f"- 持仓加权 CVaR99（平时口径）: {_pct(weighted_cvar)}",
        f"- 组合危机 CVaR99（相关性→1，无分散）: -{proposal_a.portfolio_crash_cvar:.2%}"
        if proposal_a.portfolio_crash_cvar
        else "- 组合危机 CVaR99：—",
        "",
    ])

    frag = concavity.loc[concavity["fragile"]] if not concavity.empty else concavity
    lines.extend(["### 凹性 / 脆弱性检测（冲击翻倍是否加速 · §3.12/§9.3.4）", ""])
    lines.append(
        f"> 用左尾 GPD 比较 p 与 p/2 处的损失；比值 ≥ {cfg.concavity_ratio_warn} 视为脆弱"
        "（short-gamma 形，损失随强度加速）。"
    )
    lines.append("")
    if concavity.empty or concavity["concavity_ratio"].notna().sum() == 0:
        lines.append("- (无可估算的左尾 GPD，跳过)")
    elif frag.empty:
        lines.append("- 无脆弱标记：所有可估持仓的尾部损失未见明显加速。")
    else:
        lines.extend(
            [
                "| code | name | wt | α_L | loss@p | loss@p/2 | 加速比 |",
                "|------|------|----|-----|--------|----------|--------|",
            ]
        )
        for _, row in frag.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["code"]),
                        str(row["name"]),
                        f"{float(row['weight']):.4f}",
                        _num(row.get("alpha_left"), digits=2),
                        _pct(row.get("loss_p"))
                        if row.get("loss_p") is not None
                        else "—",
                        _pct(row.get("loss_phalf"))
                        if row.get("loss_phalf") is not None
                        else "—",
                        _num(row.get("concavity_ratio"), digits=2),
                    ]
                )
                + " |"
            )

    lines.extend(["", "### Kelly 仓位上限（防 ruin · §9.3.5）", ""])
    lines.append(
        f"> f_cap = clip({cfg.kelly_fraction}·shadowμ/|CVaR_L|, 0, {cfg.max_name_weight:.0%})；"
        "尾越厚上限越小；当前权重 > 上限即超配。"
    )
    lines.append("")
    over = kelly.loc[kelly["oversized"]] if not kelly.empty else kelly
    if kelly.empty:
        lines.append("- (no holdings)")
    elif over.empty:
        lines.append("- 无超配：所有持仓权重均在 Kelly 上限内。")
    else:
        lines.extend(
            [
                "| code | name | 当前wt | Kelly上限 | shadowμ | CVaR99 |",
                "|------|------|--------|-----------|---------|--------|",
            ]
        )
        for _, row in over.sort_values("weight", ascending=False).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["code"]),
                        str(row["name"]),
                        f"{float(row['weight']):.2%}",
                        f"{float(row['kelly_cap']):.2%}",
                        _pct(row.get("shadow_mean")),
                        _pct(row.get("cvar_left")),
                    ]
                )
                + " |"
            )
    return lines


_TAIL_REGIME_SHORT = {
    "stable": "稳定",
    "tail_thickening_watch": "观察",
    "tail_thickening_confirmed": "重估",
}
_PATH_STATE_SHORT = {
    "near_high": "近高位",
    "pullback": "回撤中",
    "broken": "破位",
}


def _drop_ref_cells(row: pd.Series) -> list[str]:
    """Render the 3 single-day VaR reference depths (monitoring only, no reduction)."""
    drops = row.get("drop_tiers") or []
    cells: list[str] = []
    for idx in range(3):
        drop = drops[idx] if idx < len(drops) else None
        if drop is None or pd.isna(drop):
            cells.append("—")
        else:
            cells.append(f"≥{float(drop) * 100:.1f}%")
    return cells


def _render_pool_exit_tier_table(pool: pd.DataFrame, cfg: FatTailConfig) -> list[str]:
    """Candidate-pool monitoring table: per-name VaR95/99/99.7 single-day reference depths."""
    c1, c2, c3 = cfg.exit_var_confidences
    fb = "/".join(f"-{d:.0%}" for d in cfg.exit_drop_fallback)
    lines = [
        "#### 待选池全部标的：单日异常跌幅参考（cluster_mapping_selected）",
        "",
        f"> 三档跌幅 = 左尾 GPD 单日 VaR **{c1 * 100:g}% / {c2 * 100:g}% / {c3 * 100:g}%**"
        f"（价格跌幅 `1-exp(-loss)`，每股不同）；不可估退固定档 **{fb}**。"
        " **仅作监控参考**——表示「该股单日跌幅超过此深度即属异常」，**不绑定减仓比例、不作触发器**。",
        "> 实际减仓由**组合 NAV 回撤硬止损 + 路径状态/尾结构（§7/§12.3）+ 仓位预算**决定（见上文与持仓表「动作」列）。"
        "「状态」= 从高水位回撤的路径状态（破位 = 长窗深度回撤），**非 MA 趋势**。",
        "",
    ]
    if not pool.empty and "tail_regime" in pool.columns:
        watch_count = int(
            (pool["tail_regime"] == "tail_thickening_watch").sum()
            + (pool["tail_regime"] == "tail_thickening_confirmed").sum()
        )
        if watch_count:
            lines.append(
                f"> 尾部结构层：{watch_count} 只标的处于观察/建议重估（§7 默认不切换全窗 VaR 档）。"
            )
            lines.append("")
    if pool.empty:
        lines.append("- (no cluster symbols)")
        return lines
    lines.extend(
        [
            "| code | name | wt | leg | 尾结构 | 状态 | 回撤长 | "
            f"档1≥%(VaR{c1 * 100:g}) | 档2≥%(VaR{c2 * 100:g}) | 档3≥%(VaR{c3 * 100:g}) | 依据 |",
            "|------|------|----|-----|--------|------|--------|----------|----------|----------|------|",
        ]
    )
    for _, row in pool.iterrows():
        cells = _drop_ref_cells(row)
        basis = "尾部不可估·固定档" if row.get("tail_fallback") else "左尾VaR"
        wt = float(row.get("weight") or 0.0)
        tail_short = _TAIL_REGIME_SHORT.get(str(row.get("tail_regime") or "stable"), "稳定")
        state_short = _PATH_STATE_SHORT.get(str(row.get("path_state") or ""), "—")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row["name"]),
                    f"{wt:.2%}" if wt > 0 else "—",
                    str(row.get("barbell_leg") or "—"),
                    tail_short,
                    state_short,
                    _pct(row.get("dd_long"), digits=1),
                    *cells,
                    basis,
                ]
            )
            + " |"
        )
    return lines


def build_cluster_pool_risk_csv(
    cluster_frame: pd.DataFrame,
    pool_tiers: pd.DataFrame,
    cfg: FatTailConfig,
) -> pd.DataFrame:
    """Merge cluster risk board + VaR reference tables using report display columns."""
    sorted_frame = _sort_risk_board(cluster_frame)
    if sorted_frame.empty:
        return pd.DataFrame()

    tier_lookup = (
        {str(r["code"]): r for _, r in pool_tiers.iterrows()}
        if not pool_tiers.empty
        else {}
    )

    c1, c2, c3 = cfg.exit_var_confidences
    drop_cols = (
        f"档1≥%(VaR{c1 * 100:g})",
        f"档2≥%(VaR{c2 * 100:g})",
        f"档3≥%(VaR{c3 * 100:g})",
    )
    columns = [
        "code",
        "name",
        "wt",
        "n",
        "MAD_ann",
        "α_L",
        "α_R",
        "κ",
        "shadow μ",
        "shrink",
        "shrunk μ",
        "VaR99",
        "CVaR99",
        "leg",
        "flags",
        "尾结构",
        "状态",
        "回撤长",
        *drop_cols,
        "依据",
    ]

    rows: list[dict[str, str]] = []
    for _, row in sorted_frame.iterrows():
        tier = tier_lookup.get(str(row["code"]))
        tier_row = tier if tier is not None else pd.Series(dtype=object)
        drop_cells = _drop_ref_cells(tier_row)
        basis = "尾部不可估·固定档" if tier_row.get("tail_fallback") else "左尾VaR"
        rows.append(
            {
                "code": str(row["code"]),
                "name": str(row["name"]),
                "wt": f"{float(row['weight']):.4f}",
                "n": str(int(row["n_samples"])),
                "MAD_ann": _pct(row.get("mad_ann")),
                "α_L": _num(row.get("alpha_left"), digits=2),
                "α_R": _num(row.get("alpha_right"), digits=2),
                "κ": _num(row.get("kappa"), digits=3),
                "shadow μ": _pct(row.get("shadow_mean")),
                "shrink": _num(row.get("shrink_factor"), digits=2),
                "shrunk μ": _pct(row.get("shrunk_mean")),
                "VaR99": _pct(row.get("var_left")),
                "CVaR99": _pct(row.get("cvar_left")),
                "leg": str(row.get("barbell_leg") or "—"),
                "flags": _flags_str(row.get("flags")),
                "尾结构": _TAIL_REGIME_SHORT.get(
                    str(tier_row.get("tail_regime") or "stable"), "稳定"
                ),
                "状态": _PATH_STATE_SHORT.get(str(tier_row.get("path_state") or ""), "—"),
                "回撤长": _pct(tier_row.get("dd_long"), digits=1),
                drop_cols[0]: drop_cells[0],
                drop_cols[1]: drop_cells[1],
                drop_cols[2]: drop_cells[2],
                "依据": basis if tier is not None else "—",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _render_holdings_monitor(plan: pd.DataFrame, cfg: FatTailConfig) -> list[str]:
    """Per-holding monitoring + action board (no per-name σ-stop, no cost-basis split)."""
    c1, c2, c3 = cfg.exit_var_confidences
    lines = [
        "#### 当前持仓：监控参考 + 动作（非机械止损）",
        "",
        "> 单日异常参考 = 该股左尾 VaR 深度（监控用，不绑定减仓比例）；**成本盈亏、MA20/ret_20d 均仅展示**、不作减仓依据"
        "（§3.10 不锚定回本；§7.1/§12 不用均值/MA 预测方向）。"
        " **动作**由路径状态（高水位回撤）/尾结构（§7）+ 仓位预算（kelly 上限）共同决定；组合 NAV 回撤硬止损独立于此表。",
        "",
        "| code | name | wt | kelly上限 | 超配? | 尾结构 | 状态 | 回撤长 | "
        f"档1≥%(VaR{c1 * 100:g}) | 档2≥%(VaR{c2 * 100:g}) | 档3≥%(VaR{c3 * 100:g}) | 成本盈亏(仅展示) | 动作 |",
        "|------|------|----|----------|-------|--------|------|--------|----------|----------|----------|----------------|------|",
    ]
    if plan.empty:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — |")
        return lines
    for _, row in plan.iterrows():
        cells = _drop_ref_cells(row)
        kelly = row.get("kelly_cap")
        kelly_s = f"{float(kelly):.2%}" if kelly is not None and pd.notna(kelly) else "—"
        oversized = "是" if bool(row.get("oversized")) else "否"
        tail_short = _TAIL_REGIME_SHORT.get(str(row.get("tail_regime") or "stable"), "稳定")
        state_short = _PATH_STATE_SHORT.get(str(row.get("path_state") or ""), "—")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row["name"]),
                    f"{float(row['weight']):.2%}",
                    kelly_s,
                    oversized,
                    tail_short,
                    state_short,
                    _pct(row.get("dd_long"), digits=1),
                    *cells,
                    _pct(row.get("unrealized_return")),
                    str(row.get("action") or "—"),
                ]
            )
            + " |"
        )
    return lines


def _render_regime_section(holdings_frame: pd.DataFrame, cfg: FatTailConfig) -> list[str]:
    """§7 tail structure + §12.3 path/drawdown state (default no-switch)."""
    lines = [
        "## Regime / 路径状态层（§7 默认不切换）",
        "",
        "> **默认立场**：大跳跃是肥尾分布预期内的噪音，**不因单日大跌切换分布**或重算 VaR 三档。"
        " 仅当近期窗 α_L 相对全窗持续偏低（尾变厚）时升级尾部结构状态；"
        f"近期窗 = **{cfg.regime_alpha_recent_days}** 日，全窗 = **{cfg.lookback_days}** 日。",
        f"> 观察阈值：近期/全窗 α 比 < **{cfg.regime_alpha_watch_ratio:.0%}**；"
        f"确认阈值 < **{cfg.regime_alpha_confirm_ratio:.0%}** 且连续 **{cfg.regime_alpha_confirm_streak}** 次检查。",
        "> **路径状态层**（从高水位回撤，§12.3 路径/ruin）与尾部结构层共同决定**每股动作**：长窗"
        f"（**{cfg.drawdown_long_days}** 日）回撤 ≥ **{cfg.drawdown_broken_pct:.0%}** 判**破位**，≥ **{cfg.drawdown_pullback_pct:.0%}** 判**回撤中**；"
        f"近窗（**{cfg.drawdown_near_days}** 日）急跌但长窗未坏只算回撤、不算破位。"
        " **MA20 / ret_20d 仅展示**、不驱动动作（§7.1/§12 不用均值/MA 预测方向）。实际硬止损仍在**组合 NAV 回撤**。",
        "",
        "### 持仓：尾部结构 × 路径状态",
        "",
        "| code | name | wt | 尾结构 | 状态 | α_L全窗 | α_L近期 | κ | 回撤近 | 回撤长 | dist_ma20(展示) | ret_20d(展示) | 结构/路径动作 | 备注 |",
        "|------|------|----|--------|------|---------|---------|---|--------|--------|-----------------|---------------|---------------|------|",
    ]
    held = holdings_frame.loc[pd.to_numeric(holdings_frame["weight"], errors="coerce").fillna(0) > 0]
    if held.empty:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
        return lines

    caps = kelly_cap(holdings_frame, cfg)
    oversized_codes = (
        {str(r["code"]) for _, r in caps.iterrows() if bool(r["oversized"])}
        if not caps.empty
        else set()
    )
    for _, row in held.iterrows():
        tail_r = str(row.get("tail_regime") or "stable")
        tail_zh = {
            "stable": "稳定",
            "tail_thickening_watch": "尾变厚·观察",
            "tail_thickening_confirmed": "尾漂移·建议重估",
        }.get(tail_r, tail_r)
        state_zh = _PATH_STATE_SHORT.get(str(row.get("path_state") or ""), "—")
        action, _reason = name_risk_action(
            {
                "tail_regime": row.get("tail_regime"),
                "path_state": row.get("path_state"),
                "oversized": str(row["code"]) in oversized_codes,
            }
        )

        dist_ma = row.get("dist_ma20_pct")
        dist_s = (
            f"{float(dist_ma):+.1f}%"
            if dist_ma is not None and pd.notna(dist_ma)
            else "—"
        )
        ret20 = row.get("ret_20d")
        ret_s = _pct(ret20) if ret20 is not None and pd.notna(ret20) else "—"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row["name"]),
                    f"{float(row['weight']):.2%}",
                    tail_zh,
                    state_zh,
                    _num(row.get("alpha_left"), digits=2),
                    _num(row.get("alpha_left_recent"), digits=2),
                    _num(row.get("kappa"), digits=3),
                    _pct(row.get("dd_near"), digits=1),
                    _pct(row.get("dd_long"), digits=1),
                    dist_s,
                    ret_s,
                    action,
                    str(row.get("regime_note") or "—"),
                ]
            )
            + " |"
        )
    return lines


def _render_exit_plan_section(
    plan: pd.DataFrame,
    pool_tiers: pd.DataFrame,
    cfg: FatTailConfig,
) -> list[str]:
    lines = [
        "## 止损 / 止盈实施方案（§12 · 组合硬止损为主）",
        "",
        "> §12 原则：**唯一的机械硬止损在组合总额**（高水位回撤线），防 ruin；**止盈要软**"
        "（不机械清仓、不截断右尾）。单股层面**不设逐股价格止损**——既不用 2σ/3σ，也不用"
        "「跌到自身 VaR 分位就减仓」（那等价于随波动缩放的 σ-stop、且按固定频率触发）。"
        " 单股的减仓由**组合回撤 + 路径状态/尾结构（§7/§12.3）+ 仓位预算**共同决定（不用 MA/均值预测方向）。",
        "",
        "### 1. 组合总额硬止损（唯一硬止损 · §12.3）",
        "",
        f"- 高水位回撤硬止损：组合净值自历史高点 H 回撤达 **-{cfg.trailing_stop_drawdown:.0%}** 时，"
        "无条件降到保守端（现金 / 货币 / 对冲）。这是**唯一**不可越过的机械止损。",
        f"- 触发判据：`NAV / H - 1 ≤ -{cfg.trailing_stop_drawdown:.0%}`；每日更新高水位 H、盈利后自动抬高保护线。",
        f"- 与日度左尾预算并行：单日 CVaR99 预算 K = -{cfg.loss_budget_k:.0%}（见“组合风险控制”一节），"
        "回撤线管路径与 ruin，CVaR 预算管单日尾部。",
        "",
        "### 2. 单股仓位预算（§12.2）",
        "",
        f"- 每股仓位上限 = 分数 Kelly `clip({cfg.kelly_fraction}·shadowμ/|CVaR_L|, 0, {cfg.max_name_weight:.0%})`"
        "（见“组合风险控制”一节 kelly 上限表）。**超配**标的应**逢强减至预算上限**，而非等单日大跌。",
        "- 这是单股层面的主纪律：用**预算/仓位**控尾部，不用逐股价格触发。",
        "",
        "### 3. 单股监控参考 + 动作（非执行触发）",
        "",
    ]
    lines.extend(_render_holdings_monitor(plan, cfg))
    lines.append("")
    lines.extend(_render_pool_exit_tier_table(pool_tiers, cfg))

    lines.extend(
        [
            "",
            "### 软止盈与保护线（§12.1 / §12.3）",
            "",
            "- 浮盈个股**不机械清仓**：右尾少数大赢家贡献巨大，过早止盈会砍掉 convex payoff。",
            "- 单股仅在三种情况下减仓：**超配预算（仓位过度集中）**、**组合回撤硬止损触发**、"
            "**基本面/制度性状态改变（§7：α/κ 持续漂移 + 外生证据）或路径破位（§12.3：高水位深度回撤）**。"
            "（书中第三条本是「基本面/制度性状态改变」，本系统用 α/κ 漂移与高水位回撤量化之，**不**用 MA 趋势。）",
            "- 组合创新高时**抬高保护线**：把组合止损线从初始本金改为“高水位回撤线”（见上）。",
            "",
            "### 小结",
            "",
            "**硬止损只在组合总额（高水位回撤线）；单股不设逐股价格止损——靠仓位预算 + 路径状态/尾结构层管风险；"
            "单日 VaR 仅作监控参考，成本盈亏与 MA/动量仅展示、不作锚定或方向预测。止盈要软、保留右尾。**",
        ]
    )
    return lines


def build_fat_tail_report(
    *,
    pack_dir: Path,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    cfg: FatTailConfig | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Build the report markdown, barbell proposal, and merged cluster-pool risk CSV."""
    pack_cfg = load_decision_pack_config()
    cfg = cfg or fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    defensive_codes = frozenset(normalize_code(c) for c in pack_cfg.defensive_codes)
    as_of, holdings_frame, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_mapping_path,
        provider_uri=provider_uri,
        defensive_codes=defensive_codes,
        cfg=cfg,
    )

    holdings_sorted = _sort_risk_board(holdings_frame)
    cluster_sorted = _sort_risk_board(cluster_frame)
    summary = portfolio_weighted_summary(holdings_frame)
    alerts = _alert_rows(holdings_frame, cfg)

    proposal_a = propose_from_holdings(holdings_frame, defensive_codes, cfg)
    proposal_b = propose_from_pool(cluster_frame, defensive_codes, cfg)
    mean_signal = build_mean_signal_table(holdings_frame)
    concavity = concavity_flags(holdings_frame, cfg)
    kelly = kelly_cap(holdings_frame, cfg)
    exit_plan = name_exit_plan(holdings_frame, cfg)
    pool_exit_tiers = build_exit_tier_table(cluster_sorted, cfg)
    cluster_pool_risk = build_cluster_pool_risk_csv(cluster_sorted, pool_exit_tiers, cfg)

    unreliable = holdings_frame[
        holdings_frame["flags"].astype(str).str.contains("low_n|alpha_unreliable|missing_price")
    ]

    safe_rows = holdings_frame.loc[holdings_frame["barbell_leg"] == "safe_leg"]
    risk_rows = holdings_frame.loc[holdings_frame["barbell_leg"] == "risk_leg"]
    avoid_rows = holdings_frame.loc[holdings_frame["barbell_leg"] == "avoid"]

    cov_kappa = float(summary.get("coverage_kappa") or 0.0)
    cov_alpha = float(summary.get("coverage_alpha_left") or 0.0)
    cov_cvar = float(summary.get("coverage_cvar_left") or 0.0)
    fattest_code = summary.get("fattest_code")
    fattest_alpha = summary.get("fattest_alpha_left")
    fattest_label = (
        f"{fattest_code} α_L={_num(fattest_alpha, digits=2)}"
        if fattest_code and fattest_alpha is not None
        else "—"
    )

    # Held names that drop out of the weighted α/CVaR (no estimable left tail).
    held_weight = pd.to_numeric(holdings_frame["weight"], errors="coerce").fillna(0.0)
    held_pos = holdings_frame.loc[held_weight > 0]
    cvar_missing = held_pos.loc[
        pd.to_numeric(held_pos["cvar_left"], errors="coerce").isna()
    ]

    lines: list[str] = [
        "# Fat-Tail Risk Report",
        "",
        f"**As of:** {as_of}  ",
        f"**数据来源:** `{pack_dir}`  ",
        "**框架:** [fat-tail-mean-risk-framework.md](~/etf-daily-output/books/Fat_Tails/fat-tail-mean-risk-framework.md)",
        "",
        "## 方法学声明",
        "",
        "本报告采用书中**认可**的肥尾工具，并**不使用**以下薄尾方法：",
        "",
        "- 标准差 / 方差作为离散度",
        "- 经验分布 VaR / CVaR / 非参数分位数",
        "- 峰度 / 偏度",
        "- winsorize / trimmed / Huber 稳健统计",
        "- 协方差矩阵主导的风险排序",
        "- Sharpe 比率 / 相关系数选股 / 样本内 R²（§10.5）",
        "- PCA / 因子模型 / 随机矩阵去噪（§11.3，依赖结构改用互信息）",
        "- Markowitz 均值方差生成个股权重（个股 κ 过高，椭圆性假设不成立，§8.1）",
        "",
        "核心指标：MAD、左右尾指数 α、κ、plug-in/shadow mean、参数化 EVT VaR/CVaR；"
        "权重由“可承受的左尾损失”反解的 barbell 生成（§8），非协方差优化。",
        "",
        "## 列图例",
        "",
        "| 列 | 含义 |",
        "|----|------|",
        "| MAD_ann | 平均绝对偏差 × √252（替代标准差） |",
        "| α_L / α_R | 左/右尾 POT-GPD 尾指数（越小越厚） |",
        "| κ | 跨分布可比肥尾度，∈[0,1]，越大越厚 |",
        "| shadow μ | plug-in 均值（尾部参数化外推） |",
        "| shrink / shrunk μ | 按 α、κ、样本长度向 0 收缩后的弱信号均值 |",
        "| VaR99 / CVaR99 | 左尾参数化 99% VaR/CVaR（日对数收益） |",
        "| leg | barbell 分层：safe_leg / risk_leg / avoid |",
        "| flags | low_n/alpha_unreliable/missing_price=该侧不可靠；"
        "bounded_tail_left/right=对应一侧尾有界（ξ≤0，非幂律；α 列用 Hill 回退估计） |",
        "",
        "## 组合层快照（持仓加权）",
        "",
        f"- weighted_κ: {_num(summary.get('weighted_kappa'), digits=3)}"
        f"（覆盖 {cov_kappa:.0%} 权重）",
        f"- weighted_α_left（ξ=1/α 空间聚合，避免退化大 α 拉偏）: "
        f"{_num(summary.get('weighted_alpha_left'), digits=2)}（覆盖 {cov_alpha:.0%} 权重）",
        f"- 最厚左尾（真实风险驱动 · α_L 最小持仓）: {fattest_label}",
        f"- weighted_CVaR99: {_pct(summary.get('weighted_cvar_left'))}"
        f"（覆盖 {cov_cvar:.0%} 权重）",
        f"- barbell 权重：safe={summary.get('safe_leg_weight', 0):.2%} / "
        f"risk={summary.get('risk_leg_weight', 0):.2%} / "
        f"avoid={summary.get('avoid_weight', 0):.2%}",
        "",
        f"> safe_leg 由防御性标的（{', '.join(sorted(defensive_codes)) or '无'}）构成；"
        "权益 ETF 因日频 CVaR 普遍较大，一般落在 risk_leg/avoid。",
        "",
    ]

    if not cvar_missing.empty:
        miss_wt = float(
            pd.to_numeric(cvar_missing["weight"], errors="coerce").fillna(0.0).sum()
        )
        miss_codes = "、".join(
            f"{row['code']}({_flags_str(row.get('flags'))})"
            for _, row in cvar_missing.iterrows()
        )
        lines.extend(
            [
                f"> ⚠ **覆盖率提示**：约 {miss_wt:.1%} 权重的持仓无法估计左尾 α/CVaR，"
                "已被排除在上述加权口径之外；这类标的多为样本不足的高波动新券，"
                "因此组合尾部风险可能被**低估**。",
                f"> 未纳入加权：{miss_codes}",
                "",
            ]
        )

    lines.extend(
        [
            f"- 估计窗口：最近 {cfg.lookback_days} 交易日对数收益；左右尾各取全分布 "
            f"{cfg.tail_quantile:.1%} 分位为 POT 阈值；最少超阈值样本 {cfg.min_exceed}",
            f"- 阈值：avoid 当 α_L<{cfg.alpha_fat} 或 κ≥{cfg.kappa_fat_min} 或 CVaR99≤-{cfg.cvar_red:.0%}；"
            f"safe 需 α_L≥{cfg.alpha_thin} 且 κ≤{cfg.kappa_safe_max} 且 CVaR99≥-{cfg.cvar_safe_max:.0%}",
            "",
            "## 持仓肥尾风险板",
            "",
            "> 按左尾风险排序（α_L 越小 / CVaR99 越负优先）。",
            "",
        ]
    )
    lines.extend(_render_metrics_table(holdings_sorted, empty_message="(no holdings)"))
    lines.extend(["", "## 待选池肥尾风险板（cluster_mapping_selected）", ""])
    lines.extend(_render_metrics_table(cluster_sorted, empty_message="(no cluster symbols)"))
    lines.extend(
        [
            "",
            "## barbell 分层建议",
            "",
            "### safe_leg（极保守腿）",
            "",
        ]
    )
    if safe_rows.empty:
        lines.append("- (none)")
    else:
        for _, row in safe_rows.iterrows():
            lines.append(
                f"- {row['code']} {row['name']}：wt={float(row['weight']):.2%}, "
                f"α_L={_num(row.get('alpha_left'), digits=2)}, κ={_num(row.get('kappa'), digits=3)}"
            )

    lines.extend(["", "### risk_leg（可控风险腿）", ""])
    if risk_rows.empty:
        lines.append("- (none)")
    else:
        for _, row in risk_rows.iterrows():
            lines.append(
                f"- {row['code']} {row['name']}：wt={float(row['weight']):.2%}, "
                f"α_L={_num(row.get('alpha_left'), digits=2)}, CVaR99={_pct(row.get('cvar_left'))}"
            )

    lines.extend(["", "### avoid（左尾过重 · 建议降权或剔除）", ""])
    if avoid_rows.empty:
        lines.append("- (none)")
    else:
        for _, row in avoid_rows.iterrows():
            lines.append(
                f"- {row['code']} {row['name']}：wt={float(row['weight']):.2%}, "
                f"α_L={_num(row.get('alpha_left'), digits=2)}, κ={_num(row.get('kappa'), digits=3)}, "
                f"CVaR99={_pct(row.get('cvar_left'))}"
            )

    lines.extend(["", "## 厚尾预警（持仓）", ""])
    if alerts.empty:
        lines.append("- (no red/yellow alerts)")
    else:
        lines.extend(
            [
                "| level | code | name | wt | α_L | κ | CVaR99 | leg | flags |",
                "|-------|------|------|----|-----|---|--------|-----|-------|",
            ]
        )
        for _, row in alerts.iterrows():
            mark = "🔴" if row.get("alert_level") == "red" else "🟡"
            lines.append(
                "| "
                + " | ".join(
                    [
                        mark,
                        str(row["code"]),
                        str(row["name"]),
                        f"{float(row['weight']):.4f}",
                        _num(row.get("alpha_left"), digits=2),
                        _num(row.get("kappa"), digits=3),
                        _pct(row.get("cvar_left")),
                        str(row.get("barbell_leg") or "—"),
                        _flags_str(row.get("flags")),
                    ]
                )
                + " |"
            )

    lines.extend(["", *_render_mean_signal_section(mean_signal)])
    if cfg.proposal_objective == "return_max":
        intro = [
            "",
            "## 组合生成：满仓进攻 · 收益最大化（建议 · 非约束）",
            "",
            "> 按用户指令**追求最大化收益**：满仓投入、无安全端缓冲；权重按 Kelly 方向"
            "（`shrunkμ/|CVaR|`，收益/尾部风险）倾斜，受单只 "
            f"{cfg.max_name_weight:.0%} 上限约束。以下为**建议**，需结合 L1 Tier 与人工 IPS 决策。",
        ]
        if cfg.barbell_exclude_codes:
            intro.append(
                f"> **已剔除**（不纳入组合）：{', '.join(cfg.barbell_exclude_codes)}。"
            )
        intro.extend(
            [
                "> ⚠ 取消 barbell 安全端意味着左尾损失预算 K 不再被结构性封顶，组合危机"
                "CVaR 会超出 K（见下）；尾部纪律改由**总额硬止损 + Kelly 上限**承担。",
                "",
            ]
        )
        lines.extend(intro)
    else:
        lines.extend(
            [
                "",
                "## barbell 组合生成（建议 · 非约束 · §8）",
                "",
                "> 权重不由协方差生成，而由 κ 定股数（过度分散/等权）+ “可承受的左尾损失 K” 反解"
                "两端比例。以下为**建议**，需结合 L1 Tier 与人工 IPS 决策。",
                "",
            ]
        )
    lines.extend(_render_proposal_section(proposal_a))
    lines.append("")
    lines.extend(_render_proposal_section(proposal_b))
    lines.extend(["", *_render_risk_control_section(proposal_a, summary, concavity, kelly, cfg)])
    lines.extend(["", *_render_regime_section(holdings_frame, cfg)])
    lines.extend(["", *_render_exit_plan_section(exit_plan, pool_exit_tiers, cfg)])

    lines.extend(["", "## 可靠性与免责", ""])
    if unreliable.empty:
        lines.append("- 全部持仓指标通过最低样本/拟合可靠性检查。")
    else:
        lines.append("- 以下持仓存在 `low_n` / `alpha_unreliable` / `missing_price`，解读需谨慎：")
        for _, row in unreliable.iterrows():
            lines.append(f"  - {row['code']} {row['name']}：n={int(row['n_samples'])}, flags={_flags_str(row.get('flags'))}")

    lines.extend(
        [
            "",
            "- 本报告为**风险分析与展示**，barbell 组合/权重为**建议**，不构成自动卖单或优化器输入。",
            "- barbell 分层与生成权重仅作配置结构指引，需结合 L1 Tier 与人工 IPS 决策。",
            "",
            "---",
            "",
            "*由 `src/etf_daily/scripts/generate_fat_tail_risk_report.py` 自动生成。*",
            "",
        ]
    )
    proposal_tables = [t for t in (proposal_a.table, proposal_b.table) if not t.empty]
    proposals_df = (
        pd.concat(proposal_tables, ignore_index=True)
        if proposal_tables
        else proposal_a.table
    )
    return "\n".join(lines), proposals_df, cluster_pool_risk


def build_fat_tail_risk_markdown(
    *,
    pack_dir: Path,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    cfg: FatTailConfig | None = None,
) -> str:
    """Backward-compatible wrapper returning only the markdown."""
    markdown, _, _ = build_fat_tail_report(
        pack_dir=pack_dir,
        cluster_mapping_path=cluster_mapping_path,
        provider_uri=provider_uri,
        cfg=cfg,
    )
    return markdown


def lookback_output_suffix(lookback_days: int) -> str:
    """Filename tag from resolved lookback, e.g. ``756`` -> ``_756``."""
    return f"_{int(lookback_days)}"


def suffixed_output_path(path: Path, lookback_suffix: str) -> Path:
    """Insert lookback tag before extension, e.g. ``report.md`` -> ``report_252.md``."""
    return path.with_name(f"{path.stem}{lookback_suffix}{path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fat-tail risk analysis markdown from decision_pack output",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
        help="decision_pack generate 输出目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 MD 路径（默认：<pack-dir>/FAT_TAIL_RISK_REPORT_{lookback}.md）",
    )
    parser.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="cluster_mapping_selected.txt（默认 runtime_paths）",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="收益估计窗口（默认 config fat_tail.lookback_days 或 756）",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=None,
        help="POT 阈值分位（默认 0.925）",
    )
    parser.add_argument(
        "--provider-uri",
        type=Path,
        default=None,
        help="qlib provider URI override",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pack_dir = args.pack_dir.expanduser().resolve()
    pack_cfg = load_decision_pack_config()
    cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    overrides: dict[str, Any] = {}
    if args.lookback_days is not None:
        overrides["lookback_days"] = args.lookback_days
    if args.tail_quantile is not None:
        overrides["tail_quantile"] = args.tail_quantile
    if overrides:
        cfg = FatTailConfig(**{**cfg.__dict__, **overrides})

    lookback_suffix = lookback_output_suffix(cfg.lookback_days)
    output_path = suffixed_output_path(
        args.output.expanduser().resolve()
        if args.output
        else pack_dir / "FAT_TAIL_RISK_REPORT.md",
        lookback_suffix,
    )

    markdown, proposals_df, cluster_pool_risk = build_fat_tail_report(
        pack_dir=pack_dir,
        cluster_mapping_path=args.cluster_mapping,
        provider_uri=args.provider_uri,
        cfg=cfg,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] fat-tail risk report -> {output_path}")

    proposal_path = suffixed_output_path(
        output_path.parent / "barbell_proposal.csv",
        lookback_suffix,
    )
    proposals_df.to_csv(proposal_path, index=False, encoding="utf-8-sig")
    print(f"[OK] barbell proposal     -> {proposal_path}")

    cluster_pool_path = suffixed_output_path(
        output_path.parent / "cluster_mapping_selected_pool_risk.csv",
        lookback_suffix,
    )
    cluster_pool_risk.to_csv(cluster_pool_path, index=False, encoding="utf-8-sig")
    print(f"[OK] cluster pool risk    -> {cluster_pool_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
