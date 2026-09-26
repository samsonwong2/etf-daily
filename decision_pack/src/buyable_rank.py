"""Buyable rank: join tomorrow topk with trend / fat-tail / jump hard filters."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from decision_pack.src.tomorrow_digest import (
    RankRule,
    TopKConfig,
    compute_pick_score,
    compute_upside_skew,
)
from workspace.scripts.generate_daily_mu_position_report import normalize_code

GateName = Literal["strict", "no_path_gate", "core_risk", "positive_score", "none"]

TOMORROW_TOPK_CSV = "tomorrow_topk_picks.csv"
TREND_BOARD_CSV = "cluster_trend_board.csv"
POOL_RISK_252_CSV = "cluster_mapping_selected_pool_risk_252.csv"
POOL_RISK_756_CSV = "cluster_mapping_selected_pool_risk_756.csv"
JUMP_DIAGNOSTICS_CSV = "garch_jump_diagnostics.csv"

BUYABLE_TOPK_CSV = "buyable_topk_picks.csv"
BUYABLE_RANK_REPORT = "BUYABLE_RANK.md"

BUYABLE_COLUMNS: tuple[str, ...] = (
    "rank",
    "code",
    "name",
    "close",
    "as_of",
    "leg",
    "path_state",
    "ma_stack",
    "risk_flags",
    "jump_flags",
    "direction_bias",
    "combined_cred",
    "garch_跌5%",
    "garch_涨95%",
    "upside_skew_pct",
    "pick_score",
    "pass_filters",
    "reject_reason",
    "equal_weight_pct",
)

DEFAULT_TREND_REJECT_FLAGS: tuple[str, ...] = (
    "bear_stack",
    "deep_below_ma20",
    "below_ma20_3d+",
)
DEFAULT_JUMP_REJECT_FLAGS: tuple[str, ...] = (
    "recent_jump",
    "live_price_gap",
)


@dataclass(frozen=True)
class BuyableRankConfig:
    k: int = 8
    rank_all: bool = True
    exclude_avoid_leg: bool = True
    exclude_broken_path: bool = True
    exclude_jump_flags: bool = True
    exclude_trend_flags: bool = True
    jump_reject_flags: tuple[str, ...] = DEFAULT_JUMP_REJECT_FLAGS
    trend_reject_flags: tuple[str, ...] = DEFAULT_TREND_REJECT_FLAGS
    prefer_pool_risk: str = "252"  # 252 | 756
    min_pick_score: float | None = None
    min_credibility: float = 0.0  # equal-weight / suggested picks only


def gate_to_buyable_config(gate: GateName, *, k: int) -> BuyableRankConfig:
    """Map backtest gate name → BuyableRankConfig (same semantics as walk-forward)."""
    base = BuyableRankConfig(k=k, rank_all=True)
    if gate == "none":
        return replace(
            base,
            exclude_avoid_leg=False,
            exclude_broken_path=False,
            exclude_jump_flags=False,
            exclude_trend_flags=False,
            min_pick_score=None,
        )
    if gate == "strict":
        return base
    if gate == "no_path_gate":
        return replace(base, exclude_broken_path=False)
    if gate == "core_risk":
        return replace(
            base,
            exclude_broken_path=False,
            trend_reject_flags=("bear_stack", "deep_below_ma20"),
        )
    if gate == "positive_score":
        return replace(base, min_pick_score=1e-15)
    raise ValueError(f"unknown gate: {gate}")


def buyable_rank_config_from_raw(raw: dict[str, Any] | None) -> BuyableRankConfig:
    if not raw or not isinstance(raw, dict):
        return BuyableRankConfig()
    kwargs: dict[str, Any] = {}
    for key in BuyableRankConfig.__dataclass_fields__:
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if key in {"jump_reject_flags", "trend_reject_flags"}:
            kwargs[key] = tuple(str(x) for x in value)
        else:
            kwargs[key] = value
    return BuyableRankConfig(**kwargs)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _norm_codes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "code" not in frame.columns:
        return frame
    out = frame.copy()
    out["code"] = out["code"].map(normalize_code)
    return out.loc[out["code"].ne("")].copy()


def resolve_pool_risk_path(pack_dir: Path, *, prefer: str = "252") -> Path | None:
    primary = POOL_RISK_252_CSV if prefer == "252" else POOL_RISK_756_CSV
    fallback = POOL_RISK_756_CSV if prefer == "252" else POOL_RISK_252_CSV
    for name in (primary, fallback):
        path = pack_dir / name
        if path.exists():
            return path
    return None


def load_pack_buyable_inputs(
    pack_dir: Path,
    *,
    cfg: BuyableRankConfig | None = None,
) -> dict[str, pd.DataFrame | Path | None]:
    config = cfg or BuyableRankConfig()
    pack = pack_dir.expanduser().resolve()
    topk_path = pack / TOMORROW_TOPK_CSV
    if not topk_path.exists():
        raise FileNotFoundError(
            f"Missing {TOMORROW_TOPK_CSV} under {pack}; "
            "run generate_tomorrow_digest.py first"
        )
    pool_path = resolve_pool_risk_path(pack, prefer=config.prefer_pool_risk)
    return {
        "tomorrow_topk": _norm_codes(_read_csv(topk_path)),
        "trend": _norm_codes(_read_csv(pack / TREND_BOARD_CSV)),
        "pool_risk": _norm_codes(_read_csv(pool_path)) if pool_path else pd.DataFrame(),
        "jump": _norm_codes(_read_csv(pack / JUMP_DIAGNOSTICS_CSV)),
        "pool_risk_path": pool_path,
    }


def _parse_flag_set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "—":
            return set()
        return {f for f in text.split("|") if f}
    if isinstance(value, (list, tuple)):
        return {str(f) for f in value if f}
    return set()


def _to_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "—":
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_pct_number(value: object) -> float | None:
    """Parse '0.80%' / '-7%' style strings to numeric percent points (0.80 / -7)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace("%", "")
    if not text or text == "—":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_upside_skew(value: object) -> float | None:
    """Invert tomorrow_topk upside_skew_pct (= format_pct(upside * 100))."""
    pct_pts = _parse_pct_number(value)
    if pct_pts is None:
        return None
    return pct_pts / 100.0


def recompute_tomorrow_topk_scores(
    tomorrow_topk: pd.DataFrame,
    *,
    rank_rule: RankRule,
    require_positive_direction: bool = True,
    avoid_leg_penalty: float = 0.5,
) -> pd.DataFrame:
    """Rewrite pick_score using direction_bias / combined_cred / upside_skew."""
    if tomorrow_topk.empty:
        return tomorrow_topk.copy()
    topk_cfg = TopKConfig(
        enabled=True,
        rank_all=True,
        rank_rule=rank_rule,
        require_positive_direction=require_positive_direction,
        min_credibility=0.0,
        min_direction=0.0,
        avoid_leg_penalty=avoid_leg_penalty,
        exclude_avoid_leg=False,
    )
    out = tomorrow_topk.copy()
    scores: list[float | None] = []
    for _, row in out.iterrows():
        g_dir = _to_float(row.get("direction_bias"))
        combined = _to_float(row.get("combined_cred"))
        upside = parse_upside_skew(row.get("upside_skew_pct"))
        if upside is None:
            q05_pct = _parse_pct_number(row.get("garch_跌5%"))
            q95_pct = _parse_pct_number(row.get("garch_涨95%"))
            if q05_pct is not None and q95_pct is not None:
                # pct columns are already %-points of simple/log mapped via format_pct(log_return_to_pct)
                # recover approximate log returns: pct/100
                upside = compute_upside_skew(q05_pct / 100.0, q95_pct / 100.0)
        score = compute_pick_score(
            g_dir,
            combined,
            topk_cfg=topk_cfg,
            upside_skew=upside,
            for_ranking=True,
        )
        barbell = str(row.get("barbell_leg") or "").strip()
        if (
            math.isfinite(score)
            and barbell == "avoid"
            and avoid_leg_penalty > 0.0
            and avoid_leg_penalty < 1.0
        ):
            score *= float(avoid_leg_penalty)
        scores.append(round(score, 8) if math.isfinite(score) else None)
    out["pick_score"] = scores
    return out


def load_best_strategy_spec(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    required = ("gate", "rank_rule", "k")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"best_strategy.json missing keys: {missing}")
    return {
        "strategy": str(raw.get("strategy") or ""),
        "gate": str(raw["gate"]),
        "rank_rule": str(raw["rank_rule"]),
        "k": int(raw["k"]),
        "min_credibility": float(raw.get("min_credibility", 0.0)),
    }


def evaluate_buyable_row(
    *,
    leg: str | None,
    path_state: str | None,
    risk_flags: object,
    jump_flags: object,
    pick_score: float | None,
    cfg: BuyableRankConfig,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if cfg.exclude_avoid_leg and (leg or "").strip().lower() == "avoid":
        reasons.append("avoid_leg")

    if cfg.exclude_broken_path and (path_state or "").strip() == "破位":
        reasons.append("path_broken")

    if cfg.exclude_trend_flags:
        trend_flags = _parse_flag_set(risk_flags)
        hit = [f for f in cfg.trend_reject_flags if f in trend_flags]
        if hit:
            reasons.append("trend:" + "+".join(hit))

    if cfg.exclude_jump_flags:
        jump_set = _parse_flag_set(jump_flags)
        hit = [f for f in cfg.jump_reject_flags if f in jump_set]
        if hit:
            reasons.append("jump:" + "+".join(hit))

    if cfg.min_pick_score is not None:
        if pick_score is None or pick_score < float(cfg.min_pick_score):
            reasons.append("low_pick_score")
    elif pick_score is None:
        reasons.append("missing_pick_score")

    if reasons:
        return False, "|".join(reasons)
    return True, ""


def build_buyable_frame(
    tomorrow_topk: pd.DataFrame,
    trend: pd.DataFrame,
    pool_risk: pd.DataFrame,
    jump: pd.DataFrame,
    *,
    cfg: BuyableRankConfig | None = None,
) -> pd.DataFrame:
    config = cfg or BuyableRankConfig()
    if tomorrow_topk.empty:
        return pd.DataFrame(columns=list(BUYABLE_COLUMNS))

    topk = _norm_codes(tomorrow_topk)
    trend_idx = (
        _norm_codes(trend).drop_duplicates("code").set_index("code")
        if not trend.empty and "code" in trend.columns
        else None
    )
    pool_idx = (
        _norm_codes(pool_risk).drop_duplicates("code").set_index("code")
        if not pool_risk.empty and "code" in pool_risk.columns
        else None
    )
    jump_idx = (
        _norm_codes(jump).drop_duplicates("code").set_index("code")
        if not jump.empty and "code" in jump.columns
        else None
    )

    rows: list[dict[str, Any]] = []
    for _, raw in topk.iterrows():
        code = normalize_code(raw.get("code"))
        if not code:
            continue

        leg = None
        path_state = None
        if pool_idx is not None and code in pool_idx.index:
            leg = str(pool_idx.loc[code].get("leg") or "").strip() or None
            path_state = str(pool_idx.loc[code].get("状态") or "").strip() or None
        if not leg:
            barbell = raw.get("barbell_leg")
            if barbell is not None and str(barbell).strip() not in ("", "—", "nan"):
                leg = str(barbell).strip()

        ma_stack = None
        risk_flags = ""
        if trend_idx is not None and code in trend_idx.index:
            ma_stack = str(trend_idx.loc[code].get("ma_stack") or "").strip() or None
            risk_flags = trend_idx.loc[code].get("risk_flags")

        jump_flags = ""
        if jump_idx is not None and code in jump_idx.index:
            jump_flags = jump_idx.loc[code].get("flags")

        pick_score = _to_float(raw.get("pick_score"))
        direction_bias = _to_float(raw.get("direction_bias"))
        combined_cred = _to_float(raw.get("combined_cred"))
        passed, reject_reason = evaluate_buyable_row(
            leg=leg,
            path_state=path_state,
            risk_flags=risk_flags,
            jump_flags=jump_flags,
            pick_score=pick_score,
            cfg=config,
        )

        rows.append(
            {
                "code": code,
                "name": raw.get("name"),
                "close": raw.get("close"),
                "as_of": raw.get("as_of"),
                "leg": leg or "—",
                "path_state": path_state or "—",
                "ma_stack": ma_stack or "—",
                "risk_flags": risk_flags if isinstance(risk_flags, str) and risk_flags else "—",
                "jump_flags": jump_flags if isinstance(jump_flags, str) and jump_flags else "—",
                "direction_bias": direction_bias,
                "combined_cred": combined_cred,
                "garch_跌5%": raw.get("garch_跌5%"),
                "garch_涨95%": raw.get("garch_涨95%"),
                "upside_skew_pct": raw.get("upside_skew_pct"),
                "pick_score": pick_score,
                "pass_filters": passed,
                "reject_reason": reject_reason,
                "_sort_pass": 1 if passed else 0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(BUYABLE_COLUMNS))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["_sort_pass", "pick_score", "code"],
        ascending=[False, False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    if config.rank_all:
        ranked = frame
    else:
        passed_only = frame.loc[frame["pass_filters"]].head(int(config.k))
        rejected = frame.loc[~frame["pass_filters"]]
        ranked = pd.concat([passed_only, rejected], ignore_index=True)

    ranked["rank"] = None
    ranked["equal_weight_pct"] = None
    passers = ranked.index[ranked["pass_filters"]].tolist()
    for i, idx in enumerate(passers, start=1):
        ranked.at[idx, "rank"] = i

    k_ref = int(config.k)
    ew_pct = round(100.0 / k_ref, 2) if k_ref > 0 else None
    min_cred = float(config.min_credibility)
    suggested = 0
    for idx in passers:
        if suggested >= k_ref:
            break
        score = ranked.at[idx, "pick_score"]
        if score is None or (isinstance(score, float) and not math.isfinite(score)):
            continue
        if min_cred > 0:
            cred = ranked.at[idx, "combined_cred"]
            cred_f = _to_float(cred)
            if cred_f is None or cred_f < min_cred:
                continue
        ranked.at[idx, "equal_weight_pct"] = ew_pct
        suggested += 1

    return ranked.drop(columns=["_sort_pass"]).reindex(columns=list(BUYABLE_COLUMNS))


def format_buyable_rank_report(
    frame: pd.DataFrame,
    *,
    as_of: str,
    pack_dir: str,
    pool_risk_source: str | None = None,
    cfg: BuyableRankConfig | None = None,
    rank_rule: str | None = None,
    gate: str | None = None,
    strategy_name: str | None = None,
) -> str:
    config = cfg or BuyableRankConfig()
    n_total = len(frame)
    passers = frame.loc[frame["pass_filters"]] if not frame.empty else frame
    n_pass = len(passers)
    n_reject = n_total - n_pass

    score_note = (
        f"按 `rank_rule={rank_rule}` 重算 `pick_score` 后排序"
        if rank_rule
        else "按 `tomorrow_topk_picks.pick_score` 排序"
    )
    lines = [
        "# Buyable Rank",
        "",
        f"**As of:** {as_of}",
        f"**Pack dir:** `{pack_dir}`",
        f"**Universe (from tomorrow topk):** {n_total}",
        f"**Passed filters:** {n_pass}",
        f"**Rejected:** {n_reject}",
        "",
        f"硬过滤后{score_note}；非新预测模型。",
        "",
        "### 过滤规则",
        "",
    ]
    if strategy_name:
        lines.append(f"- strategy: `{strategy_name}`")
    if gate:
        lines.append(f"- gate: `{gate}`")
    if rank_rule:
        lines.append(f"- rank_rule: `{rank_rule}`")
    lines.extend(
        [
            f"- exclude_avoid_leg: `{config.exclude_avoid_leg}`",
            f"- exclude_broken_path (状态=破位): `{config.exclude_broken_path}`",
            f"- trend flags: `{', '.join(config.trend_reject_flags) if config.exclude_trend_flags else 'off'}`",
            f"- jump flags: `{', '.join(config.jump_reject_flags) if config.exclude_jump_flags else 'off'}`",
            f"- min_credibility (等权参考): `{config.min_credibility}`",
            f"- equal-weight top K: `{config.k}`",
        ]
    )
    if pool_risk_source:
        lines.append(f"- pool_risk source: `{pool_risk_source}`")
    lines.extend(["", "## 可买 Top（过门禁）", ""])

    if passers.empty:
        lines.append("_无标的通过硬过滤。_")
    else:
        lines.extend(
            [
                "| rank | code | name | leg | 方向偏 | pick_score | 等权% |",
                "|------|------|------|-----|--------|------------|-------|",
            ]
        )
        show = passers.head(max(int(config.k), 15))
        for _, row in show.iterrows():
            ew = row.get("equal_weight_pct")
            ew_s = "" if ew is None or (isinstance(ew, float) and math.isnan(ew)) else str(ew)
            bias = row.get("direction_bias")
            bias_s = "—" if bias is None else f"{float(bias):.4f}"
            score = row.get("pick_score")
            score_s = "—" if score is None else f"{float(score):.3e}"
            lines.append(
                f"| {int(row['rank'])} | {row['code']} | {row['name']} | {row['leg']} | "
                f"{bias_s} | {score_s} | {ew_s} |"
            )

    rejected = frame.loc[~frame["pass_filters"]] if not frame.empty else frame
    if not rejected.empty:
        lines.extend(["", "## 被拒样例（前 10）", ""])
        lines.extend(
            [
                "| code | name | reject_reason |",
                "|------|------|---------------|",
            ]
        )
        for _, row in rejected.head(10).iterrows():
            lines.append(f"| {row['code']} | {row['name']} | {row['reject_reason']} |")

    lines.extend(
        [
            "",
            "*由 `decision_pack/scripts/generate_buyable_rank.py` 生成；"
            "先看趋势/肥尾/急跌门禁，再用明日 pick_score 排序。*",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BUYABLE_COLUMNS",
    "BUYABLE_RANK_REPORT",
    "BUYABLE_TOPK_CSV",
    "BuyableRankConfig",
    "GateName",
    "build_buyable_frame",
    "buyable_rank_config_from_raw",
    "evaluate_buyable_row",
    "format_buyable_rank_report",
    "gate_to_buyable_config",
    "load_best_strategy_spec",
    "load_pack_buyable_inputs",
    "parse_upside_skew",
    "recompute_tomorrow_topk_scores",
    "resolve_pool_risk_path",
]
