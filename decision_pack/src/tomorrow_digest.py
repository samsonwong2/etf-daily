"""Tomorrow (H=1) candidate-pool digest: GARCH+HMM range + calibrated credibility."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from decision_pack.src.garch_dynamics import GarchFitResult, fit_garch_dynamics
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
    garch_short_horizon_config_from_raw,
)
from decision_pack.src.hmm_short_horizon import (
    HmmFitResult,
    HmmShortHorizonConfig,
    HmmShortHorizonMetrics,
    compute_hmm_short_horizon_metrics,
    fit_hmm_2state,
    hmm_short_horizon_config_from_raw,
)
from decision_pack.src.fat_tail_risk import slice_log_returns
from decision_pack.src.indicators import slice_through
from decision_pack.src.price_jump_flags import drop_split_log_returns

CALIBRATION_CACHE_COLUMNS: tuple[str, ...] = (
    "code",
    "n_obs",
    "brier",
    "coverage_q05",
    "coverage_q95",
    "model",
    "horizon",
)

TOMORROW_DIGEST_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "close",
    "as_of",
    "garch_跌5%",
    "garch_涨95%",
    "garch_方向偏",
    "garch_可信度",
    "hmm_跌5%",
    "hmm_涨95%",
    "hmm_方向偏",
    "hmm_可信度",
    "综合可信度",
    "可靠",
    "依据",
)

TOMORROW_TOPK_COLUMNS: tuple[str, ...] = (
    "rank",
    "code",
    "name",
    "close",
    "as_of",
    "barbell_leg",
    "direction_bias",
    "combined_cred",
    "garch_跌5%",
    "garch_涨95%",
    "upside_skew_pct",
    "pick_score",
    "equal_weight_pct",
)

TOMORROW_DIGEST_INTERNAL_COLUMNS: tuple[str, ...] = (
    "_g_dir",
    "_combined_sort",
    "_g_q05",
    "_g_q95",
    "_upside_skew",
)

CombinedMode = Literal["min", "mean"]
RankRule = Literal["dir_x_cred", "upside_x_cred", "composite"]


@dataclass(frozen=True)
class TopKConfig:
    enabled: bool = True
    k: int = 8
    min_credibility: float = 0.003
    min_direction: float = 0.0
    exclude_avoid_leg: bool = True
    candidates_only: bool = True
    rank_all: bool = True
    rank_rule: RankRule = "upside_x_cred"
    require_positive_direction: bool = True
    avoid_leg_penalty: float = 0.5


@dataclass(frozen=True)
class TomorrowDigestConfig:
    horizon: int = 1
    garch_barrier_mult: float = 1.0
    calibration_min_n_obs: int = 10
    brier_random: float = 0.67
    brier_good: float = 0.33
    band_tol: float = 0.20
    shrink_n_obs: int = 20
    combined_mode: CombinedMode = "min"
    candidates_only: bool = True
    topk: TopKConfig = field(default_factory=TopKConfig)


def tomorrow_digest_config_from_raw(raw: dict[str, Any] | None) -> TomorrowDigestConfig:
    if not raw or not isinstance(raw, dict):
        return TomorrowDigestConfig()
    kwargs: dict[str, Any] = {}
    for key in TomorrowDigestConfig.__dataclass_fields__:
        if key == "topk":
            topk_raw = raw.get("topk")
            if topk_raw and isinstance(topk_raw, dict):
                topk_kwargs = {
                    k: v
                    for k, v in topk_raw.items()
                    if k in TopKConfig.__dataclass_fields__ and v is not None
                }
                kwargs["topk"] = TopKConfig(**topk_kwargs)
            continue
        if key in raw and raw[key] is not None:
            kwargs[key] = raw[key]
    return TomorrowDigestConfig(**kwargs)


def compute_upside_skew(g_q05: float | None, g_q95: float | None) -> float | None:
    if g_q05 is None or g_q95 is None:
        return None
    if not math.isfinite(float(g_q05)) or not math.isfinite(float(g_q95)):
        return None
    return float(g_q95) - abs(float(g_q05))


def _base_pick_score(
    g_dir: float,
    combined_cred: float,
    upside_skew: float | None,
    *,
    rank_rule: RankRule,
) -> float:
    if rank_rule == "dir_x_cred":
        return g_dir * combined_cred
    if rank_rule == "upside_x_cred":
        if upside_skew is None or not math.isfinite(upside_skew):
            return float("nan")
        return upside_skew * combined_cred
    if rank_rule == "composite":
        if upside_skew is None or not math.isfinite(upside_skew):
            return float("nan")
        return max(0.0, g_dir) * combined_cred * upside_skew
    raise ValueError(f"unknown rank_rule: {rank_rule}")


def compute_pick_score(
    g_dir: float | None,
    combined_cred: float | None,
    *,
    topk_cfg: TopKConfig,
    upside_skew: float | None = None,
    for_ranking: bool = False,
) -> float:
    if g_dir is None or combined_cred is None:
        return float("nan")
    if not math.isfinite(float(g_dir)) or not math.isfinite(float(combined_cred)):
        return float("nan")
    if topk_cfg.require_positive_direction and float(g_dir) <= 0.0:
        return float("nan")
    score = _base_pick_score(
        float(g_dir),
        float(combined_cred),
        upside_skew,
        rank_rule=topk_cfg.rank_rule,
    )
    if not math.isfinite(score):
        return float("nan")
    if for_ranking:
        return score
    if float(g_dir) <= float(topk_cfg.min_direction):
        return float("nan")
    if float(combined_cred) < float(topk_cfg.min_credibility):
        return float("nan")
    return score


def _is_suggested_equal_weight_pick(
    g_dir: float | None,
    combined_cred: float | None,
    *,
    topk_cfg: TopKConfig,
    upside_skew: float | None = None,
) -> bool:
    score = compute_pick_score(
        g_dir,
        combined_cred,
        topk_cfg=topk_cfg,
        upside_skew=upside_skew,
        for_ranking=False,
    )
    return math.isfinite(score)


def select_topk_picks(
    digest_frame: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    *,
    digest_cfg: TomorrowDigestConfig,
) -> pd.DataFrame:
    topk_cfg = digest_cfg.topk
    if not topk_cfg.enabled or digest_frame.empty:
        return pd.DataFrame(columns=list(TOMORROW_TOPK_COLUMNS))

    cluster_by_code = (
        cluster_frame.set_index("code") if not cluster_frame.empty else None
    )
    candidates: list[dict[str, Any]] = []
    for _, row in digest_frame.iterrows():
        code = str(row["code"])
        g_dir = row.get("_g_dir")
        combined = row.get("_combined_sort")

        if topk_cfg.candidates_only and cluster_by_code is not None and code in cluster_by_code.index:
            weight = float(cluster_by_code.loc[code].get("weight", 0.0))
            if weight > 0:
                continue

        barbell_leg = "—"
        if cluster_by_code is not None and code in cluster_by_code.index:
            barbell_leg = str(cluster_by_code.loc[code].get("barbell_leg") or "—")

        if topk_cfg.exclude_avoid_leg and barbell_leg == "avoid":
            continue

        upside = row.get("_upside_skew")
        g_q05 = row.get("_g_q05")
        g_q95 = row.get("_g_q95")
        score = compute_pick_score(
            g_dir,
            combined,
            topk_cfg=topk_cfg,
            upside_skew=upside,
            for_ranking=True,
        )
        if (
            math.isfinite(score)
            and barbell_leg == "avoid"
            and topk_cfg.avoid_leg_penalty > 0.0
            and topk_cfg.avoid_leg_penalty < 1.0
        ):
            score *= float(topk_cfg.avoid_leg_penalty)
        pick_score_val = round(score, 8) if math.isfinite(score) else None

        candidates.append(
            {
                "code": code,
                "name": row["name"],
                "close": row["close"],
                "as_of": row["as_of"],
                "barbell_leg": barbell_leg,
                "direction_bias": round(float(g_dir), 6) if g_dir is not None else None,
                "combined_cred": round(float(combined), 6) if combined is not None else None,
                "garch_跌5%": format_pct(log_return_to_pct(g_q05)),
                "garch_涨95%": format_pct(log_return_to_pct(g_q95)),
                "upside_skew_pct": format_pct(
                    float(upside) * 100.0
                    if upside is not None and math.isfinite(float(upside))
                    else None
                ),
                "pick_score": pick_score_val,
                "_g_dir": g_dir,
                "_combined_sort": combined,
                "_upside_skew": upside,
            }
        )

    if not candidates:
        return pd.DataFrame(columns=list(TOMORROW_TOPK_COLUMNS))

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(
        ["pick_score", "code"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    )
    if topk_cfg.rank_all:
        ranked = cand_df.copy()
    else:
        ranked = cand_df.head(int(topk_cfg.k)).copy()

    n_rows = len(ranked)
    if n_rows == 0:
        return pd.DataFrame(columns=list(TOMORROW_TOPK_COLUMNS))

    ranked["rank"] = range(1, n_rows + 1)
    ranked["equal_weight_pct"] = None
    k_ref = int(topk_cfg.k)
    ew_pct = round(100.0 / k_ref, 2) if k_ref > 0 else None
    suggested = 0
    for idx in ranked.index:
        if suggested >= k_ref:
            break
        row = ranked.loc[idx]
        if _is_suggested_equal_weight_pick(
            row.get("_g_dir"),
            row.get("_combined_sort"),
            topk_cfg=topk_cfg,
            upside_skew=row.get("_upside_skew"),
        ):
            ranked.at[idx, "equal_weight_pct"] = ew_pct
            suggested += 1

    return ranked.drop(columns=["_g_dir", "_combined_sort", "_upside_skew"]).reindex(
        columns=list(TOMORROW_TOPK_COLUMNS)
    )


def log_return_to_pct(q: float | None) -> float | None:
    if q is None or not math.isfinite(q):
        return None
    return float(math.expm1(q) * 100.0)


def format_pct(value: float | None, *, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}%"


def format_cred(value: float | None, *, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def calibration_quality_row(
    row: pd.Series | dict[str, Any] | None,
    *,
    pool_median: float,
    cfg: TomorrowDigestConfig,
) -> tuple[float, int]:
    """Return (cal_quality in [0,1], n_obs). Uses pool_median when row missing."""
    if row is None or (isinstance(row, pd.Series) and row.empty):
        return pool_median, 0

    def _get(key: str, default: float = float("nan")) -> float:
        if isinstance(row, pd.Series):
            val = row.get(key, default)
        else:
            val = row.get(key, default)
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            return default
        return float(val)

    n_obs = int(_get("n_obs", 0))
    brier = _get("brier", cfg.brier_random)
    cov05 = _get("coverage_q05", 0.05)
    cov95 = _get("coverage_q95", 0.95)

    span = max(cfg.brier_random - cfg.brier_good, 1e-9)
    brier_qual = float(np.clip((cfg.brier_random - brier) / span, 0.0, 1.0))
    band_err = abs(cov05 - 0.05) + abs(cov95 - 0.95)
    band_qual = float(np.clip(1.0 - band_err / cfg.band_tol, 0.0, 1.0))
    raw_cal = 0.6 * brier_qual + 0.4 * band_qual

    shrink = min(1.0, n_obs / float(cfg.shrink_n_obs)) if cfg.shrink_n_obs > 0 else 1.0
    cal = shrink * raw_cal + (1.0 - shrink) * pool_median
    return float(np.clip(cal, 0.0, 1.0)), n_obs


def pool_median_calibration(
    cache: pd.DataFrame,
    *,
    cfg: TomorrowDigestConfig,
) -> float:
    if cache.empty:
        return 0.5
    vals = []
    for _, row in cache.iterrows():
        cal, _ = calibration_quality_row(row, pool_median=0.5, cfg=cfg)
        vals.append(cal)
    return float(np.median(vals)) if vals else 0.5


def credibility_today(
    *,
    p_up: float | None,
    p_down: float | None,
    reliable: bool,
    cal_quality: float,
) -> float:
    if not reliable or p_up is None or p_down is None:
        return 0.0
    conviction = abs(float(p_up) - float(p_down))
    return float(np.clip(conviction * cal_quality, 0.0, 1.0))


def combined_credibility(
    garch_cred: float,
    hmm_cred: float,
    *,
    garch_reliable: bool,
    hmm_reliable: bool,
    transition_cred: float | None = None,
    transition_reliable: bool = False,
    mode: CombinedMode = "min",
) -> tuple[float, str]:
    legs: list[tuple[float, str]] = []
    if garch_reliable:
        legs.append((garch_cred, "GARCH"))
    if hmm_reliable:
        legs.append((hmm_cred, "HMM"))
    if transition_cred is not None and transition_reliable:
        legs.append((transition_cred, "跃迁"))

    if not legs:
        return 0.0, "不可信"

    if mode == "mean":
        cred = sum(c for c, _ in legs) / len(legs)
    else:
        cred = min(c for c, _ in legs)

    names = [name for _, name in legs]
    if len(names) == 3:
        label = "三模可信"
    elif len(names) == 2 and set(names) == {"GARCH", "HMM"}:
        label = "双模可信"
    elif len(names) == 1:
        label = f"仅{names[0]}"
    else:
        label = "+".join(names)
    return cred, label


def write_calibration_cache(
    per_symbol: pd.DataFrame,
    path: Path,
    *,
    model: str,
    horizon: int,
) -> None:
    if per_symbol.empty:
        frame = pd.DataFrame(columns=list(CALIBRATION_CACHE_COLUMNS))
    else:
        frame = per_symbol.copy()
        frame["model"] = model
        frame["horizon"] = int(horizon)
        keep = [c for c in CALIBRATION_CACHE_COLUMNS if c in frame.columns]
        frame = frame.reindex(columns=keep)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_calibration_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(CALIBRATION_CACHE_COLUMNS))
    frame = pd.read_csv(path)
    return frame.reindex(columns=list(CALIBRATION_CACHE_COLUMNS))


def resolve_cache_path(base: Path, model: str, horizon: int) -> Path:
    return base / f"{model}_h{horizon}_per_symbol.csv"


def _garch_board_cfg_for_digest(
    board_cfg: GarchShortHorizonConfig,
    digest_cfg: TomorrowDigestConfig,
) -> GarchShortHorizonConfig:
    garch = replace(board_cfg.garch, barrier_mult=float(digest_cfg.garch_barrier_mult))
    return replace(board_cfg, garch=garch)


def _hmm_reliable(metrics: HmmShortHorizonMetrics) -> bool:
    return bool(metrics.reliable and metrics.regime_separated)


def _basis_note(
    *,
    horizon: int,
    garch_n: int,
    hmm_n: int,
    garch_dgp: str,
    hmm_sep: bool,
    pooled: bool,
) -> str:
    parts = [f"H={horizon}"]
    if pooled:
        parts.append("池校准")
    if garch_n > 0:
        parts.append(f"GARCHn={garch_n}")
    if hmm_n > 0:
        parts.append(f"HMMn={hmm_n}")
    if garch_dgp and garch_dgp != "—":
        parts.append(garch_dgp)
    if not hmm_sep:
        parts.append("HMM塌缩")
    return "·".join(parts)


def build_tomorrow_digest_frame(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    digest_cfg: TomorrowDigestConfig,
    garch_board_cfg: GarchShortHorizonConfig,
    hmm_board_cfg: HmmShortHorizonConfig,
    garch_cal: pd.DataFrame | None = None,
    hmm_cal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if cluster_frame.empty:
        return pd.DataFrame(columns=list(TOMORROW_DIGEST_COLUMNS))

    end = pd.Timestamp(as_of)
    horizon = int(digest_cfg.horizon)
    garch_cfg = _garch_board_cfg_for_digest(garch_board_cfg, digest_cfg)

    garch_cal = garch_cal if garch_cal is not None else pd.DataFrame()
    hmm_cal = hmm_cal if hmm_cal is not None else pd.DataFrame()
    garch_cal_idx = (
        garch_cal.set_index("code") if not garch_cal.empty and "code" in garch_cal.columns else None
    )
    hmm_cal_idx = (
        hmm_cal.set_index("code") if not hmm_cal.empty and "code" in hmm_cal.columns else None
    )
    garch_pool = pool_median_calibration(garch_cal, cfg=digest_cfg)
    hmm_pool = pool_median_calibration(hmm_cal, cfg=digest_cfg)
    pooled = garch_cal.empty and hmm_cal.empty

    g_fit_cache: dict[str, GarchFitResult] = {}
    h_fit_cache: dict[str, HmmFitResult] = {}
    gcfg = garch_cfg.garch
    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        if code not in close_panel.columns:
            continue
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if series.size < 20:
            continue
        g_returns = slice_log_returns(series, end, lookback_days=gcfg.train_window)
        if g_returns.size >= 20:
            g_fit_cache[code] = fit_garch_dynamics(g_returns, gcfg, max_horizon=horizon)
        if hmm_board_cfg.drop_split_returns:
            h_returns = drop_split_log_returns(
                series,
                end=end,
                lookback_days=hmm_board_cfg.train_window,
                split_threshold=hmm_board_cfg.split_threshold,
            )
        else:
            h_returns = slice_log_returns(series, end, lookback_days=hmm_board_cfg.train_window)
        if h_returns.size >= 20:
            h_fit_cache[code] = fit_hmm_2state(h_returns, hmm_board_cfg)

    rows: list[dict[str, Any]] = []
    for _, row in cluster_frame.iterrows():
        weight = float(row.get("weight", 0.0))
        holding = "持" if weight > 0 else "候选"
        if digest_cfg.candidates_only and holding != "候选":
            continue

        code = str(row["code"])
        name = str(row.get("name", code))
        flags = row.get("flags")
        if isinstance(flags, str) and flags:
            flag_tuple = tuple(f for f in flags.split("|") if f)
        elif isinstance(flags, (list, tuple)):
            flag_tuple = tuple(str(f) for f in flags)
        else:
            flag_tuple = ()

        close_val: float | None = None
        if code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            available = slice_through(end, series)
            if len(available):
                c = float(available.iloc[-1])
                if math.isfinite(c) and c > 0:
                    close_val = c

        if close_val is not None and code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            g_metrics = compute_garch_short_horizon_metrics(
                series,
                end,
                horizon=horizon,
                cfg=garch_cfg,
                flags=flag_tuple,
                fit=g_fit_cache.get(code),
                max_horizon=horizon,
            )
            h_metrics = compute_hmm_short_horizon_metrics(
                series,
                end,
                horizon=horizon,
                cfg=hmm_board_cfg,
                fit=h_fit_cache.get(code),
            )
        else:
            g_metrics = GarchShortHorizonMetrics(
                horizon=horizon,
                sigma_garch_1d=None,
                sigma_garch_h=None,
                vol_ann_5d=None,
                vol_ann_20d=None,
                vol_ratio_5_20=None,
                nu_t=None,
                arch_pvalue=None,
                p_up=None,
                p_down=None,
                p_timeout=None,
                barrier=None,
                q05=None,
                q50=None,
                q95=None,
                var95_cond=None,
                cvar95_cond=None,
                dgp_basis="none",
                reliable=False,
                basis="缺价格·仅展示",
            )
            h_metrics = HmmShortHorizonMetrics(
                horizon=horizon,
                p_high=None,
                p_low=None,
                regime_now="—",
                sigma_mix_1d=None,
                sigma_mix_h=None,
                mu_mix_1d=None,
                p_up=None,
                p_down=None,
                p_timeout=None,
                barrier=None,
                q05=None,
                q50=None,
                q95=None,
                reliable=False,
                basis="缺价格·仅展示",
                regime_separated=False,
            )

        g_cal_row = garch_cal_idx.loc[code] if garch_cal_idx is not None and code in garch_cal_idx.index else None
        h_cal_row = hmm_cal_idx.loc[code] if hmm_cal_idx is not None and code in hmm_cal_idx.index else None
        g_cal_q, g_n = calibration_quality_row(g_cal_row, pool_median=garch_pool, cfg=digest_cfg)
        h_cal_q, h_n = calibration_quality_row(h_cal_row, pool_median=hmm_pool, cfg=digest_cfg)

        g_cred = credibility_today(
            p_up=g_metrics.p_up,
            p_down=g_metrics.p_down,
            reliable=g_metrics.reliable,
            cal_quality=g_cal_q,
        )
        h_cred = credibility_today(
            p_up=h_metrics.p_up,
            p_down=h_metrics.p_down,
            reliable=_hmm_reliable(h_metrics),
            cal_quality=h_cal_q,
        )
        combined, reliability_label = combined_credibility(
            g_cred,
            h_cred,
            garch_reliable=g_metrics.reliable,
            hmm_reliable=_hmm_reliable(h_metrics),
            mode=digest_cfg.combined_mode,
        )

        g_dir = None
        if g_metrics.p_up is not None and g_metrics.p_down is not None:
            g_dir = float(g_metrics.p_up) - float(g_metrics.p_down)
        h_dir = None
        if h_metrics.p_up is not None and h_metrics.p_down is not None:
            h_dir = float(h_metrics.p_up) - float(h_metrics.p_down)
        upside_skew = compute_upside_skew(g_metrics.q05, g_metrics.q95)

        rows.append(
            {
                "code": code,
                "name": name,
                "close": round(close_val, 4) if close_val is not None else None,
                "as_of": as_of,
                "garch_跌5%": format_pct(log_return_to_pct(g_metrics.q05)),
                "garch_涨95%": format_pct(log_return_to_pct(g_metrics.q95)),
                "garch_方向偏": format_cred(g_dir),
                "garch_可信度": format_cred(g_cred),
                "hmm_跌5%": format_pct(log_return_to_pct(h_metrics.q05)),
                "hmm_涨95%": format_pct(log_return_to_pct(h_metrics.q95)),
                "hmm_方向偏": format_cred(h_dir),
                "hmm_可信度": format_cred(h_cred),
                "综合可信度": format_cred(combined),
                "可靠": reliability_label,
                "依据": _basis_note(
                    horizon=horizon,
                    garch_n=g_n,
                    hmm_n=h_n,
                    garch_dgp=g_metrics.dgp_basis or "—",
                    hmm_sep=h_metrics.regime_separated,
                    pooled=pooled,
                ),
                "_combined_sort": combined,
                "_g_dir": g_dir,
                "_g_q05": g_metrics.q05,
                "_g_q95": g_metrics.q95,
                "_upside_skew": upside_skew,
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(TOMORROW_DIGEST_COLUMNS))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("_combined_sort", ascending=False, kind="stable")
    return frame


def format_tomorrow_digest_report(
    frame: pd.DataFrame,
    *,
    as_of: str,
    pack_dir: str,
    pooled_warning: bool = False,
    topk_frame: pd.DataFrame | None = None,
    topk_cfg: TopKConfig | None = None,
) -> str:
    lines = [
        "# Tomorrow Candidates Digest (H=1)",
        "",
        f"**As of:** {as_of}  ",
        f"**Pack dir:** `{pack_dir}`  ",
        f"**Candidates:** {len(frame)}  ",
        "",
        "明日 **90% 区间**（q05~q95）与 **walk-forward 校准可信度**；"
        "非点位预测，非交易 PnL 回测。",
        "",
    ]
    if pooled_warning:
        lines.extend(
            [
                "> 未找到 H=1 校准缓存，已用全池中位校准。",
                "> 请运行：`validate_garch_short_horizon_board.py --horizon 1 --output-cache ...`",
                "> 与 `validate_hmm_short_horizon_board.py --horizon 1 --output-cache ...`",
                "",
            ]
        )
    lines.extend(["## Top 10 by 综合可信度", ""])
    if frame.empty:
        lines.append("_无候选标的_")
    else:
        export = frame.reindex(columns=list(TOMORROW_DIGEST_COLUMNS))
        lines.append(
            "| code | name | close | garch区间 | hmm区间 | 综合可信度 | 可靠 |"
        )
        lines.append("|------|------|-------|-----------|---------|------------|------|")
        for _, row in export.head(10).iterrows():
            g_rng = f"{row['garch_跌5%']}~{row['garch_涨95%']}"
            h_rng = f"{row['hmm_跌5%']}~{row['hmm_涨95%']}"
            lines.append(
                f"| {row['code']} | {row['name']} | {row['close']} | "
                f"{g_rng} | {h_rng} | {row['综合可信度']} | {row['可靠']} |"
            )
    if topk_frame is not None and topk_cfg is not None and topk_cfg.enabled:
        rule_desc = {
            "dir_x_cred": "garch_方向偏 × 综合可信度",
            "upside_skew_x_cred": "upside_skew × 综合可信度",
            "upside_x_cred": "upside_skew × 综合可信度",
            "composite": "max(0,方向偏) × 综合可信度 × upside_skew",
        }
        formula = rule_desc.get(topk_cfg.rank_rule, topk_cfg.rank_rule)
        lines.extend(["", f"## 明日选股排名（{formula}）", ""])
        n_suggested = int(topk_frame["equal_weight_pct"].notna().sum()) if not topk_frame.empty else 0
        gate = "方向偏>0" if topk_cfg.require_positive_direction else f"方向偏>{topk_cfg.min_direction}"
        lines.append(
            f"选股分 = **{formula}**（`rank_rule={topk_cfg.rank_rule}`，过滤 {gate}）；"
            f"CSV 共 **{len(topk_frame)}** 只按分降序。"
            f"等权参考标注前 **{topk_cfg.k}** 只（满足门槛 **{n_suggested}** 只），"
            f"每只 **{round(100.0 / max(topk_cfg.k, 1), 2)}%**。"
            "非预测收益率，仅供决策参考。"
        )
        lines.append("")
        if topk_frame.empty:
            lines.append("_无可排名标的_")
        else:
            lines.append(
                "| rank | code | name | close | leg | 方向偏 | 综合可信度 | 选股分 | 等权% |"
            )
            lines.append(
                "|------|------|------|-------|-----|--------|------------|--------|-------|"
            )
            show_n = len(topk_frame) if not topk_cfg.rank_all else min(15, len(topk_frame))
            for _, row in topk_frame.head(show_n).iterrows():
                ew = row["equal_weight_pct"]
                ew_str = "" if ew is None or (isinstance(ew, float) and not math.isfinite(ew)) else ew
                lines.append(
                    f"| {row['rank']} | {row['code']} | {row['name']} | {row['close']} | "
                    f"{row['barbell_leg']} | {row['direction_bias']} | {row['combined_cred']} | "
                    f"{row['pick_score']} | {ew_str} |"
                )
            if topk_cfg.rank_all and len(topk_frame) > show_n:
                lines.append("")
                lines.append(f"_… 其余 {len(topk_frame) - show_n} 只见 `tomorrow_topk_picks.csv`_")
    lines.append("")
    lines.append(
        "*由 `decision_pack/scripts/generate_tomorrow_digest.py` 生成；"
        "主 GARCH/HMM h5/h10/h20 审计板列不变。*"
    )
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_CACHE_COLUMNS",
    "TOMORROW_DIGEST_COLUMNS",
    "TOMORROW_DIGEST_INTERNAL_COLUMNS",
    "TOMORROW_TOPK_COLUMNS",
    "TopKConfig",
    "TomorrowDigestConfig",
    "build_tomorrow_digest_frame",
    "calibration_quality_row",
    "combined_credibility",
    "RankRule",
    "compute_pick_score",
    "compute_upside_skew",
    "credibility_today",
    "format_tomorrow_digest_report",
    "load_calibration_cache",
    "log_return_to_pct",
    "pool_median_calibration",
    "resolve_cache_path",
    "select_topk_picks",
    "tomorrow_digest_config_from_raw",
    "write_calibration_cache",
]
