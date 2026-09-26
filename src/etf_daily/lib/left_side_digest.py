"""Left-side (vol-compression + spike) candidate digest — symmetric to tomorrow_digest."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from etf_daily.lib.garch_dynamics import GarchFitResult, fit_garch_dynamics
from etf_daily.lib.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
)
from etf_daily.lib.hmm_short_horizon import (
    HmmFitResult,
    HmmShortHorizonConfig,
    HmmShortHorizonMetrics,
    compute_hmm_short_horizon_metrics,
    fit_hmm_2state,
)
from etf_daily.lib.fat_tail_risk import slice_log_returns
from etf_daily.lib.indicators import slice_through
from etf_daily.lib.price_jump_flags import drop_split_log_returns
from etf_daily.lib.tomorrow_digest import (
    CombinedMode,
    TomorrowDigestConfig,
    calibration_quality_row,
    combined_credibility,
    credibility_today,
    format_cred,
    format_pct,
    log_return_to_pct,
    pool_median_calibration,
)

LEFT_SIDE_DIGEST_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "close",
    "as_of",
    "vol5_pct_120d",
    "p_vol_spike_h5",
    "触轨↓",
    "q05_H",
    "compression",
    "transition_score",
    "regime_now",
    "跃迁方向",
    "综合可信度",
    "可靠",
    "依据",
)

LEFT_SIDE_TOPK_COLUMNS: tuple[str, ...] = (
    "rank",
    "code",
    "name",
    "close",
    "as_of",
    "barbell_leg",
    "vol5_pct_120d",
    "p_vol_spike_h5",
    "触轨↓",
    "q05_H",
    "left_score",
    "equal_weight_pct",
)

LEFT_SIDE_INTERNAL_COLUMNS: tuple[str, ...] = (
    "_compression",
    "_combined_sort",
    "_vol5_pct",
    "_p_spike",
    "_p_down",
    "_transition_score",
    "_transition_cred",
)

REGIME_TRANSITION_CSV = "cluster_mapping_selected_regime_transition.csv"

RankRule = Literal[
    "compression_x_spike",
    "compression_x_spike_x_down",
    "transition_x_spike_x_down",
]


@dataclass(frozen=True)
class LeftSideTopKConfig:
    enabled: bool = True
    k: int = 8
    rank_all: bool = True
    rank_rule: RankRule = "compression_x_spike_x_down"
    min_credibility: float = 0.003
    require_touch_down_min: float = 0.15
    max_vol5_pct: float = 0.35
    exclude_avoid_leg: bool = False
    exclude_jump_flags: bool = True
    candidates_only: bool = False


@dataclass(frozen=True)
class LeftSideDigestConfig:
    horizon: int = 5
    calibration_min_n_obs: int = 10
    brier_random: float = 0.67
    brier_good: float = 0.33
    band_tol: float = 0.20
    shrink_n_obs: int = 20
    combined_mode: CombinedMode = "min"
    candidates_only: bool = False
    topk: LeftSideTopKConfig = field(default_factory=LeftSideTopKConfig)


def left_side_digest_config_from_raw(raw: dict[str, Any] | None) -> LeftSideDigestConfig:
    if not raw or not isinstance(raw, dict):
        return LeftSideDigestConfig()
    kwargs: dict[str, Any] = {}
    for key in LeftSideDigestConfig.__dataclass_fields__:
        if key == "topk":
            topk_raw = raw.get("topk")
            if topk_raw and isinstance(topk_raw, dict):
                topk_kwargs = {
                    k: v
                    for k, v in topk_raw.items()
                    if k in LeftSideTopKConfig.__dataclass_fields__ and v is not None
                }
                kwargs["topk"] = LeftSideTopKConfig(**topk_kwargs)
            continue
        if key in raw and raw[key] is not None:
            kwargs[key] = raw[key]
    return LeftSideDigestConfig(**kwargs)


def _calibration_cfg(digest_cfg: LeftSideDigestConfig) -> TomorrowDigestConfig:
    return TomorrowDigestConfig(
        calibration_min_n_obs=digest_cfg.calibration_min_n_obs,
        brier_random=digest_cfg.brier_random,
        brier_good=digest_cfg.brier_good,
        band_tol=digest_cfg.band_tol,
        shrink_n_obs=digest_cfg.shrink_n_obs,
        combined_mode=digest_cfg.combined_mode,
    )


def compute_compression(vol5_pct: float | None) -> float | None:
    if vol5_pct is None:
        return None
    if not math.isfinite(float(vol5_pct)):
        return float("nan")
    return 1.0 - float(vol5_pct)


def compute_left_score(
    compression: float | None,
    p_spike: float | None,
    p_down: float | None,
    combined_cred: float | None,
    *,
    rank_rule: RankRule,
    transition_score: float | None = None,
) -> float:
    if rank_rule == "transition_x_spike_x_down":
        if transition_score is None or p_spike is None or combined_cred is None:
            return float("nan")
        if not all(
            math.isfinite(float(x))
            for x in (transition_score, p_spike, combined_cred)
        ):
            return float("nan")
        score = float(transition_score) * float(p_spike) * float(combined_cred)
        if p_down is None or not math.isfinite(float(p_down)):
            return float("nan")
        return score * float(p_down)

    if compression is None or p_spike is None or combined_cred is None:
        return float("nan")
    if not all(math.isfinite(float(x)) for x in (compression, p_spike, combined_cred)):
        return float("nan")
    score = float(compression) * float(p_spike) * float(combined_cred)
    if rank_rule == "compression_x_spike_x_down":
        if p_down is None or not math.isfinite(float(p_down)):
            return float("nan")
        score *= float(p_down)
    return score


def transition_credibility_today(
    transition_score: float | None,
    *,
    reliable: bool,
    regime_separated: bool,
    cal_quality: float,
) -> float:
    if not reliable or not regime_separated or transition_score is None:
        return 0.0
    if not math.isfinite(float(transition_score)):
        return 0.0
    return float(np.clip(float(transition_score), 0.0, 1.0) * cal_quality)


def load_regime_transition_frame(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path).expanduser()
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _hmm_reliable(metrics: HmmShortHorizonMetrics) -> bool:
    return bool(metrics.reliable and metrics.regime_separated)


def _parse_jump_flags(flags_val: Any) -> set[str]:
    if isinstance(flags_val, str) and flags_val:
        return {f for f in flags_val.split("|") if f}
    if isinstance(flags_val, (list, tuple)):
        return {str(f) for f in flags_val if f}
    return set()


def is_jump_excluded(
    code: str,
    jump_frame: pd.DataFrame | None,
    *,
    exclude_jump_flags: bool,
) -> bool:
    if not exclude_jump_flags or jump_frame is None or jump_frame.empty:
        return False
    if "code" not in jump_frame.columns:
        return False
    jump_idx = jump_frame.set_index("code")
    if code not in jump_idx.index:
        return False
    bad = {"live_price_gap", "recent_jump"}
    return bool(bad & _parse_jump_flags(jump_idx.loc[code].get("flags")))


def _transition_basis_note(
    transition_score: float | None,
    prev_regime: str | None,
    regime_now: str | None,
) -> str:
    if transition_score is None or regime_now is None or regime_now in ("—", ""):
        return ""
    if not math.isfinite(float(transition_score)):
        return ""
    score_s = f"{float(transition_score):.2f}"
    if prev_regime and prev_regime not in ("—", "") and prev_regime != regime_now:
        return f"HMM3跃迁={score_s}·{prev_regime}→{regime_now}"
    return f"HMM3跃迁={score_s}·{regime_now}"


def build_left_side_digest_frame(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    digest_cfg: LeftSideDigestConfig,
    garch_board_cfg: GarchShortHorizonConfig,
    hmm_board_cfg: HmmShortHorizonConfig,
    garch_cal: pd.DataFrame | None = None,
    hmm_cal: pd.DataFrame | None = None,
    transition_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if cluster_frame.empty:
        return pd.DataFrame(columns=list(LEFT_SIDE_DIGEST_COLUMNS))

    end = pd.Timestamp(as_of)
    horizon = int(digest_cfg.horizon)
    cal_cfg = _calibration_cfg(digest_cfg)

    garch_cal = garch_cal if garch_cal is not None else pd.DataFrame()
    hmm_cal = hmm_cal if hmm_cal is not None else pd.DataFrame()
    garch_cal_idx = (
        garch_cal.set_index("code") if not garch_cal.empty and "code" in garch_cal.columns else None
    )
    hmm_cal_idx = (
        hmm_cal.set_index("code") if not hmm_cal.empty and "code" in hmm_cal.columns else None
    )
    garch_pool = pool_median_calibration(garch_cal, cfg=cal_cfg)
    hmm_pool = pool_median_calibration(hmm_cal, cfg=cal_cfg)
    pooled = garch_cal.empty and hmm_cal.empty

    transition_idx = (
        transition_frame.set_index("code")
        if transition_frame is not None
        and not transition_frame.empty
        and "code" in transition_frame.columns
        else None
    )
    transition_codes = set(transition_idx.index.astype(str)) if transition_idx is not None else set()

    g_fit_cache: dict[str, GarchFitResult] = {}
    h_fit_cache: dict[str, HmmFitResult] = {}
    gcfg = garch_board_cfg.garch
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
        if code in transition_codes:
            continue
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

        t_row = (
            transition_idx.loc[code]
            if transition_idx is not None and code in transition_idx.index
            else None
        )
        use_transition = t_row is not None

        if close_val is not None and code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            g_metrics = compute_garch_short_horizon_metrics(
                series,
                end,
                horizon=horizon,
                cfg=garch_board_cfg,
                flags=flag_tuple,
                fit=g_fit_cache.get(code),
                max_horizon=horizon,
            )
            if use_transition:
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
                    basis="HMM3跃迁板",
                    regime_separated=False,
                )
            else:
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
        g_cal_q, g_n = calibration_quality_row(g_cal_row, pool_median=garch_pool, cfg=cal_cfg)
        h_cal_q, h_n = calibration_quality_row(h_cal_row, pool_median=hmm_pool, cfg=cal_cfg)

        g_cred = credibility_today(
            p_up=g_metrics.p_up,
            p_down=g_metrics.p_down,
            reliable=g_metrics.reliable,
            cal_quality=g_cal_q,
        )

        t_score_raw: float | None = None
        t_cred_val: float | None = None
        regime_now_disp = "—"
        direction_disp = "—"
        prev_regime: str | None = None
        t_reliable = False
        t_sep = False

        if use_transition and t_row is not None:
            raw_score = t_row.get("transition_score")
            if raw_score is not None and pd.notna(raw_score) and math.isfinite(float(raw_score)):
                t_score_raw = float(raw_score)
            regime_now_disp = str(t_row.get("regime_now") or "—")
            direction_disp = str(t_row.get("transition_direction") or "—")
            prev_raw = t_row.get("prev_regime")
            if prev_raw is not None and pd.notna(prev_raw):
                prev_regime = str(prev_raw)
            t_reliable = bool(t_row.get("reliable", False))
            t_sep = bool(t_row.get("regime_separated", False))
            t_cred = transition_credibility_today(
                t_score_raw,
                reliable=t_reliable,
                regime_separated=t_sep,
                cal_quality=h_cal_q,
            )
            t_cred_val = t_cred
            combined, reliability_label = combined_credibility(
                g_cred,
                0.0,
                garch_reliable=g_metrics.reliable,
                hmm_reliable=False,
                transition_cred=t_cred,
                transition_reliable=t_reliable and t_sep,
                mode=digest_cfg.combined_mode,
            )
            hmm_sep_for_basis = t_sep
        else:
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
            hmm_sep_for_basis = h_metrics.regime_separated

        vol5_pct = g_metrics.vol5_pct_120d
        compression = compute_compression(vol5_pct)

        basis = _basis_note(
            horizon=horizon,
            garch_n=g_n,
            hmm_n=h_n,
            garch_dgp=g_metrics.dgp_basis or "—",
            hmm_sep=hmm_sep_for_basis,
            pooled=pooled,
        )
        t_note = _transition_basis_note(t_score_raw, prev_regime, regime_now_disp)
        if t_note:
            basis = f"{basis}·{t_note}"

        rows.append(
            {
                "code": code,
                "name": name,
                "close": round(close_val, 4) if close_val is not None else None,
                "as_of": as_of,
                "vol5_pct_120d": format_pct(
                    float(vol5_pct) * 100.0
                    if vol5_pct is not None and math.isfinite(float(vol5_pct))
                    else None
                ),
                "p_vol_spike_h5": format_pct(
                    float(g_metrics.p_vol_spike_h5) * 100.0
                    if g_metrics.p_vol_spike_h5 is not None
                    and math.isfinite(float(g_metrics.p_vol_spike_h5))
                    else None
                ),
                "触轨↓": format_pct(
                    float(g_metrics.p_down) * 100.0
                    if g_metrics.p_down is not None and math.isfinite(float(g_metrics.p_down))
                    else None
                ),
                "q05_H": format_pct(log_return_to_pct(g_metrics.q05)),
                "compression": format_cred(compression),
                "transition_score": format_cred(t_score_raw),
                "regime_now": regime_now_disp,
                "跃迁方向": direction_disp,
                "综合可信度": format_cred(combined),
                "可靠": reliability_label,
                "依据": basis,
                "_compression": compression,
                "_combined_sort": combined,
                "_vol5_pct": vol5_pct,
                "_p_spike": g_metrics.p_vol_spike_h5,
                "_p_down": g_metrics.p_down,
                "_transition_score": t_score_raw,
                "_transition_cred": t_cred_val,
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(LEFT_SIDE_DIGEST_COLUMNS))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("_combined_sort", ascending=False, kind="stable")
    return frame


def _basis_note(
    *,
    horizon: int,
    garch_n: int,
    hmm_n: int,
    garch_dgp: str,
    hmm_sep: bool,
    pooled: bool,
) -> str:
    parts = [f"H={horizon}", "左侧波动"]
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


def _passes_left_suggested_pick(
    *,
    vol5_pct: float | None,
    p_down: float | None,
    combined: float | None,
    score: float,
    topk_cfg: LeftSideTopKConfig,
    jump_excluded: bool,
    rank_rule: RankRule,
) -> bool:
    if not math.isfinite(score):
        return False
    if combined is None or float(combined) < float(topk_cfg.min_credibility):
        return False
    if jump_excluded and topk_cfg.exclude_jump_flags:
        return False
    if rank_rule != "transition_x_spike_x_down":
        if vol5_pct is not None and math.isfinite(float(vol5_pct)):
            if float(vol5_pct) > float(topk_cfg.max_vol5_pct):
                return False
    if p_down is not None and math.isfinite(float(p_down)):
        if float(p_down) < float(topk_cfg.require_touch_down_min):
            return False
    return True


def select_left_topk_picks(
    digest_frame: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    *,
    digest_cfg: LeftSideDigestConfig,
    jump_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    topk_cfg = digest_cfg.topk
    if not topk_cfg.enabled or digest_frame.empty:
        return pd.DataFrame(columns=list(LEFT_SIDE_TOPK_COLUMNS))

    cluster_by_code = (
        cluster_frame.set_index("code") if not cluster_frame.empty else None
    )
    candidates: list[dict[str, Any]] = []
    for _, row in digest_frame.iterrows():
        code = str(row["code"])
        vol5_pct = row.get("_vol5_pct")
        p_down = row.get("_p_down")

        if topk_cfg.candidates_only and cluster_by_code is not None and code in cluster_by_code.index:
            weight = float(cluster_by_code.loc[code].get("weight", 0.0))
            if weight > 0:
                continue

        barbell_leg = "—"
        if cluster_by_code is not None and code in cluster_by_code.index:
            barbell_leg = str(cluster_by_code.loc[code].get("barbell_leg") or "—")

        if topk_cfg.exclude_avoid_leg and barbell_leg == "avoid":
            continue

        jump_excluded = is_jump_excluded(
            code,
            jump_frame,
            exclude_jump_flags=topk_cfg.exclude_jump_flags,
        )

        compression = row.get("_compression")
        p_spike = row.get("_p_spike")
        combined = row.get("_combined_sort")
        transition_score = row.get("_transition_score")
        rank_cred = combined
        if topk_cfg.rank_rule == "transition_x_spike_x_down":
            t_cred = row.get("_transition_cred")
            if t_cred is not None and math.isfinite(float(t_cred)) and float(t_cred) > 0:
                rank_cred = float(t_cred)
        score = compute_left_score(
            compression,
            p_spike,
            p_down,
            rank_cred,
            rank_rule=topk_cfg.rank_rule,
            transition_score=transition_score,
        )
        pick_score_val = round(score, 8) if math.isfinite(score) else None

        candidates.append(
            {
                "code": code,
                "name": row["name"],
                "close": row["close"],
                "as_of": row["as_of"],
                "barbell_leg": barbell_leg,
                "vol5_pct_120d": row["vol5_pct_120d"],
                "p_vol_spike_h5": row["p_vol_spike_h5"],
                "触轨↓": row["触轨↓"],
                "q05_H": row["q05_H"],
                "left_score": pick_score_val,
                "_compression": compression,
                "_combined_sort": combined,
                "_vol5_pct": vol5_pct,
                "_p_spike": p_spike,
                "_p_down": p_down,
                "_jump_excluded": jump_excluded,
            }
        )

    if not candidates:
        return pd.DataFrame(columns=list(LEFT_SIDE_TOPK_COLUMNS))

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(
        ["left_score", "code"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    )
    ranked = cand_df.copy() if topk_cfg.rank_all else cand_df.head(int(topk_cfg.k)).copy()

    n_rows = len(ranked)
    if n_rows == 0:
        return pd.DataFrame(columns=list(LEFT_SIDE_TOPK_COLUMNS))

    ranked["rank"] = range(1, n_rows + 1)
    ranked["equal_weight_pct"] = None
    k_ref = int(topk_cfg.k)
    ew_pct = round(100.0 / k_ref, 2) if k_ref > 0 else None
    suggested = 0
    for idx in ranked.index:
        if suggested >= k_ref:
            break
        row = ranked.loc[idx]
        score = row.get("left_score")
        if score is None or not math.isfinite(float(score)):
            continue
        if not _passes_left_suggested_pick(
            vol5_pct=row.get("_vol5_pct"),
            p_down=row.get("_p_down"),
            combined=rank_cred,
            score=float(score),
            topk_cfg=topk_cfg,
            jump_excluded=bool(row.get("_jump_excluded")),
            rank_rule=topk_cfg.rank_rule,
        ):
            continue
        ranked.at[idx, "equal_weight_pct"] = ew_pct
        suggested += 1

    drop_cols = [c for c in (*LEFT_SIDE_INTERNAL_COLUMNS, "_jump_excluded") if c in ranked.columns]
    return ranked.drop(columns=drop_cols, errors="ignore").reindex(
        columns=list(LEFT_SIDE_TOPK_COLUMNS)
    )


def format_left_side_digest_report(
    frame: pd.DataFrame,
    *,
    as_of: str,
    pack_dir: str,
    pooled_warning: bool = False,
    topk_frame: pd.DataFrame | None = None,
    topk_cfg: LeftSideTopKConfig | None = None,
) -> str:
    lines = [
        "# Left-Side Vol Digest (H=5)",
        "",
        f"**As of:** {as_of}  ",
        f"**Pack dir:** `{pack_dir}`  ",
        f"**Candidates:** {len(frame)}  ",
        "",
        "左侧尺子：**5 日波动历史分位（压缩）** × **GARCH MC 波动放大概率** × **触轨↓**；",
        "非点位预测，非自动下单。",
        "",
    ]
    if pooled_warning:
        lines.extend(
            [
                "> 未找到 H=5 校准缓存，已用全池中位校准。",
                "> 请运行：`validate_garch_short_horizon_board.py --horizon 5 --output-cache ...`",
                "",
            ]
        )
    lines.extend(["## Top 10 by 综合可信度", ""])
    if frame.empty:
        lines.append("_无候选标的_")
    else:
        export = frame.reindex(columns=list(LEFT_SIDE_DIGEST_COLUMNS))
        lines.append(
            "| code | vol5分位 | spike% | 触轨↓ | compression | 综合可信度 |"
        )
        lines.append("|------|----------|--------|-------|-------------|------------|")
        for _, row in export.head(10).iterrows():
            lines.append(
                f"| {row['code']} | {row['vol5_pct_120d']} | {row['p_vol_spike_h5']} | "
                f"{row['触轨↓']} | {row['compression']} | {row['综合可信度']} |"
            )

    if topk_frame is not None and not topk_frame.empty and topk_cfg is not None:
        rule = topk_cfg.rank_rule
        gate_note = (
            f"触轨↓≥{topk_cfg.require_touch_down_min:.0%} 等门槛"
            if rule == "transition_x_spike_x_down"
            else (
                f"满足 vol5≤{topk_cfg.max_vol5_pct:.0%}、"
                f"触轨↓≥{topk_cfg.require_touch_down_min:.0%} 等门槛"
            )
        )
        lines.extend(
            [
                "",
                f"## Left Top-K（{rule}）",
                "",
                f"等权参考前 **{topk_cfg.k}** 只（{gate_note}；全池排名见 `left_side_topk_picks.csv`）。",
                "",
            ]
        )
        show_n = min(8, len(topk_frame))
        for _, row in topk_frame.head(show_n).iterrows():
            ew = row.get("equal_weight_pct")
            ew_s = f"{ew}%" if ew is not None and math.isfinite(float(ew)) else "—"
            lines.append(
                f"- **{row['rank']}. {row['code']}** {row['name']} | "
                f"left_score={row['left_score']} | vol5={row['vol5_pct_120d']} | "
                f"spike={row['p_vol_spike_h5']} | 等权={ew_s}"
            )
        if len(topk_frame) > show_n:
            lines.append(f"_… 其余 {len(topk_frame) - show_n} 只见 `left_side_topk_picks.csv`_")

    lines.extend(
        [
            "",
            "*由 `src/etf_daily/scripts/generate_left_side_digest.py` 生成。*",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "LEFT_SIDE_DIGEST_COLUMNS",
    "LEFT_SIDE_TOPK_COLUMNS",
    "REGIME_TRANSITION_CSV",
    "LeftSideDigestConfig",
    "LeftSideTopKConfig",
    "build_left_side_digest_frame",
    "compute_compression",
    "compute_left_score",
    "format_left_side_digest_report",
    "is_jump_excluded",
    "left_side_digest_config_from_raw",
    "load_regime_transition_frame",
    "select_left_topk_picks",
    "transition_credibility_today",
]
