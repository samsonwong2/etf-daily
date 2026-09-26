"""Fat-tail portfolio generation and risk control (Taleb framework §8, §9, §10.1).

This module is a *parallel, non-binding* analysis/proposal layer on top of the
per-asset fat-tail metrics in ``fat_tail_risk.py``. It deliberately does NOT use
covariance / Markowitz / Sharpe (the framework rejects them for fat tails). It
provides:

- mean as a weak signal (plug-in / shadow mean gap, §3.1 / §11.2);
- barbell weight generation: kappa-defined breadth + closed-form left-tail-budget
  w-solve (§8);
- risk control: hard stop K on the total, crash correlation->1 CVaR (§10.1),
  tail-loss concavity / fragility (§3.12 / §9.3.4), Kelly sizing cap (§9.3.5).

Nothing here feeds the production MVO / bridge pipeline.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from etf_daily.lib.fat_tail_risk import FatTailConfig, gpd_loss_at_prob


# --------------------------------------------------------------------------- #
# Pillar 1: mean as a weak signal (plug-in / shadow mean gap, §3.1 / §11.2)
# --------------------------------------------------------------------------- #
def build_mean_signal_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per held name: sample mean vs shadow (plug-in) mean and the shrunk signal.

    The framework treats the point mean as a *weak* signal: the sample mean is
    systematically low because the missing mass sits in the tail (=risk, not
    safely-earnable return), so we report the shadow/sample gap and the shrunk
    value rather than trusting the raw mean.
    """
    if frame.empty:
        return frame.iloc[0:0]
    weight = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    held = frame.loc[weight > 0].copy()
    if held.empty:
        held = frame.copy()
    rows: list[dict[str, Any]] = []
    for _, row in held.iterrows():
        sample_mu = row.get("raw_mean")
        shadow_mu = row.get("shadow_mean")
        ratio: float | None
        if (
            sample_mu is not None
            and pd.notna(sample_mu)
            and abs(float(sample_mu)) > 1e-9
            and shadow_mu is not None
            and pd.notna(shadow_mu)
        ):
            ratio = float(shadow_mu) / float(sample_mu)
        else:
            ratio = None
        rows.append(
            {
                "code": row["code"],
                "name": row["name"],
                "weight": float(row["weight"]),
                "sample_mu": None if sample_mu is None or pd.isna(sample_mu) else float(sample_mu),
                "shadow_mu": None if shadow_mu is None or pd.isna(shadow_mu) else float(shadow_mu),
                "shadow_to_sample": ratio,
                "shrink_factor": row.get("shrink_factor"),
                "shrunk_mu": row.get("shrunk_mean"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _cvar_lookup(frame: pd.DataFrame) -> dict[str, float]:
    """code -> |left-tail CVaR| for names with a reliable estimate."""
    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        cvar = row.get("cvar_left")
        if cvar is not None and pd.notna(cvar):
            out[str(row["code"])] = abs(float(cvar))
    return out


def crash_cvar(
    weights: Mapping[str, float],
    frame: pd.DataFrame,
    cfg: FatTailConfig | None = None,
) -> tuple[float, float]:
    """Crash (correlation->1) portfolio CVaR magnitude (§10.1).

    In a crash, fat-tail assets co-move (tail dependence), so diversification
    fails. We therefore aggregate with NO diversification benefit::

        crash_cvar = sum_i w_i * |CVaR_i|

    Names without an estimable CVaR are filled conservatively with the worst
    available |CVaR| (so short-history high-vol names do not silently lower the
    figure). Returns (cvar_magnitude, fraction_of_weight_using_a_fill).
    """
    cfg = cfg or FatTailConfig()
    lookup = _cvar_lookup(frame)
    # Conservative fill for names without an estimable CVaR: use the worst
    # |CVaR| among THIS leg's own reliable members (not a distant excluded
    # outlier elsewhere in the book); fall back to frame-wide worst, then cfg.
    leg_avail = [lookup[c] for c in weights if c in lookup]
    if leg_avail:
        fill = max(leg_avail)
    elif lookup:
        fill = max(lookup.values())
    else:
        fill = float(cfg.cvar_red)
    total_w = float(sum(max(0.0, float(w)) for w in weights.values()))
    if total_w <= 0:
        return 0.0, 0.0
    acc = 0.0
    fill_w = 0.0
    for code, raw_w in weights.items():
        w = max(0.0, float(raw_w)) / total_w
        if code in lookup:
            acc += w * lookup[code]
        else:
            acc += w * fill
            fill_w += w
    return acc, fill_w


# --------------------------------------------------------------------------- #
# Pillar 2: barbell weight generation (§8)
# --------------------------------------------------------------------------- #
def kappa_to_n(kappa_eff: float | None, cfg: FatTailConfig | None = None) -> int:
    """Map portfolio kappa to a suggested number of names (§8.3 step 0).

    "How much data you need" == "how many securities you need". The framework
    supports *broad* diversification (equal-weight 1/n) for high-kappa universes.
    Heuristic (book gives the concept, not a formula): n ~= n_min / (1 - kappa),
    clipped to [n_min, n_max].
    """
    cfg = cfg or FatTailConfig()
    if kappa_eff is None or not math.isfinite(float(kappa_eff)):
        kappa_eff = 0.3
    k = float(np.clip(kappa_eff, 0.0, 0.95))
    n = math.ceil(cfg.n_min / max(1e-6, 1.0 - k))
    return int(np.clip(n, cfg.n_min, cfg.n_max))


def solve_barbell_w(
    k_budget: float,
    cvar_risk_crash: float,
    cvar_safe: float = 0.0,
) -> float:
    """Closed-form safe-leg fraction w from the left-tail loss budget (§8.3 step 3).

    The conservative leg has ~zero tail mass, the aggressive leg carries the
    crash CVaR. Setting portfolio crash CVaR == K (budget) and solving::

        w_safe = (c_risk - K) / (c_risk - c_safe)

    clipped to [0, 1]. If the aggressive leg alone already fits the budget,
    no safe leg is needed (w=0). w is reverse-engineered from the loss you can
    bear, NOT from a covariance matrix.
    """
    k = abs(float(k_budget))
    c_risk = abs(float(cvar_risk_crash))
    c_safe = abs(float(cvar_safe))
    if c_risk <= k:
        return 0.0
    denom = c_risk - c_safe
    if denom <= 1e-9:
        return 1.0 if k < c_safe else 0.0
    w = (c_risk - k) / denom
    return float(np.clip(w, 0.0, 1.0))


@dataclass(frozen=True)
class BarbellProposal:
    label: str
    k_budget: float
    w_safe: float
    w_risk: float
    n_risk: int
    n_safe: int
    n_suggested: int
    kappa_eff: float | None
    cvar_risk_crash: float
    cvar_safe: float
    portfolio_crash_cvar: float
    fill_fraction: float
    weights: dict[str, float]
    table: pd.DataFrame
    excluded_avoid: list[str] = field(default_factory=list)
    objective: str = "budget_barbell"
    no_edge: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "objective": self.objective,
            "no_edge": self.no_edge,
            "k_budget": self.k_budget,
            "w_safe": self.w_safe,
            "w_risk": self.w_risk,
            "n_risk": self.n_risk,
            "n_safe": self.n_safe,
            "n_suggested": self.n_suggested,
            "kappa_eff": self.kappa_eff,
            "cvar_risk_crash": self.cvar_risk_crash,
            "cvar_safe": self.cvar_safe,
            "portfolio_crash_cvar": self.portfolio_crash_cvar,
            "fill_fraction": self.fill_fraction,
        }


def _median_kappa(frame: pd.DataFrame, codes: Sequence[str]) -> float | None:
    vals = pd.to_numeric(
        frame.loc[frame["code"].isin(list(codes)), "kappa"], errors="coerce"
    ).dropna()
    if vals.empty:
        return None
    return float(vals.median())


def build_barbell_proposal(
    frame: pd.DataFrame,
    *,
    risk_codes: Sequence[str],
    safe_codes: Sequence[str],
    cfg: FatTailConfig | None = None,
    k_budget: float | None = None,
    kappa_eff: float | None = None,
    label: str = "",
    excluded_avoid: Sequence[str] | None = None,
) -> BarbellProposal:
    """Build a barbell proposal: equal-weight aggressive leg + budget-sized safe leg."""
    cfg = cfg or FatTailConfig()
    k_budget = float(cfg.loss_budget_k if k_budget is None else k_budget)
    present = set(frame["code"].astype(str))
    risk_codes = [c for c in dict.fromkeys(risk_codes) if c in present]
    safe_codes = [c for c in dict.fromkeys(safe_codes) if c in present]
    n_risk = len(risk_codes)
    n_safe = len(safe_codes)

    risk_eq = {c: 1.0 / n_risk for c in risk_codes} if n_risk else {}
    safe_eq = {c: 1.0 / n_safe for c in safe_codes} if n_safe else {}

    cvar_risk_crash, fill_fraction = crash_cvar(risk_eq, frame, cfg) if risk_eq else (0.0, 0.0)
    cvar_safe, _ = crash_cvar(safe_eq, frame, cfg) if safe_eq else (0.0, 0.0)

    w_safe = solve_barbell_w(k_budget, cvar_risk_crash, cvar_safe) if n_risk else 1.0
    if n_safe == 0:
        w_safe = 0.0
    w_risk = 1.0 - w_safe

    weights: dict[str, float] = {}
    for c in risk_codes:
        weights[c] = w_risk / n_risk if n_risk else 0.0
    for c in safe_codes:
        weights[c] = w_safe / n_safe if n_safe else 0.0

    portfolio_crash_cvar = w_safe * cvar_safe + w_risk * cvar_risk_crash
    if kappa_eff is None:
        kappa_eff = _median_kappa(frame, risk_codes)
    n_suggested = kappa_to_n(kappa_eff, cfg)

    lookup = frame.set_index("code")
    rows: list[dict[str, Any]] = []
    for leg, codes in (("safe", safe_codes), ("risk", risk_codes)):
        for c in codes:
            meta = lookup.loc[c] if c in lookup.index else None
            rows.append(
                {
                    "label": label,
                    "leg": leg,
                    "code": c,
                    "name": ("" if meta is None else meta.get("name", c)),
                    "target_weight": weights.get(c, 0.0),
                    "alpha_left": (None if meta is None else meta.get("alpha_left")),
                    "kappa": (None if meta is None else meta.get("kappa")),
                    "cvar_left": (None if meta is None else meta.get("cvar_left")),
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "label",
            "leg",
            "code",
            "name",
            "target_weight",
            "alpha_left",
            "kappa",
            "cvar_left",
        ],
    )
    return BarbellProposal(
        label=label,
        k_budget=k_budget,
        w_safe=w_safe,
        w_risk=w_risk,
        n_risk=n_risk,
        n_safe=n_safe,
        n_suggested=n_suggested,
        kappa_eff=kappa_eff,
        cvar_risk_crash=cvar_risk_crash,
        cvar_safe=cvar_safe,
        portfolio_crash_cvar=portfolio_crash_cvar,
        fill_fraction=fill_fraction,
        weights=weights,
        table=table,
        excluded_avoid=list(excluded_avoid or []),
    )


def _tilt_scores(
    frame: pd.DataFrame,
    codes: Sequence[str],
    cfg: FatTailConfig,
) -> dict[str, float]:
    """Kelly-direction return score per name: max(shrunk mean, 0) / |CVaR_left|.

    Higher = more expected growth per unit of (tail) risk. Names with a missing
    CVaR are filled with the worst |CVaR| among the candidate set (conservative);
    a non-positive shrunk mean -> score 0 (no long-only edge).
    """
    codes = [c for c in dict.fromkeys(codes) if c in set(frame["code"].astype(str))]
    if not codes:
        return {}
    lookup = frame.set_index("code")
    cvar_lookup = _cvar_lookup(frame)
    avail = [cvar_lookup[c] for c in codes if c in cvar_lookup]
    fill = max(avail) if avail else (max(cvar_lookup.values()) if cvar_lookup else float(cfg.cvar_red))
    scores: dict[str, float] = {}
    for c in codes:
        shrunk = lookup.loc[c, "shrunk_mean"] if c in lookup.index else None
        edge = float(shrunk) if shrunk is not None and pd.notna(shrunk) else 0.0
        edge = max(edge, 0.0)
        cvar_mag = cvar_lookup.get(c, fill)
        if cvar_mag <= 1e-9:
            scores[c] = 0.0
        else:
            scores[c] = edge / cvar_mag
    return scores


def _cap_normalize(scores: Mapping[str, float], cap: float) -> dict[str, float]:
    """Allocate weights proportional to scores, summing to 1, with per-name cap.

    Standard water-filling: names that would exceed `cap` are fixed at `cap` and
    the remainder is redistributed among the rest. If the cap makes sum=1
    infeasible (cap * n <= 1), returns equal weights.
    """
    codes = list(scores)
    n = len(codes)
    if n == 0:
        return {}
    if cap * n <= 1.0 + 1e-12:
        return {c: 1.0 / n for c in codes}
    fixed: dict[str, float] = {}
    free = {c: float(max(scores[c], 0.0)) for c in codes}
    remaining = 1.0
    for _ in range(n + 1):
        free_sum = sum(free.values())
        if not free:
            break
        if free_sum <= 0.0:
            share = remaining / len(free)
            for c in free:
                fixed[c] = share
            free = {}
            break
        alloc = {c: remaining * v / free_sum for c, v in free.items()}
        over = [c for c, a in alloc.items() if a > cap + 1e-12]
        if not over:
            fixed.update(alloc)
            free = {}
            break
        for c in over:
            fixed[c] = cap
            remaining -= cap
            del free[c]
    return fixed


def return_tilt_weights(
    frame: pd.DataFrame,
    codes: Sequence[str],
    cfg: FatTailConfig | None = None,
) -> tuple[dict[str, float], bool]:
    """Fully-invested long-only weights tilted toward Kelly return-per-tail-risk.

    Returns (weights summing to 1, no_edge). When no name has a positive edge,
    falls back to equal weight and sets no_edge=True.
    """
    cfg = cfg or FatTailConfig()
    scores = _tilt_scores(frame, codes, cfg)
    if not scores:
        return {}, True
    if sum(scores.values()) <= 0.0:
        n = len(scores)
        return {c: 1.0 / n for c in scores}, True
    return _cap_normalize(scores, cfg.max_name_weight), False


def build_return_max_proposal(
    frame: pd.DataFrame,
    *,
    candidate_codes: Sequence[str],
    cfg: FatTailConfig | None = None,
    label: str = "",
    excluded_avoid: Sequence[str] | None = None,
    kappa_eff: float | None = None,
) -> BarbellProposal:
    """Fully-invested, return-tilted aggressive proposal (no safe leg).

    Maximizes growth (Kelly direction) subject to a per-name cap; there is NO
    barbell safe leg, so the left-tail loss budget K is not structurally
    enforced (the report surfaces the resulting budget breach).
    """
    cfg = cfg or FatTailConfig()
    present = set(frame["code"].astype(str))
    candidate_codes = [c for c in dict.fromkeys(candidate_codes) if c in present]
    weights, no_edge = return_tilt_weights(frame, candidate_codes, cfg)
    n_risk = len(candidate_codes)
    cvar_crash, fill_fraction = crash_cvar(weights, frame, cfg) if weights else (0.0, 0.0)
    if kappa_eff is None:
        kappa_eff = _median_kappa(frame, candidate_codes)
    n_suggested = kappa_to_n(kappa_eff, cfg)

    lookup = frame.set_index("code")
    rows: list[dict[str, Any]] = []
    for c in sorted(candidate_codes, key=lambda x: weights.get(x, 0.0), reverse=True):
        meta = lookup.loc[c] if c in lookup.index else None
        rows.append(
            {
                "label": label,
                "leg": "risk",
                "code": c,
                "name": ("" if meta is None else meta.get("name", c)),
                "target_weight": weights.get(c, 0.0),
                "alpha_left": (None if meta is None else meta.get("alpha_left")),
                "kappa": (None if meta is None else meta.get("kappa")),
                "cvar_left": (None if meta is None else meta.get("cvar_left")),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "label",
            "leg",
            "code",
            "name",
            "target_weight",
            "alpha_left",
            "kappa",
            "cvar_left",
        ],
    )
    return BarbellProposal(
        label=label,
        k_budget=float(cfg.loss_budget_k),
        w_safe=0.0,
        w_risk=1.0,
        n_risk=n_risk,
        n_safe=0,
        n_suggested=n_suggested,
        kappa_eff=kappa_eff,
        cvar_risk_crash=cvar_crash,
        cvar_safe=0.0,
        portfolio_crash_cvar=cvar_crash,
        fill_fraction=fill_fraction,
        weights=weights,
        table=table,
        excluded_avoid=list(excluded_avoid or []),
        objective="return_max",
        no_edge=no_edge,
    )


def _reliable_mask(frame: pd.DataFrame) -> pd.Series:
    return ~frame["flags"].astype(str).str.contains(
        "low_n|alpha_unreliable|missing_price"
    )


def propose_from_holdings(
    holdings_frame: pd.DataFrame,
    defensive_codes: set[str] | frozenset[str],
    cfg: FatTailConfig | None = None,
) -> BarbellProposal:
    """Proposal A: rebuild the *current* book into a barbell.

    Aggressive leg = held non-defensive names that are not in the `avoid` leg
    (equal-weight); safe leg = defensive bonds; `avoid` names are dropped and
    listed separately. The safe fraction is sized from the left-tail budget.
    """
    cfg = cfg or FatTailConfig()
    defensive = set(defensive_codes)
    excluded_codes = set(cfg.barbell_exclude_codes)
    weight = pd.to_numeric(holdings_frame["weight"], errors="coerce").fillna(0.0)
    held = holdings_frame.loc[weight > 0]
    held_codes = set(held["code"].astype(str))

    safe_codes = [
        c
        for c in holdings_frame["code"].astype(str)
        if c in defensive and c not in excluded_codes
    ]
    code_col = holdings_frame["code"].astype(str)
    non_def = holdings_frame.loc[
        code_col.isin(held_codes)
        & ~code_col.isin(defensive)
        & ~code_col.isin(excluded_codes)
    ]
    risk_codes = [
        str(r["code"])
        for _, r in non_def.iterrows()
        if r["barbell_leg"] != "avoid"
    ]
    excluded = [
        str(r["code"])
        for _, r in non_def.iterrows()
        if r["barbell_leg"] == "avoid"
    ]
    kappa_eff = _median_kappa(holdings_frame, risk_codes)
    if cfg.proposal_objective == "return_max":
        # Tilting toward return must not concentrate into tail-unknown names:
        # restrict to reliable holdings (unreliable ones still appear in the
        # per-asset boards / Kelly section, just not in the generated weights).
        reliable_codes = set(
            holdings_frame.loc[_reliable_mask(holdings_frame), "code"].astype(str)
        )
        candidate_codes = [c for c in risk_codes if c in reliable_codes]
        return build_return_max_proposal(
            holdings_frame,
            candidate_codes=candidate_codes,
            cfg=cfg,
            label="A·当前持仓·收益最大化",
            excluded_avoid=excluded,
            kappa_eff=kappa_eff,
        )
    return build_barbell_proposal(
        holdings_frame,
        risk_codes=risk_codes,
        safe_codes=safe_codes,
        cfg=cfg,
        kappa_eff=kappa_eff,
        label="A·当前持仓重构",
        excluded_avoid=excluded,
    )


def propose_from_pool(
    full_frame: pd.DataFrame,
    defensive_codes: set[str] | frozenset[str],
    cfg: FatTailConfig | None = None,
) -> BarbellProposal:
    """Proposal B: ideal broad barbell selected from the full candidate pool.

    Pick the kappa-defined number of thinnest-tailed reliable non-`avoid` names
    (sorted by kappa asc, then alpha_left desc) for the equal-weight aggressive
    leg; safe leg = defensive bonds.
    """
    cfg = cfg or FatTailConfig()
    defensive = set(defensive_codes)
    excluded_codes = set(cfg.barbell_exclude_codes)
    reliable = full_frame.loc[_reliable_mask(full_frame)].copy()
    candidates = reliable.loc[
        ~reliable["code"].isin(defensive)
        & ~reliable["code"].isin(excluded_codes)
        & (reliable["barbell_leg"] != "avoid")
    ].copy()
    kappa_eff = _median_kappa(full_frame, candidates["code"].astype(str).tolist())
    n_suggested = kappa_to_n(kappa_eff, cfg)

    if cfg.proposal_objective == "return_max":
        scores = _tilt_scores(full_frame, candidates["code"].astype(str).tolist(), cfg)
        ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
        risk_codes = ranked[:n_suggested]
        return build_return_max_proposal(
            full_frame,
            candidate_codes=risk_codes,
            cfg=cfg,
            label="B·候选池·收益最大化",
            kappa_eff=kappa_eff,
        )

    candidates["_k"] = pd.to_numeric(candidates["kappa"], errors="coerce").fillna(1.0)
    candidates["_a"] = pd.to_numeric(candidates["alpha_left"], errors="coerce").fillna(0.0)
    candidates = candidates.sort_values(["_k", "_a"], ascending=[True, False])
    risk_codes = candidates["code"].astype(str).head(n_suggested).tolist()
    safe_codes = [
        c
        for c in full_frame["code"].astype(str)
        if c in defensive and c not in excluded_codes
    ]
    return build_barbell_proposal(
        full_frame,
        risk_codes=risk_codes,
        safe_codes=safe_codes,
        cfg=cfg,
        kappa_eff=kappa_eff,
        label="B·候选池理想组合",
    )


# --------------------------------------------------------------------------- #
# Pillar 3: risk control (§9, §10.1)
# --------------------------------------------------------------------------- #
def hard_stop_k(proposal: BarbellProposal, cfg: FatTailConfig | None = None) -> dict[str, Any]:
    """Total-level hard stop K (§25.5 / §9): acts on portfolio total, not per name.

    The barbell safe fraction is what structurally caps left-tail mass; the stop
    K is the maximum acceptable daily portfolio loss.
    """
    cfg = cfg or FatTailConfig()
    return {
        "k_total": float(cfg.loss_budget_k),
        "enforced_by_safe_w": float(proposal.w_safe),
        "portfolio_crash_cvar": float(proposal.portfolio_crash_cvar),
        "within_budget": bool(proposal.portfolio_crash_cvar <= cfg.loss_budget_k + 1e-9),
    }


def concavity_flags(frame: pd.DataFrame, cfg: FatTailConfig | None = None) -> pd.DataFrame:
    """Tail-loss concavity / fragility check (§3.12 / §9.3.4).

    Using the fitted left GPD, compare the loss at tail prob p and at p/2 (a
    "doubling shock" deeper into the tail). When loss(p/2)/loss(p) >= warn, the
    exposure behaves like short-gamma (loss accelerates) -> fragile.
    """
    cfg = cfg or FatTailConfig()
    p = max(1e-4, 1.0 - cfg.var_confidence)
    weight = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    held = frame.loc[weight > 0].copy()
    if held.empty:
        held = frame.copy()
    rows: list[dict[str, Any]] = []
    for _, row in held.iterrows():
        loss_p = gpd_loss_at_prob(
            xi=row.get("xi_left"),
            beta=row.get("beta_left"),
            threshold=row.get("threshold_left"),
            nu=row.get("nu_left"),
            tail_prob=p,
        )
        loss_half = gpd_loss_at_prob(
            xi=row.get("xi_left"),
            beta=row.get("beta_left"),
            threshold=row.get("threshold_left"),
            nu=row.get("nu_left"),
            tail_prob=p / 2.0,
        )
        ratio: float | None
        if loss_p is not None and loss_half is not None and loss_p > 1e-9:
            ratio = loss_half / loss_p
        else:
            ratio = None
        rows.append(
            {
                "code": row["code"],
                "name": row["name"],
                "weight": float(row["weight"]),
                "alpha_left": row.get("alpha_left"),
                "loss_p": loss_p,
                "loss_phalf": loss_half,
                "concavity_ratio": ratio,
                "fragile": bool(ratio is not None and ratio >= cfg.concavity_ratio_warn),
            }
        )
    return pd.DataFrame(rows)


def _single_day_drop_tiers(row: Mapping[str, Any], cfg: FatTailConfig) -> tuple[list[float | None], bool]:
    """Per-name single-day drop thresholds for §12 stop/take-profit (no σ).

    Uses the fitted left-tail GPD to read the parametric single-day VaR at each
    confidence in ``cfg.exit_var_confidences`` (e.g. 95% / 99% / 99.7%) and
    converts the log-loss to an actual price drop ``1 - exp(-loss)``. Tiers that
    cannot be estimated (no reliable left tail, or the probability lies above the
    modelled tail) fall back to the fixed ``cfg.exit_drop_fallback`` value and the
    row is marked as relying on a fallback. Returns (drops, used_fallback).
    """
    xi = row.get("xi_left")
    beta = row.get("beta_left")
    threshold = row.get("threshold_left")
    nu = row.get("nu_left")
    drops: list[float | None] = []
    used_fallback = False
    for conf, fallback in zip(cfg.exit_var_confidences, cfg.exit_drop_fallback):
        tail_prob = 1.0 - float(conf)
        loss = gpd_loss_at_prob(xi=xi, beta=beta, threshold=threshold, nu=nu, tail_prob=tail_prob)
        if loss is None or not math.isfinite(loss) or loss <= 0:
            drops.append(float(fallback))
            used_fallback = True
        else:
            drops.append(float(1.0 - math.exp(-float(loss))))
    return drops, used_fallback


def is_thin_left_tail(row: Mapping[str, Any], cfg: FatTailConfig | None = None) -> bool:
    """Left tail is thin/bounded (α_L high or ξ≤0), not a fat-tail hazard."""
    cfg = cfg or FatTailConfig()
    flags = str(row.get("flags") or "")
    if "bounded_tail_left" in flags:
        return True
    alpha = row.get("alpha_left")
    if alpha is not None and pd.notna(alpha) and float(alpha) >= cfg.alpha_thin:
        return True
    return False


def name_risk_action(row: Mapping[str, Any]) -> tuple[str, str]:
    """Per-name §12 action from path/drawdown + α/κ tail structure + position budget.

    No per-name σ-stop and no MA/return trend forecast. The ONLY hard stop is the
    portfolio HWM drawdown (book-level, handled separately). Per-name actions are
    qualitative and driven by, in priority order:
      - tail_thickening_confirmed: structural α drift -> de-risk / re-estimate tail (§7).
      - path broken (deep high-water drawdown) + oversized -> trim toward budget cap.
      - path broken -> watch / partial trim (§12.3 path & ruin).
      - tail_thickening_watch: tail thickening -> do not add, watch portfolio drawdown.
      - oversized (stable, not broken): over Kelly budget -> trim toward cap on strength.
      - otherwise: hold (no mechanical stop).

    Returns (action, reason). Cost-basis P&L and MA trend are NOT inputs (avoids §3.10
    anchoring and mean/MA forecasting).
    """
    tail_regime = str(row.get("tail_regime") or "stable")
    path_state = str(row.get("path_state") or "near_high")
    oversized = bool(row.get("oversized"))

    if tail_regime == "tail_thickening_confirmed":
        return "降risk·重估尾", "尾部结构 α 持续漂移（§7 confirmed），降低风险敞口并重估左尾"
    if path_state == "broken" and oversized:
        return "减至预算上限", "高水位深度回撤（破位）且超配 kelly 上限 → 减仓至预算上限"
    if path_state == "broken":
        return "观察/部分减", "高水位深度回撤（破位 · §12.3 路径/ruin）→ 观察，必要时部分减仓"
    if tail_regime == "tail_thickening_watch":
        return "不加仓·盯回撤", "尾变厚·观察 → 不加仓，紧盯组合 NAV 回撤硬止损"
    if oversized:
        return "逢强减至上限", "超配 kelly 预算（非急）→ 逢强减至预算上限"
    return "持有", "结构稳定且未破位 → 持有，无机械止损"


def name_exit_plan(frame: pd.DataFrame, cfg: FatTailConfig | None = None) -> pd.DataFrame:
    """Per-name §12 monitoring + action board for held positions (no per-name σ-stop).

    For every held name (``weight > 0``) emit a row that combines:
      - single-day VaR depths (``drop_tiers``) as MONITORING references only
        ("a single-day drop beyond these is abnormal for this name"); names
        without an estimable tail use the fixed reference fallback drops;
      - the Kelly position budget (``kelly_cap`` / ``oversized``, §9.3.5 / §12.2);
      - the §7 tail-structure regime + §12.3 path/drawdown state that drive ``action``.

    Cost-basis P&L (``unrealized_return``) and the MA20/20d-return metrics
    (``dist_ma20_pct`` / ``ret_20d``) are carried for DISPLAY only and never feed the
    action (avoids §3.10 break-even anchoring and mean/MA forecasting). The only hard
    stop is the portfolio HWM drawdown, handled at the book level. Sorted by weight desc.
    """
    cfg = cfg or FatTailConfig()
    columns = [
        "code",
        "name",
        "weight",
        "kelly_cap",
        "oversized",
        "tail_regime",
        "path_state",
        "dd_near",
        "dd_long",
        "drop_tiers",
        "dist_ma20_pct",
        "ret_20d",
        "unrealized_return",
        "tail_fallback",
        "action",
        "action_reason",
        "flags",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    caps = kelly_cap(frame, cfg)
    cap_by_code = {str(r["code"]): r for _, r in caps.iterrows()} if not caps.empty else {}
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        weight = float(pd.to_numeric(row.get("weight"), errors="coerce") or 0.0)
        if weight <= 0:
            continue
        cost = pd.to_numeric(row.get("cost_price"), errors="coerce")
        close = pd.to_numeric(row.get("close_price"), errors="coerce")
        unrealized = (
            None
            if pd.isna(cost) or pd.isna(close) or float(cost) <= 0
            else float(close) / float(cost) - 1.0
        )
        drops, used_fallback = _single_day_drop_tiers(row, cfg)
        cap_row = cap_by_code.get(str(row["code"]))
        kelly = float(cap_row["kelly_cap"]) if cap_row is not None else None
        oversized = bool(cap_row["oversized"]) if cap_row is not None else False
        enriched = {
            "tail_regime": row.get("tail_regime"),
            "path_state": row.get("path_state"),
            "oversized": oversized,
        }
        action, reason = name_risk_action(enriched)
        rows.append(
            {
                "code": row["code"],
                "name": row["name"],
                "weight": weight,
                "kelly_cap": kelly,
                "oversized": oversized,
                "tail_regime": row.get("tail_regime"),
                "path_state": row.get("path_state"),
                "dd_near": row.get("dd_near"),
                "dd_long": row.get("dd_long"),
                "drop_tiers": drops,
                "dist_ma20_pct": row.get("dist_ma20_pct"),
                "ret_20d": row.get("ret_20d"),
                "unrealized_return": unrealized,
                "tail_fallback": bool(used_fallback),
                "action": action,
                "action_reason": reason,
                "flags": row.get("flags"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    plan = pd.DataFrame(rows)
    return plan.sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)


def build_exit_tier_table(frame: pd.DataFrame, cfg: FatTailConfig | None = None) -> pd.DataFrame:
    """Per-name single-day VaR depths for any universe (holdings or candidate pool).

    Unlike ``name_exit_plan``, includes every row in ``frame`` (weight may be 0).
    Emits the three parametric left-tail VaR price-drop depths as MONITORING
    references only (no execution reduction ladder), plus the §7 tail-structure
    regime and §12.3 path/drawdown state labels.
    """
    cfg = cfg or FatTailConfig()
    if frame.empty:
        return frame.iloc[0:0]
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        weight = float(pd.to_numeric(row.get("weight"), errors="coerce") or 0.0)
        drops, used_fallback = _single_day_drop_tiers(row, cfg)
        rows.append(
            {
                "code": row["code"],
                "name": row["name"],
                "weight": weight,
                "barbell_leg": row.get("barbell_leg"),
                "drop_tiers": drops,
                "tail_regime": row.get("tail_regime"),
                "path_state": row.get("path_state"),
                "dd_near": row.get("dd_near"),
                "dd_long": row.get("dd_long"),
                "tail_fallback": bool(used_fallback),
                "flags": row.get("flags"),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def kelly_cap(frame: pd.DataFrame, cfg: FatTailConfig | None = None) -> pd.DataFrame:
    """Fractional-Kelly per-name sizing cap to bound ruin (§9.3.5).

    f_cap = clip(kelly_fraction * shadow_mean / |CVaR_left|, 0, max_name_weight).
    A non-positive shadow edge or a missing tail -> cap 0 (no justified size).
    Flags holdings whose current weight exceeds the cap (oversized -> ruin risk).
    The cap shrinks as the tail gets fatter (larger |CVaR|).
    """
    cfg = cfg or FatTailConfig()
    weight = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    held = frame.loc[weight > 0].copy()
    rows: list[dict[str, Any]] = []
    for _, row in held.iterrows():
        shadow = row.get("shadow_mean")
        cvar = row.get("cvar_left")
        if (
            shadow is None
            or pd.isna(shadow)
            or float(shadow) <= 0.0
            or cvar is None
            or pd.isna(cvar)
            or abs(float(cvar)) < 1e-9
        ):
            cap = 0.0
        else:
            cap = float(
                np.clip(
                    cfg.kelly_fraction * float(shadow) / abs(float(cvar)),
                    0.0,
                    cfg.max_name_weight,
                )
            )
        w = float(row["weight"])
        rows.append(
            {
                "code": row["code"],
                "name": row["name"],
                "weight": w,
                "shadow_mean": None if shadow is None or pd.isna(shadow) else float(shadow),
                "cvar_left": None if cvar is None or pd.isna(cvar) else float(cvar),
                "kelly_cap": cap,
                "oversized": bool(w > cap + 1e-9),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BarbellProposal",
    "build_barbell_proposal",
    "build_mean_signal_table",
    "build_return_max_proposal",
    "concavity_flags",
    "crash_cvar",
    "hard_stop_k",
    "kappa_to_n",
    "kelly_cap",
    "build_exit_tier_table",
    "is_thin_left_tail",
    "name_exit_plan",
    "name_risk_action",
    "propose_from_holdings",
    "propose_from_pool",
    "return_tilt_weights",
    "solve_barbell_w",
]
