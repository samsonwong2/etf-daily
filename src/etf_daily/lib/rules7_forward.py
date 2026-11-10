"""Frozen rules7 replay: ledger, failure tests, and the inputs for the forward page.

The buy and sell rules are the ones in ``rules7.py``. This module does not
change them. ``spec_fingerprint`` covers the knobs that decide a trade. A
change there fails ``tests/test_rules7_freeze.py`` until the freeze tag is
bumped and the forward clock starts over.

Segments of a trade, by its signal day:

* ``in_sample`` — before 2026-10-14. The rules were chosen on this history.
* ``pseudo_forward`` — 2026-10-14 through the day before ``FORWARD_START``.
  The rule logic was already the frozen one (``0263d0d``), but the failure
  tests were not written down yet, so these trades do not count.
* ``forward`` — on or after ``FORWARD_START``, the first trading day after
  the ``rules7-freeze`` tag.

Research only. Nothing here rewrites a listing HTML page.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etf_daily.lib import rules7

FREEZE_TAG = "rules7-freeze"
LOGIC_EQUIVALENT_TO = "0263d0d"
# Day after 0263d0d. That commit is the last change to a buy or sell decision.
PSEUDO_FORWARD_START = "2026-10-14"
# Tag date, and the first trading day after it. Filled for the rules7-freeze tag.
FREEZE_DATE = "2026-11-10"
FORWARD_START = "2026-11-11"
GATE_MONTHS = 3
COST = 0.0005
ANCHOR_CODE = "SH510300"
MIN_HOLDOUT_BARS = 500
K1_MIN_ELIGIBLE = 10

SPECS_PATH = Path(__file__).with_name("rules7_freeze_specs.json")

# One-shot leave-out on 2026-11-09 listing HTML. Types used the full history,
# so this is optimistic. It is the reference line, not a thing to recompute.
HOLDOUT_BASELINE: dict[str, Any] = {
    "as_of": "2026-11-09",
    "anchor": ANCHOR_CODE,
    "n_etf": 46,
    "n_trades": 267,
    "win_rate": 0.5430711610486891,
    "median_ret": 0.023206766309116444,
    "mean_ret": 0.04790482207472931,
    "mdd_shallower": [33, 46],
    "calmar_beats_bh": [20, 46],
    "total_beats_bh": [24, 46],
    "by_type_calmar": {
        "平稳震荡": [0, 2],
        "慢趋势": [5, 14],
        "长期下跌或周期": [14, 25],
        "高波动主题": [1, 5],
    },
    "seven_closed": 57,
    "seven_win_rate": 0.6666666666666666,
    "seven_median_ret": 0.07681463375907893,
    "seven_mean_ret": 0.2025467535949558,
    "caveats": [
        "类型用了全历史特征，留出偏乐观。",
        "买入持有回撤算不出来的标的记为不胜，所以胜出只数是下限。",
        "这是参数没在这 46 只上挑过，不是时间外的样本。",
    ],
}

CRITERIA: dict[str, Any] = {
    "k1_bh_mdd": -0.05,
    "k1_fail_share": 0.50,
    "k1_min_eligible": K1_MIN_ELIGIBLE,
    "k2_min_trades": 30,
    "k2_fail_wins": 11,
    "k2_base_rate": 0.54,
    "k3_mdd_multiple": 1.5,
    "gate_months": GATE_MONTHS,
    "gate_date": "2027-02-10",
}

LEDGER_COLUMNS = (
    "code",
    "name",
    "type",
    "spec_fp",
    "segment",
    "signal_date",
    "fill_date",
    "fill_px",
    "exit_signal_date",
    "exit_fill_date",
    "exit_fill_px",
    "exit_reason",
    "ret_gross",
    "ret_net",
    "mdd_in_trade",
    "open_unrealized",
)

DAILY_COLUMNS = ("as_of", "code", "holding", "action", "equity", "bh_equity", "spec_fp")

CLOSED_REASONS = frozenset({"卖点", "回落"})
STUDY_CODES = tuple(rules7.OVERRIDES)
DEFAULT_PAGE_CODES = STUDY_CODES


def _spec_dict(spec: rules7.RuleSpec) -> dict[str, Any]:
    return {
        "b1": bool(spec.b1),
        "fast_block": bool(spec.fast_block),
        "sell": spec.sell,
        "g1_hold": bool(spec.g1_hold),
        "trail": spec.trail,
    }


def spec_payload() -> dict[str, Any]:
    """The knobs that change a buy, a sell, or a 回落. Names and snapshots are not in here."""
    return {
        "B1_GAP": rules7.B1_GAP,
        "FAST_LOOKBACK": rules7.FAST_LOOKBACK,
        "FAST_RUN": rules7.FAST_RUN,
        "FAST_RUN_BARS": rules7.FAST_RUN_BARS,
        "NOT_APPLICABLE": sorted(rules7.NOT_APPLICABLE),
        "OVERRIDES": {
            code: [typ, _spec_dict(spec)] for code, (typ, spec) in sorted(rules7.OVERRIDES.items())
        },
        "TYPE_PARAMS": {name: _spec_dict(spec) for name, spec in rules7.TYPE_PARAMS.items()},
    }


def spec_fingerprint() -> str:
    raw = json.dumps(spec_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def segment_for(signal_date: object) -> str:
    day = pd.Timestamp(signal_date).normalize()
    if day < pd.Timestamp(PSEUDO_FORWARD_START):
        return "in_sample"
    if day < pd.Timestamp(FORWARD_START):
        return "pseudo_forward"
    return "forward"


def net_return(fill_px: float, exit_px: float, *, cost: float = COST) -> float:
    """One round trip. ``cost`` is charged on the fill and again on the exit."""
    if not np.isfinite(fill_px) or not np.isfinite(exit_px) or fill_px <= 0 or exit_px <= 0:
        return float("nan")
    return float((exit_px * (1.0 - cost)) / (fill_px * (1.0 + cost)) - 1.0)


def load_specs(path: Path | None = None) -> dict[str, Any]:
    src = Path(path) if path is not None else SPECS_PATH
    if not src.is_file():
        raise FileNotFoundError(f"missing freeze specs: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "codes" not in data:
        raise ValueError(f"freeze specs have no codes: {src}")
    return data


def frozen_type_spec(code: str, specs: dict[str, Any]) -> tuple[str, str, rules7.RuleSpec]:
    """Return (type, universe, spec) from the freeze file, not from today's history.

    The 7 documented codes always use ``OVERRIDES``, which is inside the fingerprint.
    Every other code uses the type stored at the freeze, so a later bar cannot
    move it into another type and silently change its rule.
    """
    code = code.upper()
    if code in rules7.OVERRIDES:
        typ, spec = rules7.OVERRIDES[code]
        return typ, "study", spec
    row = specs["codes"].get(code)
    if row is None:
        raise KeyError(code)
    typ = str(row["type"])
    if typ not in rules7.TYPE_PARAMS:
        raise KeyError(f"{code} frozen type {typ}")
    return typ, "holdout", rules7.TYPE_PARAMS[typ]


def _listing_helpers():
    from etf_daily.scripts.backtest_fig12_six_states_pool import list_equity_html, parse_html_label
    from etf_daily.scripts.scan_rules7_checklist import build_frame

    return list_equity_html, parse_html_label, build_frame


def load_prepared(html: Path, listing: Path, as_of: object, *, anchor: str = ANCHOR_CODE) -> pd.DataFrame:
    """Feature frame through ``as_of``, same columns ``rules7.evaluate`` expects."""
    _list, _label, build_frame = _listing_helpers()
    frame = build_frame(Path(html), Path(listing), anchor)
    cut = pd.Timestamp(as_of).normalize()
    frame = frame.loc[pd.DatetimeIndex(frame.index).normalize() <= cut]
    if frame.empty:
        raise ValueError(f"no bars on or before {cut.date()} in {html.name}")
    return frame


def _next_bar(index: pd.DatetimeIndex, day: object) -> pd.Timestamp | None:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    loc = idx.get_loc(pd.Timestamp(day).normalize())
    if isinstance(loc, slice):
        loc = loc.start
    elif isinstance(loc, np.ndarray):
        loc = int(np.flatnonzero(loc)[0])
    loc = int(loc)
    if loc + 1 >= len(idx):
        return None
    return pd.Timestamp(idx[loc + 1])


def _px(close: pd.Series, day: object) -> float:
    idx = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    series = pd.Series(close.to_numpy(dtype=float), index=idx)
    value = series.loc[pd.Timestamp(day).normalize()]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


def _blank_row(code: str, name: str, typ: str, signal_date: object, reason: str) -> dict[str, Any]:
    return {
        "code": code,
        "name": name,
        "type": typ,
        "spec_fp": spec_fingerprint(),
        "segment": segment_for(signal_date),
        "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        "fill_date": "",
        "fill_px": "",
        "exit_signal_date": "",
        "exit_fill_date": "",
        "exit_fill_px": "",
        "exit_reason": reason,
        "ret_gross": "",
        "ret_net": "",
        "mdd_in_trade": "",
        "open_unrealized": "",
    }


def filled_close(frame: pd.DataFrame) -> pd.Series:
    """Close series with holiday gaps carried forward, as ``rules7.evaluate`` does.

    Without this the state machine turns equity into NaN after one missing bar.
    """
    return frame["px"].astype(float).ffill()


def replay_frame(
    code: str,
    name: str,
    frame: pd.DataFrame,
    *,
    specs: dict[str, Any] | None = None,
    typ: str | None = None,
    spec: rules7.RuleSpec | None = None,
) -> dict[str, Any]:
    """Run the frozen rule on one prepared frame.

    Pass ``specs`` (the freeze file) in production. Tests may pass ``typ`` and
    ``spec`` directly. The state machine is ``rules7.run_state_machine``; this
    function only records fills.
    """
    code = code.upper()
    if spec is None or typ is None:
        if specs is None:
            specs = load_specs()
        typ, _universe, spec = frozen_type_spec(code, specs)
    close = filled_close(frame)
    masks = rules7.build_masks(frame)
    buy, sell = rules7.buy_sell_masks(masks, spec)
    result = rules7.run_state_machine(
        close,
        buy,
        sell,
        g1=masks["g1"] if spec.g1_hold else None,
        trail=spec.trail,
        cost=COST,
    )
    rows: list[dict[str, Any]] = []
    for trade in result["trades"]:
        fill_day = pd.Timestamp(trade["fill"])
        exit_day = _next_bar(close.index, trade["sell"])
        fill_px = _px(close, fill_day)
        exit_px = _px(close, exit_day) if exit_day is not None else float("nan")
        gross = float(trade["ret"])
        row = _blank_row(code, name, typ, trade["buy"], str(trade["why"]))
        row["fill_date"] = fill_day.strftime("%Y-%m-%d")
        row["fill_px"] = fill_px
        row["exit_signal_date"] = pd.Timestamp(trade["sell"]).strftime("%Y-%m-%d")
        row["exit_fill_date"] = "" if exit_day is None else exit_day.strftime("%Y-%m-%d")
        row["exit_fill_px"] = "" if not np.isfinite(exit_px) else exit_px
        row["ret_gross"] = gross
        row["ret_net"] = net_return(fill_px, exit_px)
        row["mdd_in_trade"] = float(trade["mdd"])
        rows.append(row)

    state = result["state"]
    if state.get("buy_date") is not None and state.get("fill_px") is None:
        rows.append(_blank_row(code, name, typ, state["buy_date"], "待成交"))
    elif state.get("holding") and state.get("fill_date") is not None:
        row = _blank_row(code, name, typ, state["buy_date"], "持有中")
        fill_px = float(state["fill_px"])
        last_px = float(close.iloc[-1])
        row["fill_date"] = pd.Timestamp(state["fill_date"]).strftime("%Y-%m-%d")
        row["fill_px"] = fill_px
        row["open_unrealized"] = last_px / fill_px - 1.0 if fill_px > 0 else ""
        fill_loc = close.index.get_loc(pd.Timestamp(state["fill_date"]))
        if isinstance(fill_loc, slice):
            fill_loc = fill_loc.start
        elif isinstance(fill_loc, np.ndarray):
            fill_loc = int(np.flatnonzero(fill_loc)[0])
        row["mdd_in_trade"] = rules7.max_drawdown(close.iloc[int(fill_loc):])
        rows.append(row)

    bh = float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close) else float("nan")
    return {
        "code": code,
        "name": name,
        "type": typ,
        "spec": spec,
        "applicable": code not in rules7.NOT_APPLICABLE,
        "frame": frame,
        "masks": masks,
        "result": result,
        "rows": rows,
        "bh_total": bh,
    }


def _num(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out


def k1_status(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Share of holdout names whose forward drawdown is shallower than buy-and-hold.

    ``rows`` entries need ``bh_mdd`` and ``strat_mdd`` (both negative when down).
    A name counts only when buy-and-hold fell at least 5%. Fail when the
    shallower share is under 50%, once at least ``K1_MIN_ELIGIBLE`` names qualify.
    """
    eligible = [r for r in rows if np.isfinite(r["bh_mdd"]) and r["bh_mdd"] <= CRITERIA["k1_bh_mdd"]]
    shallower = [
        r for r in eligible if np.isfinite(r["strat_mdd"]) and r["strat_mdd"] > r["bh_mdd"]
    ]
    n = len(eligible)
    if n < int(CRITERIA["k1_min_eligible"]):
        return {"status": "未定", "eligible": n, "shallower": len(shallower), "share": None}
    share = len(shallower) / n
    status = "不通过" if share < float(CRITERIA["k1_fail_share"]) else "通过"
    return {"status": status, "eligible": n, "shallower": len(shallower), "share": share}


def k2_status(rets: list[float]) -> dict[str, Any]:
    """Closed forward trades. Fail at 30 or more when wins are 11/30 or worse, or the median is not positive."""
    clean = [float(r) for r in rets if np.isfinite(r)]
    n = len(clean)
    wins = int(sum(r > 0 for r in clean))
    median = float(np.median(clean)) if clean else float("nan")
    if n < int(CRITERIA["k2_min_trades"]):
        return {"status": "未定", "n": n, "wins": wins, "median": median}
    fail_rate = int(CRITERIA["k2_fail_wins"]) / int(CRITERIA["k2_min_trades"])
    failed = (wins / n) <= fail_rate or not (median > 0)
    return {"status": "不通过" if failed else "通过", "n": n, "wins": wins, "median": median}


def k2_false_fail_rate(p: float | None = None) -> float:
    """P(X <= 11) for X ~ Binomial(30, p). The leave-out win rate makes this about 4%."""
    from math import comb

    n = int(CRITERIA["k2_min_trades"])
    k_max = int(CRITERIA["k2_fail_wins"])
    prob = float(CRITERIA["k2_base_rate"] if p is None else p)
    return sum(comb(n, k) * prob**k * (1.0 - prob) ** (n - k) for k in range(k_max + 1))


def k3_status(fwd_mdd: float, sample_mdd: float, bh_mdd: float) -> str:
    """Fuse for one of the 7. Both must be true: deeper than 1.5x its own sample, and deeper than buy-and-hold."""
    if not np.isfinite(fwd_mdd) or not np.isfinite(sample_mdd) or sample_mdd >= 0:
        return "未定"
    worse_than_sample = abs(fwd_mdd) > float(CRITERIA["k3_mdd_multiple"]) * abs(sample_mdd)
    deeper_than_bh = np.isfinite(bh_mdd) and fwd_mdd < bh_mdd
    if worse_than_sample and deeper_than_bh:
        return "不通过"
    return "通过"


def gate_open(as_of: object, n_forward_closed: int) -> bool:
    """True while it is still too early to call pass or fail (except the K3 fuse)."""
    ready_date = pd.Timestamp(as_of).normalize() >= pd.Timestamp(CRITERIA["gate_date"])
    ready_n = n_forward_closed >= int(CRITERIA["k2_min_trades"])
    return not (ready_date and ready_n)


def overall_status(k1: str, k2: str, k3: str, *, waiting: bool) -> str:
    if k3 == "不通过":
        return "不通过"
    if waiting:
        return "未定"
    if k1 == "不通过" or k2 == "不通过":
        return "不通过"
    if k1 == "通过" and k2 == "通过":
        return "通过"
    return "未定"


def _window_mdd(series: pd.Series, start: str) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    idx = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    values = pd.Series(np.asarray(series, dtype=float), index=idx)
    window = values.loc[values.index >= pd.Timestamp(start)]
    window = window[np.isfinite(window.to_numpy(dtype=float))]
    if len(window) < 2:
        return float("nan")  # no window yet: undecided, not "zero drawdown"
    return rules7.max_drawdown(window)


def assess(
    curves: dict[str, dict[str, pd.Series]],
    ledger_rows: list[dict[str, Any]],
    specs: dict[str, Any],
    as_of: object,
) -> dict[str, Any]:
    """K1 on the frozen holdout, K2 on closed forward trades, K3 on the 7."""
    start = FORWARD_START
    k1_rows: list[dict[str, float]] = []
    k3_codes: dict[str, Any] = {}
    for code, packed in curves.items():
        row = specs["codes"].get(code, {})
        universe = "study" if code in rules7.OVERRIDES else str(row.get("universe", "holdout"))
        strat = _window_mdd(packed["equity"], start)
        bh = _window_mdd(packed["bh"], start)
        if universe == "holdout":
            k1_rows.append({"strat_mdd": strat, "bh_mdd": bh})
        if universe == "study" and code not in rules7.NOT_APPLICABLE:
            sample = _num(row.get("in_sample_mdd"))
            k3_codes[code] = {
                "status": k3_status(strat, sample, bh),
                "fwd_mdd": strat,
                "sample_mdd": sample,
                "bh_mdd": bh,
            }
    k1 = k1_status(k1_rows)
    closed = [
        _num(r.get("ret_net"))
        for r in ledger_rows
        if r.get("segment") == "forward" and r.get("exit_reason") in CLOSED_REASONS
    ]
    k2 = k2_status(closed)
    for code in rules7.OVERRIDES:
        if code in rules7.NOT_APPLICABLE or code in k3_codes:
            continue
        sample = _num(specs.get("codes", {}).get(code, {}).get("in_sample_mdd"))
        k3_codes[code] = {"status": "未定", "fwd_mdd": None, "sample_mdd": sample, "bh_mdd": None}
    if not k3_codes:
        k3 = "未定"
    elif any(item["status"] == "不通过" for item in k3_codes.values()):
        k3 = "不通过"
    elif any(item["status"] == "未定" for item in k3_codes.values()):
        k3 = "未定"
    else:
        k3 = "通过"
    waiting = gate_open(as_of, int(k2["n"]))
    return {
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "spec_fp": spec_fingerprint(),
        "freeze_date": FREEZE_DATE,
        "forward_start": FORWARD_START,
        "overall": overall_status(k1["status"], k2["status"], k3, waiting=waiting),
        "waiting": waiting,
        "k1": k1,
        "k2": k2,
        "k3": {"status": k3, "codes": k3_codes},
        "gate_date": CRITERIA["gate_date"],
        "holdout_line": holdout_line(),
    }


def holdout_line(typ: str | None = None) -> str:
    shallower = HOLDOUT_BASELINE["mdd_shallower"]
    calmar = HOLDOUT_BASELINE["calmar_beats_bh"]
    text = f"留出：回撤更浅 {shallower[0]}/{shallower[1]}，Calmar 胜买入持有 {calmar[0]}/{calmar[1]}"
    by_type = HOLDOUT_BASELINE["by_type_calmar"]
    if typ and typ in by_type:
        won, total = by_type[typ]
        text += f"。{typ} {won}/{total}"
    return text


def _frame_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in LEDGER_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    if frame.empty:
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    return frame.loc[:, list(LEDGER_COLUMNS)].sort_values(["code", "signal_date"]).reset_index(drop=True)


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame_rows(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def update_daily(path: Path, rows: list[dict[str, Any]], as_of: object) -> Path:
    """Append one row per name. Replacing the same as-of, keeping older days."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = pd.DataFrame(rows)
    for col in DAILY_COLUMNS:
        if col not in fresh.columns:
            fresh[col] = ""
    fresh = fresh.loc[:, list(DAILY_COLUMNS)]
    day = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    fresh["as_of"] = day
    if path.is_file():
        old = pd.read_csv(path, dtype=str).fillna("")
        for col in DAILY_COLUMNS:
            if col not in old.columns:
                old[col] = ""
        old = old.loc[:, list(DAILY_COLUMNS)]
        old = old.loc[old["as_of"].astype(str) != day]
        out = pd.concat([old, fresh.astype(str)], ignore_index=True)
    else:
        out = fresh.astype(str)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_holdout(path: Path) -> Path:
    """Write the leave-out numbers once. A second call leaves the file alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    path.write_text(json.dumps(HOLDOUT_BASELINE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def render_report(verdict: dict[str, Any]) -> str:
    k1, k2, k3 = verdict["k1"], verdict["k2"], verdict["k3"]
    share = "还没有" if k1.get("share") is None else f"{k1['share']:.0%}"
    median = k2.get("median")
    median_txt = "还没有" if median is None or not np.isfinite(median) else f"{median:.2%}"
    need_n = max(int(CRITERIA["k2_min_trades"]) - int(k2["n"]), 0)
    lines = [
        f"# rules7 前向 {verdict['as_of']}",
        "",
        f"冻结日 {verdict['freeze_date']}。前向从 {verdict['forward_start']} 起。裁决：{verdict['overall']}。",
        "",
        f"- K1 回撤：{k1['status']}（买入持有回撤达 5% 的有 {k1['eligible']} 只，其中 rules7 回撤更浅的 {k1['shallower']} 只，占比 {share}；至少要 {int(CRITERIA['k1_min_eligible'])} 只才判）",
        f"- K2 笔数：{k2['status']}（前向已平仓 {k2['n']} 笔，赢 {k2['wins']} 笔，中位数 {median_txt}）",
        f"- K3 熔断：{k3['status']}",
        "",
        f"距总裁决还要：{need_n} 笔已平仓，并且日期到 {verdict['gate_date']}。"
        " K3 触发则不用等。",
        "",
        verdict["holdout_line"] + "。",
        "",
    ]
    tripped = [f"{code} {item['status']}" for code, item in sorted(k3["codes"].items()) if item["status"] == "不通过"]
    if tripped:
        lines.append("K3 不通过：" + "；".join(tripped) + "。")
        lines.append("")
    return "\n".join(lines)


def output_dir(listing: Path) -> Path:
    return Path(listing).resolve().parent / "rules7_forward"


def page_dir(listing: Path, as_of: object) -> Path:
    day = pd.Timestamp(as_of).strftime("%Y%m%d")
    return Path(listing).resolve().parent / f"{day}_forward"


def _replay_listing(
    listing: Path,
    as_of: object,
    specs: dict[str, Any],
    *,
    only: set[str] | None = None,
    htmls: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, pd.Series]], list[str]]:
    list_equity_html, parse_html_label, _build = _listing_helpers()
    if htmls is None:
        htmls = list_equity_html(Path(listing))
    rows: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    curves: dict[str, dict[str, pd.Series]] = {}
    errors: list[str] = []
    day = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    fp = spec_fingerprint()
    for html in htmls:
        code, name = parse_html_label(html)
        code = code.upper()
        if only is not None and code not in only:
            continue
        if code not in specs["codes"] and code not in rules7.OVERRIDES:
            continue
        try:
            frame = load_prepared(html, listing, as_of)
            packed = replay_frame(code, name, frame, specs=specs)
        except Exception as exc:  # noqa: BLE001 — one bad page should not drop the rest
            errors.append(f"{code} {name}: {exc}")
            continue
        rows.extend(packed["rows"])
        equity = packed["result"]["equity"]
        close = filled_close(packed["frame"])
        bh_level = close / float(close.dropna().iloc[0])
        curves[code] = {"equity": equity, "bh": bh_level}
        state = packed["result"]["state"]
        daily.append(
            {
                "as_of": day,
                "code": code,
                "holding": "1" if state.get("holding") else "0",
                "action": str(state.get("action") or ""),
                "equity": float(equity.iloc[-1]),
                "bh_equity": float(bh_level.iloc[-1]),
                "spec_fp": fp,
            }
        )
    return rows, daily, curves, errors


def run(listing: Path, as_of: object, *, out_dir: Path | None = None) -> dict[str, Any]:
    """Replay the frozen universe and write the ledger, the daily strip, and the report."""
    listing = Path(listing)
    if not listing.is_dir():
        raise FileNotFoundError(listing)
    dest = Path(out_dir) if out_dir is not None else output_dir(listing)
    list_equity_html, _parse, _build = _listing_helpers()
    htmls = list_equity_html(listing)
    if not htmls:
        return {"code": 2, "error": f"no equity html under {listing}; run etf-daily listing first"}
    specs = load_specs()
    rows, daily, curves, errors = _replay_listing(listing, as_of, specs, htmls=htmls)
    verdict = assess(curves, rows, specs, as_of)
    dest.mkdir(parents=True, exist_ok=True)
    ledger_path = write_ledger(dest / "rules7_trades_ledger.csv", rows)
    daily_path = update_daily(dest / "rules7_forward_daily.csv", daily, as_of)
    holdout_path = write_holdout(dest / "rules7_holdout_baseline.json")
    verdict_path = dest / "rules7_forward_verdict.json"
    verdict_path.write_text(json.dumps(_clean_json(verdict), ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    day = pd.Timestamp(as_of).strftime("%Y%m%d")
    report_path = dest / f"forward_report_{day}.md"
    report = render_report(verdict)
    if errors:
        report += "\n没有重放成的标的：\n\n" + "\n".join(f"- {line}" for line in errors) + "\n"
    report_path.write_text(report, encoding="utf-8")
    return {
        "code": 0,
        "verdict": verdict,
        "errors": errors,
        "n_rows": len(rows),
        "n_names": len(curves),
        "ledger": ledger_path,
        "daily": daily_path,
        "holdout": holdout_path,
        "verdict_path": verdict_path,
        "report": report_path,
        "out_dir": dest,
    }


def _clean_json(value: Any) -> Any:
    """Turn NaN and inf into None so the file is valid JSON."""
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    return value


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(type(value))


def write_freeze_specs(
    listing: Path,
    as_of: object,
    path: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Lock each name's type from the frame available on ``as_of``. Refuses to overwrite."""
    dest = Path(path) if path is not None else SPECS_PATH
    if dest.is_file() and not overwrite:
        raise FileExistsError(dest)
    list_equity_html, parse_html_label, _build = _listing_helpers()
    codes: dict[str, Any] = {}
    for html in list_equity_html(Path(listing)):
        code, name = parse_html_label(html)
        code = code.upper()
        try:
            frame = load_prepared(html, listing, as_of)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {code} {exc}")
            continue
        if code not in rules7.OVERRIDES and len(frame) < MIN_HOLDOUT_BARS:
            continue
        feat = rules7.classify_features(frame)
        typ, source, spec = rules7.resolve_spec(code, feat)
        universe = "study" if code in rules7.OVERRIDES else "holdout"
        entry: dict[str, Any] = {
            "name": name,
            "type": typ,
            "source": source,
            "universe": universe,
            "bars": int(len(frame)),
            "applicable": code not in rules7.NOT_APPLICABLE,
        }
        if universe == "study" and entry["applicable"]:
            masks = rules7.build_masks(frame)
            buy, sell = rules7.buy_sell_masks(masks, spec)
            result = rules7.run_state_machine(
                filled_close(frame),
                buy,
                sell,
                g1=masks["g1"] if spec.g1_hold else None,
                trail=spec.trail,
                cost=COST,
            )
            entry["in_sample_total"] = result["total"]
            entry["in_sample_mdd"] = result["mdd"]
            entry["in_sample_closed"] = len(result["trades"])
            entry["in_sample_open"] = bool(result["state"]["holding"])
        codes[code] = entry
        print(f"[spec] {code} {typ} {universe} bars={len(frame)}", flush=True)
    payload = {
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "logic": LOGIC_EQUIVALENT_TO,
        "freeze_tag": FREEZE_TAG,
        "spec_fp": spec_fingerprint(),
        "codes": {code: codes[code] for code in sorted(codes)},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest
