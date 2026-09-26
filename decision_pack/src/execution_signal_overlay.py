"""Per-symbol HTML-parity signal overlay for adaptive execution packs.

Mirrors markers drawn on ``*_adaptive.html``:
- green/red triangles: hold_up BUY / SELL
- hollow diamonds: early regime marks (V / ↑ / X / ↓)
- ED: early-down diagnostic alert / watch

Does **not** change buy/sell actions — display / ops annotation only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# HTML legend parity
EARLY_MARK_META: dict[str, dict[str, str]] = {
    "leave_down": {
        "glyph": "◇V",
        "label": "提前离↓(V)",
        "color": "橙",
        "html_marker": "diamond-open orange",
    },
    "enter_up": {
        "glyph": "◇↑",
        "label": "提前进↑",
        "color": "绿",
        "html_marker": "diamond-open green",
    },
    "leave_up": {
        "glyph": "◇X",
        "label": "提前离↑(X)",
        "color": "紫",
        "html_marker": "diamond-open purple",
    },
    "enter_down": {
        "glyph": "◇↓",
        "label": "提前进↓",
        "color": "红",
        "html_marker": "diamond-open red",
    },
}

SIGNAL_BOARD_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "method",
    "regime_asof",
    "signals_compact",
    "triangle_buy",
    "triangle_sell",
    "diamond_V",
    "diamond_up",
    "diamond_X",
    "diamond_down",
    "ed_level",
    "ed_run",
    "ed_reasons",
    "signals_detail",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _norm_date(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        text = str(value).strip()
        return text[:10] if len(text) >= 10 else (text or None)


def load_symbol_trade_signals(trades_dir: Path, code: str) -> dict[str, pd.DataFrame]:
    code_u = str(code).upper()
    return {
        "trades": _read_csv(trades_dir / f"{code_u}_trades.csv"),
        "early_regime": _read_csv(trades_dir / f"{code_u}_early_regime_marks.csv"),
        "early_down": _read_csv(trades_dir / f"{code_u}_early_down_marks.csv"),
        "recovery_up": _read_csv(trades_dir / f"{code_u}_recovery_up_marks.csv"),
    }


def _filter_dates(frame: pd.DataFrame, col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if frame is None or frame.empty or col not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["_d"] = pd.to_datetime(out[col], errors="coerce")
    out = out[out["_d"].notna()]
    if start:
        out = out[out["_d"] >= pd.Timestamp(start)]
    if end:
        out = out[out["_d"] <= pd.Timestamp(end)]
    return out.drop(columns=["_d"])


def _fmt_trade_marks(trades: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (buy_bits, sell_bits, compact_parts)."""
    buys: list[str] = []
    sells: list[str] = []
    compact: list[str] = []
    if trades is None or trades.empty or "side" not in trades.columns:
        return buys, sells, compact
    work = trades.copy()
    work["date"] = work.get("date", pd.Series(dtype=str)).map(_norm_date)
    for _, r in work.sort_values("date").iterrows():
        d = r.get("date")
        if not d:
            continue
        side = str(r.get("side") or "").upper()
        px = r.get("px")
        px_s = f"@{float(px):.3f}" if pd.notna(px) else ""
        if side == "BUY":
            bit = f"🟢▲B {d}{px_s}"
            buys.append(bit)
            compact.append(f"▲B{d[5:]}")
        elif side == "SELL":
            bit = f"🔴▼S {d}{px_s}"
            sells.append(bit)
            compact.append(f"▼S{d[5:]}")
    return buys, sells, compact


def _fmt_early_marks(early: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    by_kind: dict[str, list[str]] = {k: [] for k in EARLY_MARK_META}
    compact: list[str] = []
    if early is None or early.empty:
        return by_kind, compact
    work = early.copy()
    if "date" not in work.columns:
        return by_kind, compact
    work["date"] = work["date"].map(_norm_date)
    for _, r in work.sort_values("date").iterrows():
        kind = str(r.get("kind") or "")
        meta = EARLY_MARK_META.get(kind)
        if not meta:
            continue
        d = r.get("date")
        if not d:
            continue
        mv = r.get("move")
        mv_s = f" {float(mv):+.1%}" if pd.notna(mv) else ""
        bit = f"{meta['glyph']}({meta['color']}) {d}{mv_s}"
        by_kind[kind].append(bit)
        compact.append(f"{meta['glyph']}{d[5:]}")
    return by_kind, compact


def _fmt_ed_marks(ed: pd.DataFrame, *, as_of: str | None = None) -> tuple[str, str, str, list[str]]:
    """Return level, run, reasons, compact parts for ED overlays."""
    if ed is None or ed.empty:
        return "none", "", "", []
    work = ed.copy()
    for col in ("date", "end"):
        if col in work.columns:
            work[col] = work[col].map(_norm_date)
    # Prefer run covering as_of; else latest run by end/date.
    active = None
    if as_of and "date" in work.columns:
        a = pd.Timestamp(as_of)
        for _, r in work.iterrows():
            d0 = r.get("date")
            d1 = r.get("end") or d0
            if not d0:
                continue
            if pd.Timestamp(d0) <= a <= pd.Timestamp(d1 or d0):
                active = r
                break
    if active is None:
        sort_col = "end" if "end" in work.columns else "date"
        work = work.sort_values(sort_col)
        active = work.iloc[-1]
    level = str(active.get("level") or active.get("kind") or "none")
    if level.startswith("ed_"):
        level = level.replace("ed_", "")
    d0 = active.get("date") or ""
    d1 = active.get("end") or d0
    run = f"{d0}→{d1}" if d0 else ""
    reasons = str(active.get("reasons") or "")
    if level == "alert":
        compact = [f"⚠ED{str(d0)[5:]}" if d0 else "⚠ED"]
        level_out = "alert"
    elif level == "watch":
        compact = [f"ED{str(d0)[5:]}" if d0 else "ED"]
        level_out = "watch"
    else:
        compact = []
        level_out = "none"
    return level_out, run, reasons, compact


def _fmt_ru_marks(ru: pd.DataFrame, *, as_of: str | None = None) -> tuple[str, str, list[str]]:
    """Return level, run, compact parts for recovery_up diagnostic overlays."""
    if ru is None or ru.empty:
        return "none", "", []
    work = ru.copy()
    for col in ("date", "end"):
        if col in work.columns:
            work[col] = work[col].map(_norm_date)
    active = None
    if as_of and "date" in work.columns:
        a = pd.Timestamp(as_of)
        for _, r in work.iterrows():
            d0 = r.get("date")
            d1 = r.get("end") or d0
            if not d0:
                continue
            if pd.Timestamp(d0) <= a <= pd.Timestamp(d1 or d0):
                active = r
                break
    if active is None:
        sort_col = "end" if "end" in work.columns else "date"
        work = work.sort_values(sort_col)
        active = work.iloc[-1]
    d0 = active.get("date") or ""
    d1 = active.get("end") or d0
    run = f"{d0}→{d1}" if d0 else ""
    compact = [f"RU{str(d0)[5:]}" if d0 else "RU"]
    return "alert", run, compact


def build_symbol_signal_row(
    *,
    code: str,
    name: str,
    method: str,
    regime_asof: str,
    trades_dir: Path,
    window_start: str | None = None,
    window_end: str | None = None,
    as_of: str | None = None,
    early_down_asof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packs = load_symbol_trade_signals(trades_dir, code)
    trades = _filter_dates(packs["trades"], "date", window_start, window_end)
    early = _filter_dates(packs["early_regime"], "date", window_start, window_end)
    ed_hist = packs["early_down"]
    # ED runs: keep if overlap window
    if not ed_hist.empty and (window_start or window_end):
        tmp = ed_hist.copy()
        if "date" in tmp.columns:
            tmp["_d0"] = pd.to_datetime(tmp["date"], errors="coerce")
            end_col = "end" if "end" in tmp.columns else "date"
            tmp["_d1"] = pd.to_datetime(tmp[end_col], errors="coerce")
            lo = pd.Timestamp(window_start) if window_start else tmp["_d0"].min()
            hi = pd.Timestamp(window_end) if window_end else tmp["_d1"].max()
            tmp = tmp[(tmp["_d1"] >= lo) & (tmp["_d0"] <= hi)]
            ed_hist = tmp.drop(columns=[c for c in ("_d0", "_d1") if c in tmp.columns])

    buys, sells, c_trade = _fmt_trade_marks(trades)
    by_kind, c_early = _fmt_early_marks(early)
    ed_level, ed_run, ed_reasons, c_ed = _fmt_ed_marks(ed_hist, as_of=as_of)
    ru_hist = packs.get("recovery_up")
    if ru_hist is not None and not ru_hist.empty and (window_start or window_end):
        tmp = ru_hist.copy()
        if "date" in tmp.columns:
            tmp["_d0"] = pd.to_datetime(tmp["date"], errors="coerce")
            end_col = "end" if "end" in tmp.columns else "date"
            tmp["_d1"] = pd.to_datetime(tmp[end_col], errors="coerce")
            lo = pd.Timestamp(window_start) if window_start else tmp["_d0"].min()
            hi = pd.Timestamp(window_end) if window_end else tmp["_d1"].max()
            tmp = tmp[(tmp["_d1"] >= lo) & (tmp["_d0"] <= hi)]
            ru_hist = tmp.drop(columns=[c for c in ("_d0", "_d1") if c in tmp.columns])
    ru_level, ru_run, c_ru = _fmt_ru_marks(ru_hist, as_of=as_of)
    # Prefer live early_down as-of row when provided (execution pack diagnostic).
    if early_down_asof:
        lvl = str(early_down_asof.get("level") or "none")
        if lvl in {"alert", "watch"}:
            ed_level = lvl
            ed_reasons = str(early_down_asof.get("reasons") or ed_reasons)
            if not ed_run:
                ed_run = str(early_down_asof.get("as_of") or as_of or "")
            mark = "⚠ED" if lvl == "alert" else "ED"
            if mark not in "".join(c_ed):
                c_ed = [mark] + c_ed

    detail_parts: list[str] = []
    detail_parts.extend(buys)
    detail_parts.extend(sells)
    for kind in ("leave_down", "enter_up", "leave_up", "enter_down"):
        detail_parts.extend(by_kind.get(kind) or [])
    if ed_level == "alert":
        detail_parts.append(f"⚠ED告警 {ed_run} {ed_reasons}".strip())
    elif ed_level == "watch":
        detail_parts.append(f"ED观察 {ed_run} {ed_reasons}".strip())
    if ru_level == "alert":
        detail_parts.append(f"RU恢复上涨(诊断) {ru_run}".strip())

    compact = " ".join(c_trade + c_early + c_ed + c_ru) or "—"
    return {
        "code": str(code).upper(),
        "name": str(name),
        "method": str(method),
        "regime_asof": str(regime_asof),
        "signals_compact": compact,
        "triangle_buy": " | ".join(buys),
        "triangle_sell": " | ".join(sells),
        "diamond_V": " | ".join(by_kind.get("leave_down") or []),
        "diamond_up": " | ".join(by_kind.get("enter_up") or []),
        "diamond_X": " | ".join(by_kind.get("leave_up") or []),
        "diamond_down": " | ".join(by_kind.get("enter_down") or []),
        "ed_level": ed_level,
        "ed_run": ed_run,
        "ed_reasons": ed_reasons,
        "ru_level": ru_level,
        "ru_run": ru_run,
        "signals_detail": " ； ".join(detail_parts),
    }


def build_signal_board(
    plan_frame: pd.DataFrame,
    *,
    adaptive_dir: Path,
    window_start: str | None,
    window_end: str | None,
    as_of: str | None,
    early_down: pd.DataFrame | None = None,
) -> pd.DataFrame:
    trades_dir = Path(adaptive_dir) / "trades"
    ed_map: dict[str, dict[str, Any]] = {}
    if early_down is not None and len(early_down):
        for _, r in early_down.iterrows():
            ed_map[str(r.get("code") or "").upper()] = r.to_dict()

    rows: list[dict[str, Any]] = []
    if plan_frame is None or plan_frame.empty:
        return pd.DataFrame(columns=list(SIGNAL_BOARD_COLUMNS))
    for _, r in plan_frame.iterrows():
        code = str(r.get("code") or "").upper()
        rows.append(
            build_symbol_signal_row(
                code=code,
                name=str(r.get("name") or code),
                method=str(r.get("method") or ""),
                regime_asof=str(r.get("regime_asof") or ""),
                trades_dir=trades_dir,
                window_start=window_start,
                window_end=window_end,
                as_of=as_of,
                early_down_asof=ed_map.get(code),
            )
        )
    out = pd.DataFrame(rows)
    return out.reindex(columns=list(SIGNAL_BOARD_COLUMNS))


def render_signal_board_markdown(board: pd.DataFrame, *, meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    as_of = str(meta.get("as_of") or "")
    trade_date = str(meta.get("trade_date") or "")
    window = str(meta.get("signal_window") or f"{as_of}..{trade_date}")
    lines = [
        "## 信号看板（与 HTML 标记对齐 · 不改买卖动作）",
        "",
        f"> 窗口 `{window}` · 🟢▲买入三角 / 🔴▼卖出三角 · "
        "空心菱形：橙◇V离↓ / 绿◇↑进涨 / 紫◇X离↑ / 红◇↓进跌 · "
        "⚠ED告警 / ED观察",
        "",
    ]
    if board is None or board.empty:
        lines.extend(["（无信号）", ""])
        return "\n".join(lines)

    def _section(title: str, mask: pd.Series) -> None:
        lines.extend([f"### {title}", ""])
        sub = board.loc[mask]
        if sub.empty:
            lines.extend(["（无）", ""])
            return
        lines.append("| 代码 | 名称 | regime | 标记 |")
        lines.append("|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.get('code')} | {r.get('name')} | {r.get('regime_asof')} | "
                f"{r.get('signals_compact') or '—'} |"
            )
        lines.append("")

    has = board["signals_compact"].fillna("—").ne("—")
    _section("有信号的标的", has)
    _section("🟢▲ / 🔴▼ 三角（买卖）", board["triangle_buy"].fillna("").ne("") | board["triangle_sell"].fillna("").ne(""))
    _section(
        "空心菱形提前标（V/↑/X/↓）",
        board["diamond_V"].fillna("").ne("")
        | board["diamond_up"].fillna("").ne("")
        | board["diamond_X"].fillna("").ne("")
        | board["diamond_down"].fillna("").ne(""),
    )
    _section("ED 告警/观察", board["ed_level"].isin(["alert", "watch"]))

    lines.extend(
        [
            "### 全市场明细（信号）",
            "",
            "| 代码 | 名称 | 方法 | regime | 压缩标记 | 明细 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for _, r in board.iterrows():
        detail = str(r.get("signals_detail") or "—").replace("|", "/")
        if len(detail) > 120:
            detail = detail[:117] + "..."
        lines.append(
            f"| {r.get('code')} | {r.get('name')} | {r.get('method')} | {r.get('regime_asof')} | "
            f"{r.get('signals_compact') or '—'} | {detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_signal_board_artifacts(
    board: pd.DataFrame,
    out_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "signal_board.csv"
    json_path = out_dir / "signal_board.json"
    md_path = out_dir / "signal_board.md"
    board.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "meta": dict(meta or {}),
        "legend": {
            "triangle_buy": "绿色实心三角 ▲B = hold_up BUY",
            "triangle_sell": "红色实心三角 ▼S = hold_up SELL",
            "diamonds": EARLY_MARK_META,
            "ed_alert": "ED告警(模型滞后) hexagon",
            "ed_watch": "ED观察(已对齐↓)",
        },
        "rows": board.replace({pd.NA: None}).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(
        "# 信号看板\n\n" + render_signal_board_markdown(board, meta=meta),
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


__all__ = [
    "EARLY_MARK_META",
    "SIGNAL_BOARD_COLUMNS",
    "build_signal_board",
    "build_symbol_signal_row",
    "load_symbol_trade_signals",
    "render_signal_board_markdown",
    "write_signal_board_artifacts",
]
