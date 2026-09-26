#!/usr/bin/env python3
"""EOD scan: fig9 aux_edge buy/sell + watchlist for a listing batch.

Long-only. Signal close → next-bar close fill. Research scan only
(does not change production hold_up).

Writes into --listing-dir:
  fig9_aux_edge_eod_YYYYMMDD.md
  fig9_aux_edge_eod_YYYYMMDD.csv

Optional HRP cluster join (--cluster-csv; default prefers
decision_packs/{asof}/cluster_representatives_{asof}_d080.csv, dist≈0.8)
appends a per-cluster pick section to the md.

Run from repo root (this project's Python)::

    cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
    PYTHONPATH=. ~/etf-daily-output/python/envs/py312/bin/python \\
      decision_pack/scripts/scan_fig9_aux_edge_eod.py \\
      --listing-dir ~/etf-daily-output/temp/plotly_outputs/20260827_from_listing \\
      --as-of 2026-08-27 --skip-html

换日：改 --listing-dir 和 --as-of（需已有该日 from_listing 的 adaptive HTML + listing_dates.csv）。
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    _extract_plotly_traces,
    find_regime_html,
)
from decision_pack.src.incremental_from_listing import _trace_dates  # noqa: E402

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260827_from_listing"
DEFAULT_CLUSTER_PACK_ROOT = Path("~/etf-daily-output/temp/decision_packs")
_HTML_STEM = re.compile(
    r"^regime_transition_([A-Z]{2}\d+)_(.+)_(\d{8})_(\d{8})_adaptive$"
)
POS_STRETCH = 0.80

# Documented per-name overlays (annotation only; pool rule stays aux_edge).
DOC_NOTES: dict[str, str] = {
    "SH512530": "文档=同一套 aux_edge",
    "SH515220": "文档=EWMA L0+触上离开（非池内 aux_edge）",
    "SH513050": "文档默认 aux_fail_lo（g>0 才买；再触下轨卖；失败后须见近上才再买）",
    "SH515880": "文档默认 aux_extrema 买 + 骑轨卖（对照只跑 aux_extrema 立刻卖）",
    "SZ159985": "文档=图9 L2+S1（图10 浅贴仅对照）",
    "SH518880": "文档：震荡用图9 aux_edge，趋势改图7/8，勿把趋势当震荡卖",
    "SH510300": "文档=aux_edge + 图7不溢价（gap≤0 或 pos<0.70）",
}


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--as-of", default="2026-08-27")
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument(
        "--skip-html",
        action="store_true",
        help="accepted for CLI compatibility; this scan never writes overlays",
    )
    p.add_argument(
        "--cluster-csv",
        type=Path,
        default=None,
        help=(
            "HRP cluster_representatives CSV. "
            "Default: prefer *_d080.csv (dist≈0.8), then *_k12.csv, then untagged; "
            "missing → skip section"
        ),
    )
    p.add_argument(
        "--rewrite-md-from-csv",
        action="store_true",
        help=(
            "Skip fig9 rescan; rewrite md from existing "
            "fig9_aux_edge_eod_{asof}.csv + fig1 markers"
        ),
    )
    return p.parse_args(argv)


def default_cluster_csv(as_of: pd.Timestamp | str) -> Path:
    """Prefer coarse dist≈0.8 reps for fig9 summary; then k12; then DIST_T cut."""
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    pack = DEFAULT_CLUSTER_PACK_ROOT / tag
    for name in (
        f"cluster_representatives_{tag}_d080.csv",
        f"cluster_representatives_{tag}_k12.csv",
        f"cluster_representatives_{tag}.csv",
    ):
        cand = pack / name
        if cand.is_file():
            return cand
    return pack / f"cluster_representatives_{tag}_d080.csv"


def load_cluster_frame(path: Path | None) -> pd.DataFrame | None:
    """Load cluster_representatives CSV; None if missing/unreadable."""
    if path is None:
        return None
    path = _resolve(path)
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or "code" not in df.columns:
        return None
    out = df.copy()
    out["code"] = out["code"].astype(str).str.upper()
    if "is_representative" in out.columns:
        out["is_representative"] = out["is_representative"].map(
            lambda x: True
            if x is True or str(x).strip().lower() in {"true", "1", "yes"}
            else False
        )
    return out


def _group_zh(group: object) -> str:
    g = str(group or "").strip().lower()
    if g == "up":
        return "涨"
    if g == "down":
        return "跌"
    return str(group or "—")


def _truthy_in_pos(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    return str(v).strip().lower() in {"true", "1", "yes"}


def format_hrp_cluster_section(
    rows: list[dict[str, Any]],
    cluster: pd.DataFrame | None,
    *,
    cluster_path: Path | None = None,
) -> list[str]:
    """Markdown lines for ## 按 HRP 大类（约 k 簇）.

    One pick per ``cluster``: among ``is_representative`` members, the one with
    largest ``|ret20|`` that appears in the scan. Clusters with no representative
    are listed as 无明确方向.
    """
    lines = [
        "## 按 HRP 大类（粗簇）",
        "",
        "默认读 `cluster_representatives_*_d080.csv`（Ward 距离阈值≈0.8 自然切簇）。"
        "每簇挑一只 **|ret20| 最大的聚类代表股**；无代表则只展示。",
        "",
    ]
    if cluster is None or cluster.empty:
        note = (
            f"（未找到 cluster_representatives，跳过大类汇总"
            f"{f': {cluster_path}' if cluster_path else ''}）"
        )
        lines += [note, ""]
        return lines

    scan = pd.DataFrame(rows)
    if scan.empty or "code" not in scan.columns:
        lines += ["（扫描为空）", ""]
        return lines
    scan = scan.copy()
    scan["code"] = scan["code"].astype(str).str.upper()
    scan_cols = {
        "code": "code",
        "name": "scan_name",
        "action": "action",
        "watch": "watch",
        "fig9_pos": "fig9_pos",
        "in_pos": "in_pos",
        "unrealized": "unrealized",
        "doc_action": "doc_action",
    }
    slim = scan[[c for c in scan_cols if c in scan.columns]].rename(
        columns={c: scan_cols[c] for c in scan_cols if c in scan.columns and c != "code"}
    )
    merged = cluster.merge(slim, on="code", how="left")
    if "scan_name" in merged.columns:
        merged["name"] = merged["scan_name"].where(
            merged["scan_name"].notna()
            & (merged["scan_name"].astype(str).str.strip() != ""),
            merged.get("name"),
        )

    clustered_codes = set(cluster["code"].astype(str).str.upper())
    unclustered = scan.loc[~scan["code"].isin(clustered_codes)].copy()

    lines += [
        "| 大类 | 挑选(代表) | ret20 | 动作 | 看盘 | 图9分位 | 持仓 | 未实现 | 组内扫描 |",
        "|---|---|---:|---|---|---:|---|---:|---|",
    ]

    for cid, sub in merged.groupby("cluster", sort=True):
        sub = sub.copy()
        # Unique codes in this cluster (cluster CSV may list up+down rows)
        n_members = int(sub["code"].nunique())
        if "n_members" in sub.columns:
            nm = pd.to_numeric(sub["n_members"], errors="coerce").dropna()
            if not nm.empty:
                n_members = int(nm.iloc[0])

        is_rep = (
            sub["is_representative"].fillna(False).astype(bool)
            if "is_representative" in sub.columns
            else pd.Series(False, index=sub.index)
        )
        reps = sub.loc[is_rep].copy()
        pick_code = ""
        pick_name = ""
        pick_action = ""
        pick_watch = ""
        pick_pos = None
        pick_in = False
        pick_unreal = None
        pick_ret = None
        if not reps.empty:
            reps = reps.assign(
                _abs=pd.to_numeric(reps["ret20_pct"], errors="coerce").abs()
            ).sort_values("_abs", ascending=False)
            # Prefer a rep that is in the scan
            for _, r0 in reps.iterrows():
                if pd.notna(r0.get("action")):
                    pick_code = str(r0["code"])
                    pick_name = str(r0.get("name") or "")
                    pick_action = str(r0.get("action") or "")
                    pick_watch = (
                        str(r0.get("watch") or "")
                        if pd.notna(r0.get("watch"))
                        else ""
                    )
                    pick_pos = _f(r0.get("fig9_pos"))
                    pick_in = _truthy_in_pos(r0.get("in_pos"))
                    pick_unreal = _f(r0.get("unrealized"))
                    pick_ret = _f(r0.get("ret20_pct"))
                    break

        # Member blurb: unique codes in scan, sorted by |ret20|
        in_scan = sub.loc[sub["action"].notna()].copy()
        if not in_scan.empty:
            in_scan = in_scan.drop_duplicates(subset=["code"], keep="first")
            if "ret20_pct" in in_scan.columns:
                in_scan = in_scan.assign(
                    _abs=pd.to_numeric(in_scan["ret20_pct"], errors="coerce").abs()
                ).sort_values("_abs", ascending=False)
        members_bits: list[str] = []
        for _, m in in_scan.iterrows():
            nm = str(m.get("name") or "").strip()
            bit = nm if nm else str(m["code"])
            gzh = _group_zh(m.get("group"))
            if gzh in {"涨", "跌"}:
                bit += gzh
            act = str(m.get("action") or "")
            w = str(m.get("watch") or "") if pd.notna(m.get("watch")) else ""
            if act:
                bit += f" {act}"
            if w:
                bit += f"/{w}"
            if _truthy_in_pos(m.get("in_pos")):
                bit += " 持"
            members_bits.append(bit)
        members_txt = "；".join(members_bits) if members_bits else "（组内无扫描行）"

        label = f"簇{int(cid)}"
        if pick_code:
            pick_cell = f"**{pick_code}** {pick_name}".strip()
            ret_cell = f"{pick_ret:.1f}%" if pick_ret is not None else "—"
            pos_cell = _pct(pick_pos, 0)
            unreal_cell = _pct(pick_unreal)
            in_cell = "是" if pick_in else ""
            action_cell = pick_action or "—"
            watch_cell = pick_watch
        elif reps.empty:
            pick_cell = "（无明确方向）"
            action_cell = watch_cell = pos_cell = unreal_cell = in_cell = "—"
            ret_series = pd.to_numeric(sub.get("ret20_pct"), errors="coerce")
            ret_raw = _f(ret_series.abs().max()) if ret_series is not None else None
            ret_cell = f"max={ret_raw:.1f}%" if ret_raw is not None else "—"
        else:
            pick_cell = "（代表未进扫描）"
            ret_cell = action_cell = watch_cell = pos_cell = unreal_cell = in_cell = "—"

        lines.append(
            f"| {label}（n={n_members}） | {pick_cell} | {ret_cell} | "
            f"{action_cell} | {watch_cell} | {pos_cell} | {in_cell} | {unreal_cell} | "
            f"{members_txt} |"
        )

    lines.append("")
    if not unclustered.empty:
        codes = ", ".join(sorted(unclustered["code"].astype(str).tolist()))
        lines += [f"未聚类扫描代码：{codes}", ""]
    if cluster_path is not None:
        lines += [f"聚类文件：`{cluster_path}`", ""]
    return lines


def _name_from_html(html: Path, code: str) -> str:
    m = _HTML_STEM.match(html.stem)
    if m:
        return str(m.group(2))
    return code


_FIG1_YAXES = {"", "y", "y1"}
_EXCLUDE_MARKER_NAMES = ("持仓未平",)
# ED scatter hover: "ED告警(模型滞后) 2026-08-13→2026-09-03<br>..."
_ED_RUN_HOVER_RE = re.compile(
    r"(?P<start>\d{4}-\d{2}-\d{2})\s*→\s*(?P<end>\d{4}-\d{2}-\d{2})"
)


def _fig1_marker_label(name: Any, text_on: list[Any]) -> str | None:
    """Build display label; None means exclude (e.g. 持仓未平)."""
    nm = str(name or "").strip()
    if any(ex in nm for ex in _EXCLUDE_MARKER_NAMES):
        return None
    short: str | None = None
    for t in text_on:
        if t is None:
            continue
        s = str(t).strip()
        if not s or s == nm or s == "持":
            continue
        short = s
        break
    if not nm:
        return short
    if short and short not in nm:
        return f"{nm}({short})"
    return nm


def _ed_end_label(trace_name: str) -> str:
    nm = str(trace_name or "")
    if "观察" in nm:
        return "ED观察结束"
    if "告警" in nm:
        return "ED告警结束"
    return f"{nm}结束" if nm else "ED结束"


def _fig1_ed_end_labels(traces: list[dict[str, Any]], asof: pd.Timestamp) -> list[str]:
    """ED run ends on as-of (from hover start→end); skip same-day 1-bar runs."""
    asof_s = asof.strftime("%Y-%m-%d")
    labels: list[str] = []
    seen: set[str] = set()
    for tr in traces:
        mode = str(tr.get("mode") or "")
        if "markers" not in mode:
            continue
        yax = str(tr.get("yaxis") or "y")
        if yax not in _FIG1_YAXES:
            continue
        nm = str(tr.get("name") or "")
        if not nm.startswith("ED"):
            continue
        hover = tr.get("hovertext")
        items: list[Any]
        if isinstance(hover, (list, tuple)):
            items = list(hover)
        elif hover is not None:
            items = [hover]
        else:
            continue
        for h in items:
            m = _ED_RUN_HOVER_RE.search(str(h or ""))
            if not m:
                continue
            start_s, end_s = m.group("start"), m.group("end")
            if end_s != asof_s or start_s == end_s:
                continue
            label = _ed_end_label(nm)
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels


def fig1_markers_on_asof(html: str | Path, asof: pd.Timestamp | str) -> list[str]:
    """Event marker names on fig1 whose x includes as-of (excl. 持仓未平).

    Also appends ED 结束 labels when an ED run's hover end date equals as-of
    (scatter points exist only on run start; end is recovered from hover).
    """
    asof = pd.Timestamp(asof).normalize()
    if isinstance(html, Path):
        txt = html.read_text(encoding="utf-8")
    else:
        txt = html
    traces = _extract_plotly_traces(txt)
    labels: list[str] = []
    seen: set[str] = set()
    for tr in traces:
        mode = str(tr.get("mode") or "")
        if "markers" not in mode:
            continue
        yax = str(tr.get("yaxis") or "y")
        if yax not in _FIG1_YAXES:
            continue
        dates = _trace_dates(tr)
        if dates.empty:
            continue
        mask = dates == asof
        if not bool(mask.any()):
            continue
        text = tr.get("text")
        text_on: list[Any] = []
        if isinstance(text, (list, tuple)):
            for i, hit in enumerate(mask):
                if hit and i < len(text):
                    text_on.append(text[i])
        elif text is not None:
            text_on.append(text)
        label = _fig1_marker_label(tr.get("name"), text_on)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    for label in _fig1_ed_end_labels(traces, asof):
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def collect_listing_fig1_markers(
    listing_dir: Path,
    asof: pd.Timestamp | str,
) -> list[dict[str, Any]]:
    """Scan adaptive HTMLs for fig1 markers on as-of; one row per symbol."""
    listing_dir = Path(listing_dir)
    asof = pd.Timestamp(asof).normalize()
    rows: list[dict[str, Any]] = []
    for html in sorted(listing_dir.glob("regime_transition_*_adaptive.html")):
        stem = html.stem
        if "_fig9" in stem or "_oracle" in stem:
            continue
        m = _HTML_STEM.match(stem)
        if not m:
            continue
        code = str(m.group(1)).upper()
        name = str(m.group(2))
        try:
            markers = fig1_markers_on_asof(html, asof)
            markers_txt = "、".join(markers) if markers else "无"
        except Exception as exc:
            markers_txt = f"解析失败: {type(exc).__name__}: {exc}"
        rows.append({"code": code, "name": name, "markers": markers_txt})
    rows.sort(
        key=lambda r: (
            0 if r["markers"] not in ("无",) and not str(r["markers"]).startswith("解析失败") else 1,
            r["code"],
        )
    )
    return rows


def format_fig1_markers_section(
    as_of: str,
    fig1_markers: list[dict[str, Any]] | None,
) -> list[str]:
    if not fig1_markers:
        return []
    lines = [
        f"## 图1当日新标识（{as_of}）",
        "",
        "| 代码 | 名称 | 图1新标识 |",
        "|---|---|---|",
    ]
    for r in fig1_markers:
        lines.append(
            f"| {r.get('code') or ''} | {r.get('name') or ''} | {r.get('markers') or '无'} |"
        )
    lines.append("")
    return lines


def _bool(v: Any) -> bool:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return False
    return bool(v)


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}%}"


def _md_cell(v: Any) -> str:
    """Blank-out pandas/CSV NaN so rewrite-from-csv does not print 'nan'."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    return s


def _streak_true(s: pd.Series) -> int:
    n = 0
    for v in reversed(s.fillna(False).astype(bool).tolist()):
        if not v:
            break
        n += 1
    return n


def _open_trade(trades: pd.DataFrame) -> dict[str, Any] | None:
    if trades is None or trades.empty:
        return None
    last = trades.iloc[-1]
    if bool(last.get("open", False)) or pd.isna(last.get("signal_sell")):
        return last.to_dict()
    return None


def _doc_action(
    code: str,
    *,
    bt,
    sm,
    frame: pd.DataFrame,
    in_pos: bool,
    buy_today: bool,
    sell_today: bool,
    action: str,
    row: pd.Series,
) -> tuple[str, str]:
    note = DOC_NOTES.get(code, "")
    if code not in DOC_NOTES:
        return "", ""
    if code in ("SH512530", "SH515220"):
        return action, note
    if code in ("SZ159985", "SH518880"):
        return action, note
    if code == "SH510300":
        gap = _f(row.get("fig7_gap"))
        pos7 = _f(row.get("fig7_pos"))
        ok = (gap is not None and gap <= 0.0) or (pos7 is not None and pos7 < 0.70)
        extra = (
            f"图7 gap={_pct(gap)} pos={_pct(pos7, 0)}"
            + (" 过不溢价" if ok else " 未过不溢价")
        )
        if action == "BUY" and not ok:
            return "FLAT", f"{note}；今日买被图7挡：{extra}"
        return action, f"{note}；{extra}"
    if code in ("SH513050", "SH515880"):
        if code == "SH513050":
            trades, _, _ = bt.run_aux_fail_lo_state_machine(frame)
            doc_pos, b, s = bt.fail_lo_eod_flags(frame, trades)
        else:
            buy_c, sell_c, _ = bt.variant_masks(frame, "aux_extrema")
            trades, _, _ = sm.run_state_machine(frame, buy_c, sell_c)
            open_tr = _open_trade(trades)
            doc_pos = open_tr is not None
            b = bool(buy_c.iloc[-1])
            s = bool(sell_c.iloc[-1])
        if doc_pos and s:
            da = "SELL"
        elif (not doc_pos) and b:
            da = "BUY"
        elif doc_pos:
            da = "HOLD"
        else:
            da = "FLAT"
        return da, note
    return action, note


def classify_row(
    *,
    in_pos: bool,
    buy_cand: bool,
    sell_cand: bool,
    near_lo: bool,
    near_hi: bool,
    mid: bool,
    fig9_pos: float | None,
) -> tuple[str, str]:
    watches: list[str] = []
    if in_pos:
        action = "SELL" if sell_cand else "HOLD"
    else:
        action = "BUY" if buy_cand else "FLAT"
    if (not in_pos) and near_lo and not buy_cand:
        watches.append("WATCH_NEAR_LO")
    if in_pos and near_hi and not sell_cand:
        watches.append("WATCH_NEAR_HI")
    if (
        in_pos
        and fig9_pos is not None
        and fig9_pos >= POS_STRETCH
        and not mid
        and not sell_cand
    ):
        watches.append("WATCH_STRETCH")
    return action, ",".join(watches)


def scan_one(
    *,
    code: str,
    listing: Path,
    listing_day: str,
    as_of: pd.Timestamp,
    anchor: str,
    lookback: int,
    bt,
    sm,
) -> dict[str, Any]:
    html = find_regime_html(listing, code)
    if html is None:
        return {
            "code": code,
            "name": "",
            "action": "ERROR",
            "in_pos": False,
            "watch": "",
            "error": "missing adaptive html",
            "listing": listing_day,
        }
    name = _name_from_html(html, code)
    frame, _ohlcv = sm.build_feature_frame(html, listing, anchor, lookback=lookback)
    frame = bt.prepare_fig9_touch_frame(frame)
    frame = frame.loc[frame.index <= as_of]
    if frame.empty:
        return {
            "code": code,
            "name": name,
            "action": "ERROR",
            "in_pos": False,
            "watch": "",
            "error": f"no bars on/before {as_of.date()}",
            "listing": listing_day,
            "html": str(html),
        }
    buy_c, sell_c, _note = bt.variant_masks(frame, "aux_edge")
    trades, _bs, _ss = sm.run_state_machine(frame, buy_c, sell_c)
    open_tr = _open_trade(trades)
    in_pos = open_tr is not None
    last_i = frame.index[-1]
    row = frame.loc[last_i]
    buy_today = bool(buy_c.loc[last_i])
    sell_today = bool(sell_c.loc[last_i])
    near_lo = _bool(row.get("fig9_near_lo"))
    near_hi = _bool(row.get("fig9_near_hi"))
    mid = _bool(row.get("fig9_mid_stretch_sell"))
    aux_b = _bool(row.get("aux_buy"))
    aux_s = _bool(row.get("aux_sell"))
    pos9 = _f(row.get("fig9_pos"))
    px = _f(row.get("px"))
    lo = _f(row.get("fig9_lower"))
    hi = _f(row.get("fig9_upper"))
    dist_lo = None if px is None or lo is None or lo == 0 else px / lo - 1.0
    dist_hi = None if px is None or hi is None or hi == 0 else px / hi - 1.0
    edge_lo = bool(bt.rising_edge(frame["fig9_near_lo"]).loc[last_i])
    edge_hi = bool(bt.rising_edge(frame["fig9_near_hi"]).loc[last_i])
    action, watch = classify_row(
        in_pos=in_pos,
        buy_cand=buy_today,
        sell_cand=sell_today,
        near_lo=near_lo,
        near_hi=near_hi,
        mid=mid,
        fig9_pos=pos9,
    )
    miss: list[str] = []
    if "WATCH_NEAR_LO" in watch:
        if not edge_lo:
            miss.append("缺近下上跳沿")
        if not aux_b:
            miss.append("缺aux买")
    if "WATCH_NEAR_HI" in watch:
        if not edge_hi:
            miss.append("缺近上上跳沿")
        if not aux_s:
            miss.append("缺aux卖")
        n_hi = _streak_true(frame["fig9_near_hi"])
        if n_hi >= 2:
            miss.append(f"连续近上{n_hi}日")
    if "WATCH_STRETCH" in watch:
        if not _bool(row.get("locmax")):
            miss.append("缺局部高")
        if not aux_s:
            miss.append("缺aux卖")
        if not mid:
            miss.append("拉伸卖未齐")
    unreal = None
    last_buy = ""
    entry = ""
    if open_tr is not None:
        last_buy = str(open_tr.get("signal_buy") or "")
        entry = str(open_tr.get("entry") or "")
        unreal = _f(open_tr.get("ret"))
    elif trades is not None and not trades.empty:
        last_buy = str(trades.iloc[-1].get("signal_buy") or "")
        entry = str(trades.iloc[-1].get("entry") or "")

    doc_action, doc_note = _doc_action(
        code,
        bt=bt,
        sm=sm,
        frame=frame,
        in_pos=in_pos,
        buy_today=buy_today,
        sell_today=sell_today,
        action=action,
        row=row,
    )
    return {
        "code": code,
        "name": name,
        "listing": listing_day,
        "n": int(len(frame)),
        "as_of": last_i.strftime("%Y-%m-%d"),
        "px": None if px is None else round(px, 4),
        "fig9_pos": None if pos9 is None else round(pos9, 4),
        "dist_lo": None if dist_lo is None else round(dist_lo, 4),
        "dist_hi": None if dist_hi is None else round(dist_hi, 4),
        "aux_buy": aux_b,
        "aux_sell": aux_s,
        "near_lo": near_lo,
        "near_hi": near_hi,
        "edge_lo": edge_lo,
        "edge_hi": edge_hi,
        "mid_stretch": mid,
        "buy_cand": buy_today,
        "sell_cand": sell_today,
        "in_pos": in_pos,
        "action": action,
        "watch": watch,
        "watch_why": "；".join(miss),
        "unrealized": None if unreal is None else round(unreal, 4),
        "last_buy_signal": last_buy,
        "last_entry": entry,
        "doc_action": doc_action,
        "doc_note": doc_note,
        "html": html.name,
        "error": "",
    }


def _sort_by_fig9_pos(frame: pd.DataFrame) -> pd.DataFrame:
    """High fig9 band position first; NaN / missing last; code as tiebreak."""
    if frame.empty:
        return frame
    out = frame.copy()
    if "fig9_pos" in out.columns:
        key = pd.to_numeric(out["fig9_pos"], errors="coerce")
    else:
        key = pd.Series(float("nan"), index=out.index, dtype=float)
    out = out.assign(_fig9_sort=key)
    return out.sort_values(
        by=["_fig9_sort", "code"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_fig9_sort"])

def write_markdown(
    path: Path,
    as_of: str,
    rows: list[dict[str, Any]],
    *,
    cluster: pd.DataFrame | None = None,
    cluster_path: Path | None = None,
    fig1_markers: list[dict[str, Any]] | None = None,
) -> None:
    df = pd.DataFrame(rows)
    lines = [
        f"# 图9 aux_edge 收盘扫描（{as_of}）",
        "",
        "全池统一红利图9 `aux_edge`（近下上跳沿 ∧ 辅助买 / 近上上跳沿 ∧ 辅助卖 ∨ 带内拉伸）。",
        "**信号日收盘确认，下一交易日收盘成交**。研究扫描，不是生产 `hold_up`。",
        "",
        f"扫描 {len(df)} 只。全表按图9分位从高到低。",
        "",
        "## 全表",
        "",
        "| 代码 | 名称 | n | 动作 | 看盘 | 图9分位 | 持仓 | 未实现 | 最近买信号 | 文档动作 |",
        "|---|---|---:|---|---|---:|---|---:|---|---|",
    ]
    for _, r in _sort_by_fig9_pos(df).iterrows():
        lines.append(
            f"| {r['code']} | {_md_cell(r.get('name'))} | {r.get('n') if pd.notna(r.get('n')) else '—'} | "
            f"{_md_cell(r.get('action'))} | {_md_cell(r.get('watch'))} | {_pct(_f(r.get('fig9_pos')), 0)} | "
            f"{'是' if _truthy_in_pos(r.get('in_pos')) else ''} | {_pct(_f(r.get('unrealized')))} | "
            f"{_md_cell(r.get('last_buy_signal'))} | {_md_cell(r.get('doc_action'))} |"
        )
    lines.append("")
    lines.extend(
        format_hrp_cluster_section(rows, cluster, cluster_path=cluster_path)
    )
    lines.extend(format_fig1_markers_section(as_of, fig1_markers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    listing = _resolve(args.listing_dir)
    as_of = pd.Timestamp(args.as_of).normalize()
    listing_csv = listing / "listing_dates.csv"
    if not listing_csv.is_file():
        raise FileNotFoundError(listing_csv)
    cluster_path = (
        _resolve(args.cluster_csv)
        if args.cluster_csv is not None
        else default_cluster_csv(as_of)
    )
    cluster = load_cluster_frame(cluster_path)
    if cluster is None:
        print(f"[WARN] cluster csv missing/unreadable: {cluster_path}", flush=True)
    else:
        print(f"[INFO] cluster csv: {cluster_path} rows={len(cluster)}", flush=True)

    out_csv = listing / f"fig9_aux_edge_eod_{as_of.strftime('%Y%m%d')}.csv"
    out_md = listing / f"fig9_aux_edge_eod_{as_of.strftime('%Y%m%d')}.md"

    if args.rewrite_md_from_csv:
        if not out_csv.is_file():
            raise FileNotFoundError(out_csv)
        df = pd.read_csv(out_csv)
        rows = df.to_dict(orient="records")
        print(f"[INFO] rewrite md from {out_csv} rows={len(rows)}", flush=True)
    else:
        codes = pd.read_csv(listing_csv)
        bt = _load_mod(_SCRIPTS / "backtest_fig9_touch.py", "fig9_touch_bt")
        sm = _load_mod(_SCRIPTS / "run_reusable_extrema_state_machine.py", "extrema_sm")
        rows = []
        for rec in codes.itertuples(index=False):
            code = str(rec.code).upper()
            listing_day = str(getattr(rec, "listing", "") or "")
            print(f"[scan] {code}", flush=True)
            try:
                row = scan_one(
                    code=code,
                    listing=listing,
                    listing_day=listing_day,
                    as_of=as_of,
                    anchor=str(args.anchor_code).upper(),
                    lookback=int(args.lookback),
                    bt=bt,
                    sm=sm,
                )
            except Exception as exc:
                msg = str(exc)
                if "fig7 candlestick not found" in msg:
                    msg = "HTML 无 fig7 K 线（国债类，图9规则不适用）"
                else:
                    msg = f"{type(exc).__name__}: {exc}"
                row = {
                    "code": code,
                    "name": "",
                    "listing": listing_day,
                    "action": "ERROR",
                    "in_pos": False,
                    "watch": "",
                    "error": msg,
                }
                traceback.print_exc()
            rows.append(row)
            print(
                f"  -> {row.get('action')} watch={row.get('watch') or ''} "
                f"{row.get('error') or ''}",
                flush=True,
            )
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_csv}")

    print("[INFO] collect fig1 markers…", flush=True)
    fig1_markers = collect_listing_fig1_markers(listing, as_of)
    n_hit = sum(
        1
        for r in fig1_markers
        if r["markers"] not in ("无",) and not str(r["markers"]).startswith("解析失败")
    )
    print(f"[INFO] fig1 markers: {n_hit}/{len(fig1_markers)} with events", flush=True)
    write_markdown(
        out_md,
        as_of.strftime("%Y-%m-%d"),
        rows,
        cluster=cluster,
        cluster_path=cluster_path,
        fig1_markers=fig1_markers,
    )
    print(f"[OK] {out_md}")
    df = pd.DataFrame(rows)
    buy = df.loc[df["action"].astype(str) == "BUY", "code"].tolist()
    sell = df.loc[df["action"].astype(str) == "SELL", "code"].tolist()
    print(f"BUY ({len(buy)}): {', '.join(str(c) for c in buy) or '—'}")
    print(f"SELL ({len(sell)}): {', '.join(str(c) for c in sell) or '—'}")
    return df


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
