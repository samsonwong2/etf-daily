"""Morning receipt for one listing directory.

```
accept_eod steps 1-8
        |
        v
etf_daily.morning
        |
        +-- pool codes -------- CLUSTER_MAPPING_SELECTED_TXT
        +-- html + last bar --- listing/*.html   (all pool codes, serial)
        +-- ticket codes ------ load_universe(adaptive_dir)
        +-- trigger codes ----- pool codes where is_bond is false
        +-- go_nogo / reversal  VALIDATION_DIR
        +-- d080 -------------- HRP_OUTPUT_DIR
        |
        v
listing/morning_receipt.json
listing/morning.html
exit 1 iff any check status is fail
```

Any fail hides the short list (STOP). go_nogo and a stale reversal file are
warnings, so the normal SKIP_REBUILD day is WARN with the list still shown.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from etf_daily.lib.fair_path_band_signals import load_ohlcv_from_regime_html
from etf_daily.scripts.backtest_beat_or_hold_513650 import load_universe
from etf_daily.scripts.backtest_fig12_six_states_pool import is_bond, parse_html_label

# Same default string as scripts/daily_adaptive_from_listing.sh line 31.
DEFAULT_VALIDATION_REL = (
    "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
)
GO_NOGO_WARN = "go_nogo pass 不是 true（这是模型校准，不是文件是否齐）"
EMPTY_LIST = "今天没有同时过门槛又有触发价的名字"
STOP_LINE = "STOP 这页不是今天。下面没有短名单。"


def _project_root() -> Path:
    return next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def parse_pool_codes(text: str) -> list[str]:
    """Qlib instrument lines: first tab field, uppercased. Blank lines skipped."""
    seen: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        code = raw.split("\t", 1)[0].strip().upper()
        if code and code not in seen:
            seen.append(code)
    return seen


def parse_day(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def cell_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "nat"}:
        return True
    return False


def ok_is_true(value: Any) -> bool:
    """True only for boolean true, numeric 1, or the strings true / 1."""
    if isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return False
        return float(value) == 1.0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


def numeric_one(value: Any) -> bool:
    """pass_up_and_q05 passes only when the cell parses as the number 1."""
    if isinstance(value, bool) or cell_missing(value):
        return False
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _check(check_id: str, status: str, detail: str) -> dict[str, str]:
    return {"id": check_id, "status": status, "detail": detail}


def _html_matches(listing_dir: Path, code: str) -> list[Path]:
    return sorted(listing_dir.glob(f"regime_transition_{code}_*_adaptive.html"))


def _bond_code(code: str, matches: list[Path]) -> bool:
    name = ""
    if len(matches) == 1:
        _parsed, name = parse_html_label(matches[0])
    return is_bond(code, name)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _index_by_code(frame: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if "code" not in frame.columns:
        return out
    for _, row in frame.iterrows():
        code = str(row["code"]).strip().upper()
        if code and code not in {"NAN", "NONE"}:
            out[code] = row
    return out


def _fmt_px(value: Any) -> str:
    if cell_missing(value):
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ret(value: Any) -> str:
    if cell_missing(value):
        return "—"
    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "—"


def _trigger_cell(px: Any, ret: Any) -> str:
    price = _fmt_px(px)
    change = _fmt_ret(ret)
    if price == "—" and change == "—":
        return "—"
    return f"{price} {change}"


def build_receipt(
    *,
    as_of: date,
    next_day: date,
    pool_txt: Path,
    listing_dir: Path,
    adaptive_dir: Path,
    validation_dir: Path,
    hrp_file: Path,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    as_of_text = as_of.isoformat()

    pool_codes: list[str] = []
    if not pool_txt.is_file():
        checks.append(_check("pool", "fail", "池文件缺失"))
    else:
        pool_codes = parse_pool_codes(pool_txt.read_text(encoding="utf-8"))
        if not pool_codes:
            checks.append(_check("pool", "fail", "池是空的"))
        else:
            checks.append(_check("pool", "ok", str(len(pool_codes))))

    missing_html: list[str] = []
    html_fail_bits: list[str] = []
    single_html: dict[str, Path] = {}
    for code in pool_codes:
        matches = _html_matches(listing_dir, code)
        if len(matches) == 0:
            missing_html.append(code)
            html_fail_bits.append(f"missing {code}")
        elif len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            html_fail_bits.append(f"{code} 多文件 {names}")
        else:
            single_html[code] = matches[0]
    if html_fail_bits:
        checks.append(_check("html", "fail", "; ".join(html_fail_bits)))
    elif pool_codes:
        checks.append(_check("html", "ok", str(len(pool_codes))))
    else:
        checks.append(_check("html", "fail", "没有可核对的代码"))

    last_bar_dates: dict[str, str] = {}
    last_fail_bits: list[str] = []
    # One HTML resident at a time. The frame is dropped before the next code.
    for code, path in single_html.items():
        try:
            _close, ohlcv = load_ohlcv_from_regime_html(path)
        except (OSError, ValueError, KeyError, TypeError):
            last_fail_bits.append(f"{code} 读不出 K 线")
            continue
        if ohlcv.empty:
            last_fail_bits.append(f"{code} 读不出 K 线")
            del ohlcv
            continue
        last = parse_day(ohlcv.index.max())
        del ohlcv
        if last is None:
            last_fail_bits.append(f"{code} 读不出 K 线")
            continue
        last_bar_dates[code] = last.isoformat()
        if last != as_of:
            last_fail_bits.append(f"{code} 最后一根 {last.isoformat()}")
    if last_fail_bits:
        checks.append(_check("last_bar", "fail", "; ".join(last_fail_bits)))
    elif single_html:
        checks.append(_check("last_bar", "ok", as_of_text))
    else:
        checks.append(_check("last_bar", "fail", "无单文件可对最后一根"))

    ticket_path = adaptive_dir / "ticket_card.csv"
    meta_path = adaptive_dir / "ticket_card_meta.json"
    ticket_rows: dict[str, pd.Series] = {}
    ticket_bits: list[str] = []
    expected_ticket: list[str] = []
    try:
        expected_ticket, _names = load_universe(adaptive_dir)
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        ticket_bits.append(f"票卡代码表读失败 {exc}")
    if not ticket_path.is_file():
        ticket_bits.append("ticket_card.csv 缺失")
    else:
        ticket_frame = _read_csv(ticket_path)
        if "as_of" not in ticket_frame.columns or "code" not in ticket_frame.columns:
            ticket_bits.append("ticket_card.csv 缺 as_of 或 code")
        else:
            for _, row in ticket_frame.iterrows():
                row_day = parse_day(row["as_of"])
                if row_day != as_of:
                    shown = "缺失" if row_day is None else row_day.isoformat()
                    ticket_bits.append(f"行 as_of {shown}")
                    break
            ticket_rows = _index_by_code(ticket_frame)
    if not meta_path.is_file():
        ticket_bits.append("ticket_card_meta.json 缺失")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
            ticket_bits.append("ticket_card_meta.json 不是 JSON")
        meta_day = parse_day(meta.get("as_of"))
        if meta_day != as_of:
            shown = "缺失" if meta_day is None else meta_day.isoformat()
            ticket_bits.append(f"meta as_of {shown}")
    missing_ticket = [code for code in expected_ticket if code not in ticket_rows]
    if missing_ticket:
        ticket_bits.append("缺票卡行 " + ", ".join(missing_ticket))
    if ticket_bits:
        checks.append(_check("ticket_card", "fail", "; ".join(ticket_bits)))
    else:
        checks.append(_check("ticket_card", "ok", "as_of matches"))

    trigger_name = f"next_day_triggers_{next_day.strftime('%Y%m%d')}.csv"
    trigger_path = listing_dir / trigger_name
    trigger_rows: dict[str, pd.Series] = {}
    required_triggers = [
        code for code in pool_codes if not _bond_code(code, _html_matches(listing_dir, code))
    ]
    missing_triggers: list[str] = []
    trigger_errors: list[dict[str, str]] = []
    trigger_bits: list[str] = []
    if not trigger_path.is_file():
        trigger_bits.append(f"{trigger_name} 缺失")
        missing_triggers = list(required_triggers)
    else:
        trigger_frame = _read_csv(trigger_path)
        trigger_rows = _index_by_code(trigger_frame)
        for code in required_triggers:
            row = trigger_rows.get(code)
            if row is None:
                missing_triggers.append(code)
                continue
            if not ok_is_true(row.get("ok")):
                error = "" if cell_missing(row.get("error")) else str(row.get("error"))
                trigger_errors.append({"code": code, "error": error})
    if missing_triggers:
        trigger_bits.append("缺触发价行 " + ", ".join(missing_triggers))
    for item in trigger_errors:
        trigger_bits.append(f"{item['code']} ok 不是 true {item['error']}".rstrip())
    if trigger_bits:
        checks.append(_check("triggers", "fail", "; ".join(trigger_bits)))
    else:
        checks.append(_check("triggers", "ok", f"{len(required_triggers)} rows"))

    if hrp_file.is_file():
        checks.append(_check("hrp_d080", "ok", "present"))
    else:
        checks.append(_check("hrp_d080", "fail", "d080 缺失"))

    go_path = validation_dir / "go_nogo.json"
    go_pass = False
    if go_path.is_file():
        try:
            payload = json.loads(go_path.read_text(encoding="utf-8"))
            go_pass = payload.get("pass") is True
        except json.JSONDecodeError:
            go_pass = False
    if go_pass:
        checks.append(_check("go_nogo", "ok", "pass=true"))
    else:
        checks.append(_check("go_nogo", "warn", GO_NOGO_WARN))

    reversal_path = validation_dir / "signals_with_reversal_labels.csv"
    reversal_day: date | None = None
    if reversal_path.is_file():
        reversal = _read_csv(reversal_path)
        if "as_of" in reversal.columns:
            parsed = [parse_day(value) for value in reversal["as_of"]]
            days = [item for item in parsed if item is not None]
            if days:
                reversal_day = max(days)
    if reversal_day is not None and reversal_day >= as_of:
        checks.append(_check("reversal", "ok", reversal_day.isoformat()))
    else:
        shown = "缺失" if reversal_day is None else reversal_day.isoformat()
        detail = f"反转标签最大日期 {shown}，早于 {as_of_text}。悬停里的验证不是今天"
        checks.append(_check("reversal", "warn", detail))

    failed = any(item["status"] == "fail" for item in checks)
    warned = any(item["status"] == "warn" for item in checks)
    if failed:
        label = "STOP"
    elif warned:
        label = "WARN"
    else:
        label = "TODAY"

    rows: list[dict[str, str]] = []
    if label != "STOP":
        for code, ticket in ticket_rows.items():
            trigger = trigger_rows.get(code)
            if trigger is None or not numeric_one(ticket.get("pass_up_and_q05")):
                continue
            if cell_missing(trigger.get("buy_px")) and cell_missing(trigger.get("sell_px")):
                continue
            name = trigger.get("name")
            if cell_missing(name):
                name = ticket.get("name")
            name_text = "" if cell_missing(name) else str(name)
            regime = "" if cell_missing(ticket.get("regime")) else str(ticket.get("regime"))
            link = ""
            matches = _html_matches(listing_dir, code)
            if len(matches) == 1:
                link = matches[0].name
            rows.append(
                {
                    "code": code,
                    "name": name_text,
                    "regime": regime,
                    "px": _fmt_px(trigger.get("px")),
                    "buy": _trigger_cell(trigger.get("buy_px"), trigger.get("buy_ret")),
                    "sell": _trigger_cell(trigger.get("sell_px"), trigger.get("sell_ret")),
                    "html": link,
                }
            )
        rows.sort(key=lambda item: item["code"])

    try:
        hrp_href = os.path.relpath(hrp_file, listing_dir)
    except ValueError:
        hrp_href = str(hrp_file)

    receipt: dict[str, Any] = {
        "as_of": as_of_text,
        "next_day": next_day.isoformat(),
        "label": label,
        "checks": checks,
        "expected_codes": pool_codes,
        "missing_html": missing_html,
        "last_bar_dates": last_bar_dates,
        "rows": rows,
        "trigger_csv": trigger_name,
        "hrp_href": hrp_href,
    }
    if missing_triggers:
        receipt["missing_triggers"] = missing_triggers
    if trigger_errors:
        receipt["trigger_errors"] = trigger_errors
    return receipt


def render_html(receipt: dict[str, Any]) -> str:
    label = str(receipt["label"])
    checks = receipt["checks"]
    if label == "STOP":
        status = STOP_LINE
    elif label == "TODAY":
        status = f"文件齐，最后一根是 {receipt['as_of']}，go_nogo pass 为 true"
    else:
        warn_bits = [item["detail"] for item in checks if item["status"] == "warn"]
        status = "；".join(warn_bits)

    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{html.escape(label)} {html.escape(str(receipt['as_of']))}</title>",
        "<style>",
        "body{font:16px/1.45 sans-serif;margin:24px;color:#1a1a1a}",
        "h1{font-size:20px;margin:0 0 12px}",
        ".stop{color:#8a1c1c}.warn{color:#8a5a00}",
        "table{border-collapse:collapse;margin-top:16px}",
        "td,th{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left}",
        "footer{margin-top:24px;font-size:14px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1 class=\"{html.escape(label.lower())}\">{html.escape(label)}</h1>",
        f"<p>{html.escape(status)}</p>",
    ]
    if label == "STOP":
        parts.append("<ul>")
        for item in checks:
            if item["status"] in {"fail", "warn"}:
                parts.append(f"<li>{html.escape(item['detail'])}</li>")
        parts.append("</ul>")
    else:
        rows = receipt.get("rows") or []
        if not rows:
            parts.append(f"<p>{html.escape(EMPTY_LIST)}</p>")
        else:
            parts.append("<table>")
            parts.append(
                "<tr><th>代码</th><th>名称</th><th>阶段</th><th>收盘</th>"
                "<th>买触发</th><th>卖触发</th><th>打开 K 线</th></tr>"
            )
            for row in rows:
                href = html.escape(row["html"], quote=True)
                link = f"<a href=\"{href}\">打开 K 线</a>" if href else "—"
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(row['code'])}</td>"
                    f"<td>{html.escape(row['name'])}</td>"
                    f"<td>{html.escape(row['regime'])}</td>"
                    f"<td>{html.escape(row['px'])}</td>"
                    f"<td>{html.escape(row['buy'])}</td>"
                    f"<td>{html.escape(row['sell'])}</td>"
                    f"<td>{link}</td>"
                    "</tr>"
                )
            parts.append("</table>")
        parts.append("<footer>")
        parts.append(f"<div>{html.escape(str(receipt['trigger_csv']))}</div>")
        parts.append(f"<div>{html.escape(str(receipt['hrp_href']))}</div>")
        parts.append("</footer>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _default_pool() -> Path:
    from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT

    return CLUSTER_MAPPING_SELECTED_TXT


def _default_plotly() -> Path:
    env = os.environ.get("PLOTLY_ROOT")
    if env:
        return Path(env).expanduser()
    from etf_daily.paths import PLOTLY_OUTPUTS_DIR

    return PLOTLY_OUTPUTS_DIR


def _default_hrp_root() -> Path:
    env = os.environ.get("HRP_OUTPUT_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "etf-daily-output" / "decision_packs"


def _default_validation() -> Path:
    env = os.environ.get("VALIDATION_DIR")
    if env:
        return Path(env).expanduser()
    return _project_root() / DEFAULT_VALIDATION_REL


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="etf-daily morning")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--next-day", required=True)
    parser.add_argument("--listing-dir", default=None)
    parser.add_argument("--adaptive-dir", default=None)
    parser.add_argument("--pool-txt", default=None)
    parser.add_argument("--validation-dir", default=None)
    parser.add_argument("--hrp-file", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = parse_day(args.as_of)
    next_day = parse_day(args.next_day)
    if as_of is None or next_day is None:
        print("pass --as-of and --next-day as YYYY-MM-DD", file=sys.stderr)
        return 2
    tag = as_of.strftime("%Y%m%d")
    plotly = _default_plotly()
    listing = Path(args.listing_dir).expanduser() if args.listing_dir else plotly / f"{tag}_from_listing"
    adaptive = (
        Path(args.adaptive_dir).expanduser()
        if args.adaptive_dir
        else plotly / f"{tag}all_adaptive"
    )
    pool_txt = Path(args.pool_txt).expanduser() if args.pool_txt else _default_pool()
    validation = (
        Path(args.validation_dir).expanduser() if args.validation_dir else _default_validation()
    )
    hrp = (
        Path(args.hrp_file).expanduser()
        if args.hrp_file
        else _default_hrp_root() / tag / f"hrp_dendrogram_{tag}_d080.html"
    )
    receipt = build_receipt(
        as_of=as_of,
        next_day=next_day,
        pool_txt=pool_txt,
        listing_dir=listing,
        adaptive_dir=adaptive,
        validation_dir=validation,
        hrp_file=hrp,
    )
    page = render_html(receipt)
    public = {key: value for key, value in receipt.items() if key not in {"rows", "trigger_csv", "hrp_href"}}
    atomic_write(listing / "morning_receipt.json", json.dumps(public, ensure_ascii=False, indent=2) + "\n")
    atomic_write(listing / "morning.html", page)
    print(f"morning {receipt['label']} {listing / 'morning.html'}")
    return 1 if receipt["label"] == "STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
