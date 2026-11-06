"""Replay and daily trade-report checks. No qlib and no network."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from etf_daily.lib.rules7_trade_plan import (
    breadth_stats,
    breadth_tier,
    d080_cluster_csv,
    signal_trades,
    tier_history,
)
from etf_daily.scripts.rules7_trade_report import report_path, update_ledger, write_trade_report
from etf_daily.scripts.scan_rules7_checklist import parse_args

SH = ZoneInfo("Asia/Shanghai")


def _px(n: int, price: float = 1.0) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.Series(price, index=idx, dtype=float)


def _buy(px: pd.Series, *positions: int) -> pd.Series:
    sig = pd.Series(False, index=px.index)
    for pos in positions:
        sig.iloc[pos] = True
    return sig


def test_signal_is_pending_until_the_next_close():
    px = _px(3)
    pending = signal_trades(px.iloc[:1], _buy(px, 0).iloc[:1])
    assert pending[0]["status"] == "pending"
    assert pending[0]["entry_date"] is None

    held = signal_trades(px.iloc[:2], _buy(px, 0).iloc[:2])[0]
    assert held["status"] == "open"
    assert held["entry_date"] == px.index[1].strftime("%Y-%m-%d")
    assert held["days_held"] == 0
    assert held["sell_tomorrow"] is False


def test_signal_inside_an_open_trade_is_kept():
    px = _px(8)
    trades = signal_trades(px, _buy(px, 0, 3))
    assert len(trades) == 2
    assert all(trade["status"] == "open" for trade in trades)
    assert trades[1]["entry_date"] == px.index[4].strftime("%Y-%m-%d")


def test_signal_expiry_sells_tomorrow_on_day_19():
    px = _px(21)
    trade = signal_trades(px, _buy(px, 0))[0]
    assert trade["status"] == "open"
    assert trade["sell_tomorrow"] is True
    assert trade["reason"] == "expiry"
    assert trade["days_held"] == 19
    assert trade["entry_date"] == px.index[1].strftime("%Y-%m-%d")


def test_signal_stop_sells_the_next_close():
    px = _px(6)
    px.iloc[3] = 0.89
    early = signal_trades(px.iloc[:4], _buy(px, 0).iloc[:4])[0]
    assert early["status"] == "open"
    assert early["sell_tomorrow"] is True
    assert early["reason"] == "stop"

    stopped = signal_trades(px.iloc[:5], _buy(px, 0).iloc[:5])[0]
    assert stopped["status"] == "closed"
    assert stopped["reason"] == "stop"
    assert stopped["exit_px"] == px.iloc[4]
    assert stopped["exit_date"] == px.index[4].strftime("%Y-%m-%d")

    both = signal_trades(px, _buy(px, 0, 4))
    assert len(both) == 2
    assert both[0]["status"] == "closed"
    assert both[1]["status"] == "open"
    assert both[1]["entry_date"] == px.index[5].strftime("%Y-%m-%d")


def test_report_lists_five_signal_states(tmp_path: Path):
    frame = _frame().iloc[:1].copy()
    frame.loc[0, "_buy_raw"] = False
    frame["_sig_trades"] = pd.Series([[
        {"status": "pending", "signal_date": "2026-11-05", "entry_date": None, "entry_px": None,
         "days_held": None, "pnl": None, "stop_px": None, "dist_stop": None,
         "reason": None, "sell_tomorrow": False, "bars_from_end": None},
        {"status": "open", "signal_date": "2026-10-09", "entry_date": "2026-10-10", "entry_px": 1.0,
         "days_held": 19, "pnl": 0.01, "stop_px": 0.9, "dist_stop": 0.12,
         "reason": "expiry", "sell_tomorrow": True, "bars_from_end": 19},
        {"status": "open", "signal_date": "2026-10-16", "entry_date": "2026-10-17", "entry_px": 1.0,
         "days_held": 8, "pnl": -0.02, "stop_px": 0.9, "dist_stop": 0.08,
         "reason": "stop", "sell_tomorrow": True, "bars_from_end": 8},
        {"status": "open", "signal_date": "2026-10-20", "entry_date": "2026-10-21", "entry_px": 1.0,
         "days_held": 10, "days_left": 10, "pnl": 0.03, "stop_px": 0.9, "dist_stop": 0.14,
         "reason": None, "sell_tomorrow": False, "bars_from_end": 10},
        {"status": "closed", "signal_date": "2026-10-22", "entry_date": "2026-10-23", "entry_px": 1.0,
         "exit_date": "2026-10-28", "days_held": 4, "pnl": -0.1, "stop_px": 0.9, "dist_stop": -0.01,
         "reason": "stop", "sell_tomorrow": False, "bars_from_end": 8},
    ]], dtype=object)
    text = _report(tmp_path, frame, _cluster().iloc[:1])
    recent = text.split("## 最近")[1].split("## P90")[0]
    for label in ("待成交", "明日到期卖", "明日止损卖", "持有中，剩 10 天", "已于 2026-10-28 止损"):
        assert label in recent


def test_tier_history_uses_the_pool_that_day():
    day = "2026-06-01"
    rows = []
    for i in range(21):
        trades = [{"signal_date": day, "status": "closed", "ret": -0.02}] if i == 0 else []
        rows.append({
            "applicable": True,
            "bar250_date": "2020-01-01",
            "last_date": day,
            "trades": trades,
        })
    stats = tier_history(rows)
    assert stats["单独"][0] == 1
    assert stats["集中"][0] == 0
    assert stats["一般"][0] == 0


def test_d080_cluster_csv_falls_back_to_previous_day(tmp_path: Path):
    day = tmp_path / "20261104"
    day.mkdir()
    (day / "cluster_representatives_20261104_d080.csv").write_text("code\nSH510300\n", encoding="utf-8")
    (tmp_path / "20261105").mkdir()

    path, used = d080_cluster_csv("2026-11-05", roots=[tmp_path])
    assert path is not None and path.parent.name == "20261104"
    assert used == pd.Timestamp("2026-11-04")

    exact = tmp_path / "20261105" / "cluster_representatives_20261105_d080.csv"
    exact.write_text("code\nSH510300\n", encoding="utf-8")
    path, used = d080_cluster_csv("2026-11-05", roots=[tmp_path])
    assert path == exact
    assert used == pd.Timestamp("2026-11-05")


def _frame() -> pd.DataFrame:
    recent = tuple(f"2026-09-{day:02d}" for day in range(1, 11))
    rows = []
    for code, state, dist, rep in (
        ("SH510300", "entry_tomorrow", 0.02, True),
        ("SH510500", "entry_tomorrow", 0.05, False),
        ("SZ159915", "flat", 0.20, False),
    ):
        rows.append(
            {
                "代码": code,
                "名称": code,
                "T收盘": "1.5000",
                "今日状态": "空仓",
                "EWMA买点状态": "",
                "买点价": "1.4000",
                "_plan_state": state,
                "_plan_sell_reason": None,
                "_plan_applicable": True,
                "_plan_entry_date": None,
                "_plan_entry_px": None,
                "_plan_days_held": None,
                "_plan_pnl": None,
                "_plan_stop_px": None,
                "_plan_dist_stop": None,
                "_plan_days_left": None,
                "_plan_closed": [],
                "_plan_recent_dates": recent,
                "_dist_lo9": dist,
                "_dist_ewma": dist,
                "_t_fast": False,
                "_buy_raw": state == "entry_tomorrow",
            }
        )
    return pd.DataFrame(rows)


def _cluster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["SH510300", "SH510500", "SZ159915"],
            "cluster": [1, 1, 2],
            "is_representative": [True, False, False],
        }
    )


def test_report_keeps_one_buy_per_cluster_and_names_every_code(tmp_path: Path):
    as_of = pd.Timestamp("2026-11-05")
    path = write_trade_report(
        tmp_path,
        _frame(),
        as_of=as_of,
        now=datetime(2026, 11, 5, 16, 0, tzinfo=SH),
        cluster=_cluster(),
        cluster_path=tmp_path / "cluster.csv",
        cluster_date=as_of,
    )
    text = path.read_text(encoding="utf-8")
    assert path.name == "rules7_trade_report_20261105.md"
    buy_section = text.split("## 最近")[0]
    assert "推荐持有" not in text
    assert "推荐卖出" not in text
    assert "同簇已持有" not in text
    assert "SH510300" in buy_section
    assert "SH510500" not in buy_section
    assert "同簇今天已推荐 SH510300" in text
    for code in ("SH510300", "SH510500", "SZ159915"):
        assert text.count(code) >= 1
    assert "离下轨较远，今天没有 P95 正式买点" in text


def test_report_filename_uses_hhmm_intraday(tmp_path: Path):
    intraday = report_path(tmp_path, "2026-11-05", datetime(2026, 11, 5, 10, 30, tzinfo=SH))
    closed = report_path(tmp_path, "2026-11-05", datetime(2026, 11, 5, 15, 0, tzinfo=SH))
    other_day = report_path(tmp_path, "2026-11-05", datetime(2026, 11, 6, 10, 30, tzinfo=SH))
    assert intraday.name == "rules7_trade_report_20261105_1030.md"
    assert closed.name == "rules7_trade_report_20261105.md"
    assert other_day.name == "rules7_trade_report_20261105.md"

    written = write_trade_report(
        tmp_path,
        _frame(),
        as_of="2026-11-05",
        now=datetime(2026, 11, 5, 10, 30, tzinfo=SH),
        cluster=_cluster(),
        cluster_path=tmp_path / "cluster.csv",
        cluster_date=pd.Timestamp("2026-11-05"),
    )
    assert written.name == "rules7_trade_report_20261105_1030.md"
    assert "最后一根 K 线未收盘" in written.read_text(encoding="utf-8")


def test_report_scenario_section_counts_and_threshold(tmp_path: Path):
    frame = _frame()
    frame.loc[0, "_plan_state"] = "flat"
    frame["_sc"] = [
        {"block": "快形态挡住", "buy_ranges": [], "p90_ranges": [], "sell_kind": None,
         "sell_move": None, "hold_lo": None, "hold_hi": None},
        None,
        {"block": None, "buy_ranges": [(-0.1, -0.03)], "p90_ranges": [(-0.1, -0.02)],
         "sell_kind": None, "sell_move": None, "hold_lo": None, "hold_hi": None},
    ]
    text = write_trade_report(
        tmp_path, frame, as_of=pd.Timestamp("2026-11-05"),
        now=datetime(2026, 11, 5, 16, 0, tzinfo=SH),
        cluster=_cluster(), cluster_path=tmp_path / "cluster.csv",
        cluster_date=pd.Timestamp("2026-11-05"),
    ).read_text(encoding="utf-8")
    section = text.split("## 明日情景")[1].split("## 观望")[0]
    assert "| ±2% | 0 只 |" in section
    assert "| ±5% | 1 只 |" in section
    assert "SZ159915" in section
    assert "跌到 -3.0% 或更低" in section
    assert "跌到 -2.0% 或更低" in section  # P90 triggers on a smaller dip
    assert "快形态挡住" in section
    assert "后天收盘才成交" in section


def test_report_blocked_table_matches_checklist_wording(tmp_path: Path):
    frame = _frame()
    frame["买点状态"] = ["快形态中，未到买点", "快形态挡住", "还差这个价"]
    block = {"block": "快形态挡住", "buy_ranges": [], "p90_ranges": [], "sell_kind": None,
             "sell_move": None, "hold_lo": None, "hold_hi": None}
    frame["_sc"] = [dict(block), dict(block), None]
    text = write_trade_report(
        tmp_path, frame, as_of=pd.Timestamp("2026-11-05"),
        now=datetime(2026, 11, 5, 16, 0, tzinfo=SH),
        cluster=_cluster(), cluster_path=tmp_path / "cluster.csv",
        cluster_date=pd.Timestamp("2026-11-05"),
    ).read_text(encoding="utf-8")
    table = text.split("### 这几种情况不给触发价")[1].split("## 观望")[0]
    assert "| SH510300 | SH510300 | 1 | 快形态中，未到买点 |" in table
    assert "| SH510500 | SH510500 | 1 | 快形态挡住 |" in table


def test_report_lists_p90_watch_without_taking_buy_slot(tmp_path: Path):
    frame = _frame()
    frame["_w_buy_raw"] = [True, True, True]
    frame["_w_lower"] = 1.4
    frame["_w_dist"] = 0.07
    frame["_w_trades"] = pd.Series(
        [[], [], [{"signal_date": "2026-03-02", "status": "closed", "ret": 0.05}]],
        dtype=object,
    )
    as_of = pd.Timestamp("2026-11-05")
    text = write_trade_report(
        tmp_path,
        frame,
        as_of=as_of,
        now=datetime(2026, 11, 5, 16, 0, tzinfo=SH),
        cluster=_cluster(),
        cluster_path=tmp_path / "cluster.csv",
        cluster_date=as_of,
    ).read_text(encoding="utf-8")
    watch_section = text.split("## P90 观察档")[1].split("## 单独出现")[0]
    assert "SH510300" not in watch_section
    assert "SH510500" not in watch_section
    assert "SZ159915" in watch_section
    assert "同簇已有买入" not in watch_section
    assert "SZ159915" not in text.split("## 最近")[0]
    assert "1 笔，胜率 100%" in watch_section


def test_breadth_tier_cutoffs_and_pool():
    assert breadth_tier(0, 0.5) == "无买点"
    assert breadth_tier(3, float("nan")) == "无买点"
    assert breadth_tier(4, 0.08) == "集中"
    assert breadth_tier(4, 0.0799) == "一般"
    assert breadth_tier(2, 0.05) == "一般"
    assert breadth_tier(1, 0.0499) == "单独"

    rows = []
    for i in range(17):
        rows.append({
            "_plan_applicable": i != 16,
            "_n_bars": 100 if i == 15 else 250,
            "_buy_raw": i in {0, 15, 16},
            "买入信号": "否",
        })
    stats = breadth_stats(pd.DataFrame(rows))
    # i=15 is too short, i=16 is not applicable, so only i=0 counts. Pool is i=0..14.
    assert stats["n_buy"] == 1
    assert stats["pool"] == 15
    assert stats["tier"] == "一般"
    assert stats["degraded"] is False

    missing = breadth_stats(pd.DataFrame([{"_plan_applicable": True, "买入信号": "是"}]))
    assert missing["degraded"] is True
    assert missing["n_buy"] == 1
    assert missing["tier"] == "集中"


def _breadth_frame(n: int, raw_buys: set[int], *, state_buys: set[int] | None = None) -> pd.DataFrame:
    """n applicable names. raw_buys feed the share; state_buys are entry_tomorrow."""
    if state_buys is None:
        state_buys = set(raw_buys)
    recent = tuple(f"2026-09-{day:02d}" for day in range(1, 11))
    rows = []
    for i in range(n):
        code = f"SH{510000 + i}"
        rows.append({
            "代码": code,
            "名称": code,
            "T收盘": "1.5000",
            "今日状态": "空仓",
            "EWMA买点状态": "",
            "买点价": "1.4000",
            "_plan_state": "entry_tomorrow" if i in state_buys else "flat",
            "_plan_sell_reason": None,
            "_plan_applicable": True,
            "_plan_entry_date": None,
            "_plan_entry_px": None,
            "_plan_days_held": None,
            "_plan_pnl": None,
            "_plan_stop_px": None,
            "_plan_dist_stop": None,
            "_plan_days_left": None,
            "_plan_closed": [],
            "_plan_recent_dates": recent,
            "_dist_lo9": 0.01,
            "_dist_ewma": 0.01,
            "_t_fast": False,
            "_buy_raw": i in raw_buys,
            "_n_bars": 250,
            "_w_state": "flat",
            "_w_lower": 1.4,
            "_w_dist": 0.02,
            "_w_closed": [],
        })
    return pd.DataFrame(rows)


def _breadth_cluster(n: int, same: set[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "code": [f"SH{510000 + i}" for i in range(n)],
        "cluster": [1 if i in same else 2 for i in range(n)],
        "is_representative": [i == min(same) for i in range(n)],
    })


def _report(tmp_path: Path, frame: pd.DataFrame, cluster: pd.DataFrame, **kwargs) -> str:
    as_of = pd.Timestamp("2026-11-05")
    path = write_trade_report(
        tmp_path, frame, as_of=as_of, now=datetime(2026, 11, 5, 16, 0, tzinfo=SH),
        cluster=cluster, cluster_path=tmp_path / "cluster.csv", cluster_date=as_of, **kwargs,
    )
    return path.read_text(encoding="utf-8")


def _buy_lines(text: str) -> list[str]:
    section = text.split("## 今日 P95 买点")[1].split("## 最近")[0]
    return [line for line in section.splitlines() if line.startswith("| SH")]


def test_solo_breadth_demotes_without_taking_cluster_slot(tmp_path: Path):
    # 2 / 50 = 4% < 5%. Both same-cluster buys drop to 观察 and neither blocks the other.
    frame = _breadth_frame(50, {0, 1})
    frame.loc[2, "_w_buy_raw"] = True
    text = _report(tmp_path, frame, _breadth_cluster(50, {0, 1, 2}))
    assert "属于单独" in text
    assert "今日全池无买点" not in text
    buy = text.split("## 最近")[0]
    assert "SH510000" not in buy
    assert "SH510001" not in buy
    solo = text.split("## 单独出现")[1].split("## 明日情景")[0]
    assert "SH510000" in solo and "SH510001" in solo
    assert "同簇今天已推荐" not in solo
    assert "低于 5%" in solo
    p90 = text.split("## P90 观察档")[1].split("## 单独出现")[0]
    assert "SH510002" in p90
    assert "同簇已有买入" not in p90


def test_strong_breadth_keeps_one_buy_per_cluster(tmp_path: Path):
    # 5 / 50 = 10% >= 8%. Cluster still keeps one name.
    frame = _breadth_frame(50, {0, 1, 2, 3, 4})
    text = _report(tmp_path, frame, _breadth_cluster(50, {0, 1}))
    assert "属于集中" in text
    buy = text.split("## 最近")[0]
    assert "SH510000" in buy
    assert "SH510001" not in buy
    assert "属于集中" in buy
    assert "同簇今天已推荐 SH510000" in text
    assert "## 单独出现" in text
    assert "SH510000" not in text.split("## 单独出现")[1].split("## 观望")[0]


def test_mid_breadth_stays_a_buy_with_a_warning(tmp_path: Path):
    # 3 / 50 = 6%, the middle bucket. Still recommended, labeled 一般.
    frame = _breadth_frame(50, {0, 1, 2})
    text = _report(tmp_path, frame, _breadth_cluster(50, {0}))
    assert "属于一般" in text
    assert "SH510000" in text.split("## 最近")[0]
    assert "不宜加仓" in text
    assert "SH510000" not in text.split("## 单独出现")[1]


def test_breadth_filter_off_keeps_the_old_buy_rows(tmp_path: Path):
    frame = _breadth_frame(50, {0, 1})
    cluster = _breadth_cluster(50, {0, 1})
    off = _report(tmp_path, frame, cluster, breadth_filter=False)
    assert "集中度" not in off
    assert "## 单独出现" not in off
    assert "SH510000" in off.split("## 最近")[0]
    assert "SH510001" not in off.split("## 最近")[0]
    assert "同簇今天已推荐 SH510000" in off

    original = _frame()
    on = _report(tmp_path, original, _cluster())
    off_small = _report(tmp_path, original, _cluster(), breadth_filter=False)
    assert [line.split("|")[1].strip() for line in _buy_lines(on)] == [
        line.split("|")[1].strip() for line in _buy_lines(off_small)
    ]
    assert "缺少回放列" in on
    assert "属于集中" in on
    assert "属于集中" not in off_small


def test_no_buy_day_says_breadth_does_not_apply(tmp_path: Path):
    frame = _breadth_frame(8, set())
    text = _report(tmp_path, frame, _breadth_cluster(8, {0}))
    assert "今日全池无买点，集中度不适用。" in text
    assert "缺少回放列" not in text


def test_ledger_is_idempotent_and_backfills_without_overwrite(tmp_path: Path):
    frame = _breadth_frame(50, {0})
    ledger = tmp_path / "rules7_signal_ledger.csv"
    _report(tmp_path, frame, _breadth_cluster(50, {0}), ledger_path=ledger)
    _report(tmp_path, frame, _breadth_cluster(50, {0}), ledger_path=ledger)
    saved = pd.read_csv(ledger)
    assert len(saved) == 1
    assert saved.loc[0, "code"] == "SH510000"
    assert bool(saved.loc[0, "filtered"]) is True
    assert saved.loc[0, "action"] == "观察"

    dates = [f"2026-08-{day:02d}" for day in range(1, 26)]
    prices = [1.0] * 25
    prices[5], prices[10], prices[20] = 1.2, 1.3, 1.4
    old = pd.DataFrame([{
        "date": dates[0], "code": "SH510099", "name": "old",
        "n_buy": 1, "pool": 50, "share": 0.02, "tier": "单独",
        "action": "观察", "filtered": True, "entry_close": 1.0,
        "ret_5": 9.0, "ret_10": "", "ret_20": "",
    }])
    old.to_csv(ledger, index=False, encoding="utf-8-sig")
    quiet = _breadth_frame(1, set())
    quiet.loc[0, "代码"] = "SH510099"
    quiet.loc[0, "名称"] = "old"
    quiet["_px_tail"] = pd.Series([list(zip(dates, prices))], dtype=object)
    update_ledger(ledger, quiet, "2026-11-05", breadth_stats(quiet))
    filled = pd.read_csv(ledger)
    row = filled[filled["code"] == "SH510099"].iloc[0]
    assert float(row["ret_5"]) == 9.0
    assert abs(float(row["ret_10"]) - 0.3) < 1e-9
    assert abs(float(row["ret_20"]) - 0.4) < 1e-9
    update_ledger(ledger, quiet, "2026-11-05", breadth_stats(quiet))
    again = pd.read_csv(ledger)
    assert abs(float(again.loc[again["code"] == "SH510099", "ret_10"].iloc[0]) - 0.3) < 1e-9


def test_no_ledger_writes_no_file(tmp_path: Path):
    frame = _breadth_frame(50, {0})
    _report(tmp_path, frame, _breadth_cluster(50, {0}))
    assert not (tmp_path / "rules7_signal_ledger.csv").exists()
    args = parse_args([
        "--listing-dir", str(tmp_path), "--as-of", "2026-11-05",
        "--no-ledger", "--breadth-filter", "off",
    ])
    assert args.no_ledger is True
    assert args.breadth_filter == "off"
    assert args.ledger is None
