"""Regression tests for triangle OPS resolve + markdown."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_daily.regime.scan_triangle_decision_ops import (
    build_ops_markdown,
    collect_candidates,
    pick_latest_deep_red_for_reclaim_watch,
    pick_latest_deep_red_reclaimed,
    plan_sell_continuation_reclaim_buys,
)
from etf_daily.lib.triangle_ops_resolve import (
    Candidate,
    candidates_from_history_row,
    candidates_from_today_triangle,
    main_table,
    resolve_main_per_code,
)

PACK = Path(
    "runtime/decision_packs/intraday/20260728/103630"
)
SCAN = PACK / "triangle_decision_scan_20260728"


def test_sh515050_deep_red_not_buy_watch_and_not_green():
    """Deep red prior10 with rv5<rv20 → 红三角不抄底; never green section wording."""
    c = candidates_from_today_triangle(
        {
            "code": "SH515050",
            "name": "通信ETF华夏",
            "side": "红",
            "action": "观察",
            "signal_date": "2026-07-28",
            "signal_px": 0.982,
            "last_px": 0.982,
            "prior10": -0.204858,
            "rv_expand": False,
            "vol_pct": 0.75,
            "pred_cdf": 0.12,
        },
        trade_s="2026-07-28",
        intraday=True,
    )
    assert c is not None
    assert c.decision == "红三角不抄底"
    assert c.section == "pending_confirm"
    assert "不抄底" in c.operation
    assert "近10日累计" in c.operation
    assert "大跌" not in c.operation
    assert "绿" not in c.operation

    hist = pd.DataFrame(
        [
            {
                "code": "SH515050",
                "name": "通信ETF华夏",
                "decision": "买观察",
                "status": "待收复",
                "signal_date": "2026-07-22",
                "side": "红",
                "signal_px": 1.061,
                "signal_high": 1.122,
                "last_px": 0.982,
                "entry_state": "forward",
                "operation": "等收复",
            },
            {
                "code": "SH515050",
                "name": "通信ETF华夏",
                "decision": "顶部交替减仓",
                "status": "顶部交替区",
                "signal_date": "2026-07-02~2026-07-10",
                "side": "RGR",
                "signal_px": 1.282,
                "last_px": 0.982,
                "entry_state": "forward",
                "operation": "旧区",
            },
        ]
    )
    today = pd.DataFrame(
        [
            {
                "code": "SH515050",
                "name": "通信ETF华夏",
                "side": "红",
                "action": "观察",
                "signal_date": "2026-07-28",
                "signal_px": 0.982,
                "last_px": 0.982,
                "prior10": -0.204858,
                "rv_expand": False,
                "vol_pct": 0.75,
                "pred_cdf": 0.12,
            }
        ]
    )
    resolved = collect_candidates(
        today=today,
        dec=hist,
        trade_s="2026-07-28",
        asof_s="2026-07-28",
        intraday=True,
    )
    main = main_table(resolved)
    assert len(main) == 1
    assert main.iloc[0]["decision"] == "红三角不抄底"
    buy = [c for c in resolved if c.decision == "买观察"]
    assert buy and all(c.suppressed_by for c in buy)

    md = build_ops_markdown(
        as_of=pd.Timestamp("2026-07-28"),
        trade_date=pd.Timestamp("2026-07-28"),
        html_dir=Path("html"),
        val=Path("val"),
        n_codes=1,
        live_captured="10:36:30",
        main_df=main,
        all_cand=pd.DataFrame([{**c.__dict__} for c in resolved]),
        today=today,
        intraday=True,
    )
    assert "## 3. 收复观察位" in md
    assert "## 4. 趋势下跌中" in md
    assert "新绿三角" not in md
    assert "SH515050" in md
    assert "不抄底" in md or "不进买入" in md
    assert "快照现价" in md
    assert "禁止当作收盘价下单" in md
    # Must not put SH515050 into a green-triangle chapter
    assert "新绿三角" not in md
    assert "未达卖预警" not in md.split("## 附录")[0]


def test_today_bearish_covers_old_buy_watch():
    resolved = resolve_main_per_code(
        [
            Candidate(
                code="SH513310",
                name="x",
                decision="卖/看跌延续",
                section="immediate_risk",
                source="today",
                signal_date="2026-07-28",
            ),
            Candidate(
                code="SH513310",
                name="x",
                decision="买观察",
                section="conditional",
                source="history",
                signal_date="2026-07-20",
                trigger_price=5.0,
            ),
        ]
    )
    main = [c for c in resolved if c.is_main]
    assert len(main) == 1
    assert main[0].decision == "卖/看跌延续"
    buy = next(c for c in resolved if c.decision == "买观察")
    assert buy.suppressed_by


def test_buy_confirm_intraday_vs_eod():
    row = {
        "code": "SZ000001",
        "name": "t",
        "side": "绿",
        "action": "买确认候选",
        "signal_date": "2026-07-28",
        "signal_px": 1.0,
        "last_px": 1.0,
    }
    intra = candidates_from_today_triangle(row, trade_s="2026-07-28", intraday=True)
    eod = candidates_from_today_triangle(row, trade_s="2026-07-29", intraday=False)
    assert intra is not None and eod is not None
    assert intra.decision == "买确认待收盘"
    assert "禁止盘中追买" in intra.operation
    assert eod.decision == "买确认下一开盘"
    assert "开盘试多" in eod.operation


def test_multiple_buy_watch_keeps_latest_only():
    resolved = resolve_main_per_code(
        [
            Candidate(
                code="X",
                name="",
                decision="买观察",
                section="conditional",
                source="history",
                signal_date="2026-07-10",
                trigger_price=1.0,
            ),
            Candidate(
                code="X",
                name="",
                decision="买观察",
                section="conditional",
                source="history",
                signal_date="2026-07-22",
                trigger_price=1.2,
            ),
        ]
    )
    mains = [c for c in resolved if c.is_main]
    assert len(mains) == 1
    assert mains[0].signal_date == "2026-07-22"
    assert mains[0].trigger_price == 1.2


def test_top_zone_beats_buy_watch():
    resolved = resolve_main_per_code(
        [
            Candidate(
                code="Y",
                name="",
                decision="顶部交替减仓",
                section="immediate_risk",
                source="history",
                signal_date="2026-07-17~2026-07-28",
            ),
            Candidate(
                code="Y",
                name="",
                decision="买观察",
                section="conditional",
                source="history",
                signal_date="2026-07-20",
            ),
        ]
    )
    main = next(c for c in resolved if c.is_main)
    assert main.decision == "顶部交替减仓"
    buy = next(c for c in resolved if c.decision == "买观察")
    assert "顶部交替减仓" in buy.suppressed_by


def test_main_table_unique_code_and_audit_keeps_all_today():
    today = pd.DataFrame(
        [
            {
                "code": "A",
                "name": "a",
                "side": "红",
                "action": "卖/看跌延续",
                "signal_date": "2026-07-28",
                "last_px": 1.0,
                "prior10": -0.05,
                "rv_expand": False,
                "vol_pct": 0.4,
                "pred_cdf": 0.3,
            },
            {
                "code": "B",
                "name": "b",
                "side": "红",
                "action": "买观察",
                "signal_date": "2026-07-28",
                "last_px": 2.0,
                "signal_high": 2.1,
                "prior10": -0.06,
                "rv_expand": True,
                "vol_pct": 0.9,
                "pred_cdf": 0.2,
            },
        ]
    )
    resolved = collect_candidates(
        today=today,
        dec=pd.DataFrame(),
        trade_s="2026-07-28",
        asof_s="2026-07-28",
        intraday=True,
    )
    main = main_table(resolved)
    assert main["code"].is_unique
    md = build_ops_markdown(
        as_of=pd.Timestamp("2026-07-28"),
        trade_date=pd.Timestamp("2026-07-28"),
        html_dir=Path("h"),
        val=Path("v"),
        n_codes=2,
        live_captured=None,
        main_df=main,
        all_cand=pd.DataFrame([c.__dict__ for c in resolved]),
        today=today,
        intraday=True,
    )
    assert md.count("| A |") + md.count("| B |") >= 4  # main + audit
    assert "## 附录A. 2026-07-28 今日三角（审计）" in md
    assert "归入五类" in md


def test_intraday_vs_eod_filenames_contract(tmp_path: Path):
    """Document expected file naming; full scan covered by live pack regen."""
    asof_tag = "20260728"
    trade_tag = "20260729"
    # Simulate what main() writes
    md = "# test\n"
    (tmp_path / f"OPS_{asof_tag}.md").write_text(md, encoding="utf-8")
    # intraday
    (tmp_path / f"OPS_intraday_{asof_tag}.md").write_text(md, encoding="utf-8")
    stale = tmp_path / f"OPS_for_{trade_tag}_from_{asof_tag}.md"
    stale.write_text("stale", encoding="utf-8")
    # cleanup like main
    for p in tmp_path.glob("OPS_for_*_from_*.md"):
        p.unlink()
    assert (tmp_path / f"OPS_intraday_{asof_tag}.md").exists()
    assert not list(tmp_path.glob("OPS_for_*_from_*.md"))
    # eod keeps OPS_for
    eod = tmp_path / f"OPS_for_{trade_tag}_from_{asof_tag}.md"
    eod.write_text(md, encoding="utf-8")
    assert eod.exists()


def test_regenerated_20260728_pack_sh515050_context():
    """Assert live regenerated pack (skip if scan outputs missing)."""
    md_path = SCAN / "OPS_intraday_20260728.md"
    primary = SCAN / "ops_primary_fresh.csv"
    if not md_path.exists() or not primary.exists():
        return
    md = md_path.read_text(encoding="utf-8")
    assert "快照现价" in md
    assert "新绿三角" not in md
    assert list(SCAN.glob("OPS_for_*_from_*.md")) == []
    best = pd.read_csv(primary)
    assert best["code"].is_unique
    row = best[best["code"] == "SH515050"]
    assert not row.empty
    assert row.iloc[0]["decision"] == "红三角不抄底"
    body = md.split("## E.")[0]
    assert body.count("| SH515050 |") == 1
    assert "不抄底" in body
    # Sample triad sanity
    assert "| SH513310 |" in body
    assert "| SZ159320 |" in body
    # SH516310: if still above MA20 → 趋势内交替观察（非顶部）
    bank = best[best["code"] == "SH516310"]
    if not bank.empty and str(bank.iloc[0]["decision"]) in {
        "趋势内交替观察",
        "顶部交替警戒",
    }:
        op = str(bank.iloc[0]["operation"])
        assert str(bank.iloc[0]["decision"]) == "趋势内交替观察" or "非顶部" in op
        assert "借反弹清" not in op or "不要按顶部" in op
        assert "趋势未破" in md or "未破@" in md
        assert "顶部交替区警戒" not in op


def test_scanned_no_action_roster_lists_quiet_codes():
    from etf_daily.regime.scan_triangle_decision_ops import (
        build_scanned_no_action_rows,
    )

    main = pd.DataFrame(
        [
            {
                "code": "A",
                "name": "活跃",
                "decision": "买观察",
                "section": "conditional",
                "is_main": True,
            }
        ]
    )
    quiet = build_scanned_no_action_rows(
        main_df=main,
        all_cand=main,
        today=pd.DataFrame(),
        scanned_codes=["A", "B", "C"],
        name_map={"B": "静默乙", "C": "静默丙"},
        failed=[("C", "boom")],
    )
    codes = {r["code"]: r for r in quiet}
    assert "A" not in codes
    assert "B" in codes
    assert "已扫无动作" in codes["B"]["reason"]
    assert "扫描失败" in codes["C"]["reason"]

    md = build_ops_markdown(
        as_of=pd.Timestamp("2026-07-28"),
        trade_date=pd.Timestamp("2026-07-28"),
        html_dir=Path("h"),
        val=Path("v"),
        n_codes=3,
        live_captured=None,
        main_df=main,
        all_cand=main,
        today=pd.DataFrame(),
        intraday=True,
        scanned_codes=["A", "B"],
        name_map={"B": "静默乙"},
    )
    assert "## 附录B. 已扫无动作 / 未入主表" in md
    assert "| B | 静默乙 |" in md
    assert "已扫无动作" in md


def test_ma20_trend_state_intact_and_break():
    from etf_daily.lib.indicators import ma20_trend_state

    idx = pd.bdate_range("2026-05-01", periods=60)
    # Uptrend then stay above rising MA20
    vals = [100 + i * 0.5 for i in range(60)]
    close = pd.Series(vals, index=idx)
    st = ma20_trend_state(close, idx[-1])
    assert st["trend_intact"] is True
    assert st["above_ma20"] is True
    assert st["trend_since"] is not None
    assert st["days_above"] >= 1

    # Break last 3 days below MA → last_break = first day of current below streak
    vals2 = vals[:]
    for j in range(3):
        vals2[-(j + 1)] = 50.0
    close2 = pd.Series(vals2, index=idx)
    st2 = ma20_trend_state(close2, idx[-1])
    assert st2["trend_intact"] is False
    assert st2["above_ma20"] is False
    assert st2["last_break"] == str(idx[-3].date())


def test_ops_bucket_five_way():
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    assert ops_bucket("买观察") == "buy"
    assert ops_bucket("顶部交替减仓") == "sell"
    assert ops_bucket("红三角不抄底") == "down"
    assert ops_bucket("趋势内交替观察") == "up"
    # 无趋势上下文时底部交替仍归横盘；确认跌破/站上则跟趋势。
    assert ops_bucket("底部交替观察") == "range"
    assert ops_bucket("底部交替观察", trend_intact=False) == "down"
    assert ops_bucket("底部交替观察", trend_intact=True) == "up"


def test_reclaim_buy_trigger_note_shows_pct():
    from etf_daily.lib.triangle_ops_resolve import reclaim_buy_trigger_note

    note = reclaim_buy_trigger_note(trigger_hi=1.069, ref_px=1.042)
    assert "1.069" in note
    assert "+2.6%" in note or "+2.5%" in note  # 1.069/1.042-1 ≈ 2.59%
    assert "趋势可能买入点" in note
    assert "买入触发" in note

    done = reclaim_buy_trigger_note(trigger_hi=1.069, ref_px=1.072)
    assert "已达/超过" in done or "已触发" in done


def test_bottom_rgr_all_below_ma20_not_top_zone():
    """numpy.False_ must not be treated as 'not False' → fake 顶部交替区."""
    import numpy as np
    from etf_daily.scripts.backtest_triangle_decision_rules import (
        RuleConfig,
        detect_alternating_zones,
    )

    ann = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-13"),
                "side": "down",
                "close": 0.518,
                "above_ma20": np.False_,
                "vol5_pct_120d": 0.67,
            },
            {
                "as_of": pd.Timestamp("2026-07-23"),
                "side": "up",
                "close": 0.527,
                "above_ma20": np.False_,
                "vol5_pct_120d": 0.59,
            },
            {
                "as_of": pd.Timestamp("2026-07-24"),
                "side": "down",
                "close": 0.510,
                "above_ma20": np.False_,
                "vol5_pct_120d": 0.83,
            },
        ]
    )
    zs = detect_alternating_zones(ann, RuleConfig())
    assert len(zs) == 1
    assert zs.iloc[0]["kind"] == "底部交替区"
    assert zs.iloc[0]["pattern"] == "RGR"
    assert zs.iloc[0]["action_hint"] == "底部变盘观察"


def test_bottom_zone_resolve_not_top_clear():
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    hist = pd.DataFrame(
        [
            {
                "code": "SZ159745",
                "name": "建材ETF国泰",
                "decision": "底部交替观察",
                "status": "底部交替区",
                "signal_date": "2026-07-13~2026-07-24",
                "side": "RGR",
                "signal_px": 0.527,
                "last_px": 0.520,
                "entry_state": "forward",
                "operation": (
                    "2026-07-29底部盘整/变盘观察（形态RGR；波动聚集）；"
                    "非顶部清仓结构；不按顶部借反弹清"
                ),
                "invalidation": "放量跌破区低或重新站上MA20后改判",
                "trend_intact": False,
                "trend_label": "已破MA20",
                "above_ma20": False,
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    main = main_table(resolved)
    assert main.iloc[0]["decision"] == "底部交替观察"
    assert main.iloc[0]["section"] == "pending_confirm"
    assert "借反弹清" not in str(main.iloc[0]["operation"]) or "不按" in str(
        main.iloc[0]["operation"]
    )
    # 确认跌破时底部交替进趋势下跌，不再误进横盘（SH588710 类）。
    assert ops_bucket(
        main.iloc[0]["decision"], trend_intact=main.iloc[0]["trend_intact"]
    ) == "down"

    hug = hist.copy()
    hug.loc[0, "trend_hugging"] = True
    hug.loc[0, "trend_label"] = "MA20缠绕"
    resolved_hug = collect_candidates(
        today=pd.DataFrame(),
        dec=hug,
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    main_hug = main_table(resolved_hug)
    assert main_hug.iloc[0]["trend_intact"] is None
    assert ops_bucket(
        main_hug.iloc[0]["decision"], trend_intact=main_hug.iloc[0]["trend_intact"]
    ) == "range"


def test_top_zone_superseded_by_post_zone_red_run():
    from etf_daily.scripts.backtest_triangle_decision_rules import (
        top_zone_post_red_run,
        top_zone_superseded_by_red_run,
    )

    ann = pd.DataFrame(
        [
            {"as_of": "2026-07-13", "side": "down", "above_ma20": False, "high": 1.18, "close": 1.13, "low": 1.12},
            {"as_of": "2026-07-15", "side": "up", "above_ma20": True, "high": 1.18, "close": 1.17, "low": 1.10},
            {"as_of": "2026-07-17", "side": "down", "above_ma20": False, "high": 1.18, "close": 1.11, "low": 1.10},
            {"as_of": "2026-07-22", "side": "down", "above_ma20": False, "high": 1.14, "close": 1.10, "low": 1.09},
            {"as_of": "2026-07-24", "side": "down", "above_ma20": False, "high": 1.069, "close": 1.042, "low": 1.040},
        ]
    )
    ok, why = top_zone_superseded_by_red_run(
        ann,
        zone_end=pd.Timestamp("2026-07-17"),
        as_of=pd.Timestamp("2026-07-28"),
        min_reds=2,
    )
    assert ok is True
    assert "2026-07-22" in why and "2026-07-24" in why
    ok2, why2, last = top_zone_post_red_run(
        ann,
        zone_end=pd.Timestamp("2026-07-17"),
        as_of=pd.Timestamp("2026-07-28"),
        min_reds=2,
    )
    assert ok2 is True
    assert last is not None
    assert str(pd.Timestamp(last["as_of"]).date()) == "2026-07-24"
    assert float(last["high"]) == 1.069

    # Only one post-zone red → still valid top zone
    ok3, _ = top_zone_superseded_by_red_run(
        ann.iloc[:4],
        zone_end=pd.Timestamp("2026-07-17"),
        as_of=pd.Timestamp("2026-07-28"),
        min_reds=2,
    )
    assert ok3 is False


def test_post_top_buy_watch_beats_expired_top_note():
    resolved = resolve_main_per_code(
        [
            Candidate(
                code="SH516770",
                name="游戏ETF",
                decision="顶部区已失效",
                section="position_mgmt",
                source="history",
                signal_date="2026-07-13~2026-07-17",
            ),
            Candidate(
                code="SH516770",
                name="游戏ETF",
                decision="买观察",
                section="conditional",
                source="history",
                signal_date="2026-07-24",
                trigger_price=1.069,
                operation="顶部破坏后连红低点；等收复",
            ),
        ]
    )
    main = next(c for c in resolved if c.is_main)
    assert main.decision == "买观察"
    assert main.section == "conditional"


def test_top_zone_expired_resolve_not_immediate_risk():
    hist = pd.DataFrame(
        [
            {
                "code": "SH516770",
                "name": "游戏ETF华泰柏瑞",
                "decision": "顶部区已失效",
                "status": "顶部交替区已破坏",
                "signal_date": "2026-07-13~2026-07-17",
                "side": "RGR",
                "signal_px": 1.172,
                "last_px": 1.068,
                "entry_state": "expired",
                "operation": (
                    "原顶部区2026-07-13~2026-07-17（形态RGR）不再作为减仓主依据；"
                    "区后同色连红；2026-07-29不按借反弹清执行"
                ),
                "invalidation": "结构已破坏",
                "trend_intact": False,
                "trend_label": "已破MA20",
                "above_ma20": False,
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    main = main_table(resolved)
    assert main.iloc[0]["decision"] == "顶部区已失效"
    assert main.iloc[0]["section"] == "position_mgmt"
    assert "不按借反弹清" in str(main.iloc[0]["operation"])


def test_top_zone_intact_is_midtrend_not_top_alert():
    hist = pd.DataFrame(
        [
            {
                "code": "SH516310",
                "name": "银行ETF易方达",
                "decision": "趋势内交替观察",
                "status": "趋势交替区",
                "signal_date": "2026-07-20~2026-07-22",
                "side": "GRG",
                "signal_px": 1.339,
                "last_px": 1.357,
                "entry_state": "forward",
                "operation": (
                    "2026-07-29趋势上行中的交替（形态GRG），非顶部结构；"
                    "自2026-07-10起收盘仍站MA20（趋势未破），持有/可逢低；"
                    "跌破MA20后再评估减仓，不要按顶部借反弹清"
                ),
                "invalidation": "跌破MA20则退出上行中继解读",
                "trend_intact": True,
                "trend_since": "2026-07-10",
                "trend_label": "未破@2026-07-10",
                "above_ma20": True,
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    main = main_table(resolved)
    assert len(main) == 1
    assert main.iloc[0]["decision"] == "趋势内交替观察"
    assert main.iloc[0]["section"] == "pending_confirm"
    op = str(main.iloc[0]["operation"])
    assert "非顶部" in op
    assert "顶部交替" not in op
    assert "借反弹清" not in op or "不要按顶部" in op


def test_legacy_top_alert_maps_to_midtrend():
    hist = pd.DataFrame(
        [
            {
                "code": "SH516310",
                "name": "银行ETF易方达",
                "decision": "顶部交替警戒",
                "status": "顶部交替区",
                "signal_date": "2026-07-20~2026-07-22",
                "side": "GRG",
                "signal_px": 1.339,
                "last_px": 1.343,
                "entry_state": "forward",
                "operation": "legacy",
                "invalidation": "跌破MA20则改减仓",
                "trend_intact": True,
                "trend_since": "2026-06-30",
                "trend_label": "未破@2026-06-30",
                "above_ma20": True,
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-07-28",
        asof_s="2026-07-28",
        intraday=True,
    )
    main = main_table(resolved)
    assert main.iloc[0]["decision"] == "趋势内交替观察"


def test_top_zone_broken_keeps_cut_wording():
    hist = pd.DataFrame(
        [
            {
                "code": "X",
                "name": "x",
                "decision": "顶部交替减仓",
                "status": "顶部交替区",
                "signal_date": "2026-07-20~2026-07-22",
                "side": "GRG",
                "signal_px": 1.0,
                "last_px": 0.9,
                "entry_state": "forward",
                "operation": "若持仓：2026-07-28借反弹清（第一红@2026-07-21）；形态GRG；已破MA20；2026-07-28不加仓",
                "invalidation": "跌破后反弹不加仓",
                "trend_intact": False,
                "trend_label": "已破MA20",
                "above_ma20": False,
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-07-28",
        asof_s="2026-07-28",
        intraday=True,
    )
    main = main_table(resolved)
    assert main.iloc[0]["decision"] == "顶部交替减仓"
    assert main.iloc[0]["section"] == "immediate_risk"
    assert "借反弹清" in str(main.iloc[0]["operation"])


def test_top_alert_does_not_suppress_buy_watch():
    resolved = resolve_main_per_code(
        [
            Candidate(
                code="Y",
                name="",
                decision="趋势内交替观察",
                section="pending_confirm",
                source="history",
                signal_date="2026-07-20~2026-07-22",
                trend_intact=True,
                trend_since="2026-06-30",
                trend_label="未破@2026-06-30",
            ),
            Candidate(
                code="Y",
                name="",
                decision="买观察",
                section="conditional",
                source="history",
                signal_date="2026-07-22",
                trigger_price=1.2,
            ),
        ]
    )
    # Mid-trend watch is higher section priority than conditional buy watch → main is watch,
    # but buy watch must NOT be marked suppressed_by risk.
    buy = next(c for c in resolved if c.decision == "买观察")
    assert not buy.suppressed_by or "顶部交替减仓" not in buy.suppressed_by
    # specifically: alert must not use immediate_risk suppress path
    assert "immediate_risk" not in (buy.suppressed_by or "")
    assert "顶部交替警戒" not in (buy.suppressed_by or "")
    assert "趋势内交替观察" not in (buy.suppressed_by or "")


def test_gated_predicates():
    from etf_daily.lib.triangle_ops_resolve import (
        buy_watch_passes_gate,
        post_top_buy_passes_gate,
        sell_passes_gate,
    )

    assert sell_passes_gate("中") is True
    assert sell_passes_gate("弱") is True
    assert sell_passes_gate("强") is False
    assert buy_watch_passes_gate(0.85) is True
    assert buy_watch_passes_gate(0.72) is False
    assert buy_watch_passes_gate(None) is False
    assert post_top_buy_passes_gate(-0.09) is True   # 深跌
    assert post_top_buy_passes_gate(0.01) is True    # 翻红
    assert post_top_buy_passes_gate(-0.04) is False  # 中段
    assert post_top_buy_passes_gate(None) is False


def test_today_strong_sell_gated_to_green_watch():
    c = candidates_from_today_triangle(
        {
            "code": "SH513050",
            "name": "中概互联网ETF易方达",
            "side": "绿",
            "action": "卖预警",
            "strength": "强",
            "signal_date": "2026-07-28",
            "signal_px": 1.5,
            "last_px": 1.5,
            "prior10": 0.12,
            "rv_expand": True,
            "vol_pct": 0.95,
            "pred_cdf": 0.95,
        },
        trade_s="2026-07-29",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "绿三角观察"
    assert "gated" in c.decision_reason


def test_today_mid_sell_passes_gate():
    c = candidates_from_today_triangle(
        {
            "code": "SH513050",
            "name": "中概互联网ETF易方达",
            "side": "绿",
            "action": "卖预警",
            "strength": "中",
            "signal_date": "2026-07-28",
            "signal_px": 1.5,
            "last_px": 1.5,
            "prior10": 0.08,
            "rv_expand": True,
            "vol_pct": 0.85,
            "pred_cdf": 0.93,
        },
        trade_s="2026-07-29",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "卖预警待确认"


def test_today_low_vol_buy_watch_gated_to_no_bottom_fishing():
    c = candidates_from_today_triangle(
        {
            "code": "SH516770",
            "name": "游戏ETF华泰柏瑞",
            "side": "红",
            "action": "买观察",
            "strength": "弱",
            "signal_date": "2026-07-28",
            "signal_px": 1.0,
            "signal_high": 1.05,
            "last_px": 1.0,
            "prior10": -0.06,
            "rv_expand": True,
            "vol_pct": 0.72,
            "pred_cdf": 0.08,
        },
        trade_s="2026-07-29",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "红三角不抄底"
    assert "gated" in c.decision_reason


def test_today_high_vol_buy_watch_passes_gate():
    c = candidates_from_today_triangle(
        {
            "code": "SH516770",
            "name": "游戏ETF华泰柏瑞",
            "side": "红",
            "action": "买观察",
            "strength": "中",
            "signal_date": "2026-07-28",
            "signal_px": 1.0,
            "signal_high": 1.05,
            "last_px": 1.0,
            "prior10": -0.06,
            "rv_expand": True,
            "vol_pct": 0.85,
            "pred_cdf": 0.08,
        },
        trade_s="2026-07-29",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "买观察"


def test_late_entry_state_classification():
    from etf_daily.lib.triangle_ops_resolve import late_entry_state

    assert late_entry_state(1.068, 1.061, 1.040) == "接近执行点"   # +0.66%
    assert late_entry_state(1.10, 1.061, 1.040) == "已远离执行点"    # +3.7%
    assert late_entry_state(1.03, 1.061, 1.040) == "反弹失败"        # below stop
    assert late_entry_state(None, 1.061, 1.040) == "已远离执行点"


def test_buy_exec_summary_late_joinable():
    from etf_daily.lib.triangle_ops_resolve import buy_exec_summary

    trig, need, exec_s = buy_exec_summary(
        {
            "decision": "买观察",
            "status": "接近执行点仍可参与",
            "last_px": 1.068,
            "exec_px": 1.061,
            "operation": (
                "执行点@1.061（@2026-07-28）；现价距执行点+0.7%（参与带内）；"
                "可轻仓参与，止损1.040"
            ),
        }
    )
    assert trig == "1.061"
    assert need.startswith("已达@2026-07-28")
    assert "已涨+0.7%" in need
    assert "轻仓参与" in exec_s


def test_buy_exec_summary_expired_lists_trigger_and_rise():
    from etf_daily.lib.triangle_ops_resolve import buy_exec_summary

    trig, need, exec_s = buy_exec_summary(
        {
            "decision": "已过试多窗口",
            "status": "收复窗口已过",
            "entry_state": "expired",
            "last_px": 1.835,
            "operation": (
                "忽略红@2026-07-24曾于2026-07-27收复，执行点@1.735（@2026-07-28）已过；"
                "现价已涨离+5.8%；不追买"
            ),
        }
    )
    assert trig == "1.735"
    assert "已达@2026-07-28" in need
    assert "已涨+5.8%" in need
    assert "不追买" in exec_s


def test_history_row_late_joinable_becomes_buy_watch():
    c = candidates_from_history_row(
        {
            "code": "SH516770",
            "name": "游戏ETF华泰柏瑞",
            "decision": "买观察",
            "status": "接近执行点仍可参与",
            "entry_state": "forward",
            "signal_date": "2026-07-24",
            "side": "红",
            "signal_px": 1.042,
            "last_px": 1.068,
            "exec_px": 1.061,
            "stop": 1.040,
            "operation": "现价距执行点+0.7%（参与带内）；可轻仓参与，止损1.040",
        },
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "买观察"
    assert c.section == "pending_confirm"
    assert abs((c.trigger_price or 0) - 1.061) < 1e-9


def test_history_row_run_away_stays_expired():
    c = candidates_from_history_row(
        {
            "code": "SH516770",
            "name": "游戏ETF华泰柏瑞",
            "decision": "买观察",
            "status": "收复窗口已过",
            "entry_state": "expired",
            "signal_date": "2026-07-24",
            "side": "红",
            "signal_px": 1.042,
            "last_px": 1.12,
            "operation": "现价已涨离执行点+5.6%；不追买，等回踩执行点/MA5企稳再看",
        },
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "已过试多窗口"
    assert c.section == "pending_confirm"


def test_no_triangle_trend_decisions_bucket_and_candidate():
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    assert ops_bucket("趋势上涨-无三角") == "up"
    assert ops_bucket("趋势下跌-无三角") == "down"
    assert ops_bucket("横盘-无三角") == "range"

    c = candidates_from_history_row(
        {
            "code": "SZ159502",
            "name": "标普生物科技ETF嘉实",
            "decision": "趋势下跌-无三角",
            "status": "无三角趋势跟踪",
            "entry_state": "forward",
            "signal_date": "2026-07-28",
            "last_px": 1.498,
            "operation": "无红/绿三角信号；已破MA20（本轮跌破自@2026-07-16）（-6.0%）；不抄底不加仓",
            "trend_intact": False,
            "trend_label": "已破MA20",
        },
        trade_s="2026-07-29",
        asof_s="2026-07-28",
        intraday=False,
    )
    assert c is not None
    assert c.decision == "趋势下跌-无三角"
    assert c.section == "position_mgmt"
    assert ops_bucket(c.decision) == "down"


def test_today_gated_green_bucket_follows_trend():
    """Gated 绿三角观察 must land in 趋势上涨/下跌 by trend, not always 横盘."""
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    row = {
        "code": "SH516770",
        "name": "游戏ETF华泰柏瑞",
        "side": "绿",
        "action": "卖预警",
        "strength": "强",
        "signal_date": "2026-07-30",
        "signal_px": 1.13,
        "signal_high": 1.15,
        "last_px": 1.178,
        "vol_pct": 1.0,
        "prior10": 0.05,
        "above_ma20": True,
        "rv_expand": True,
        "pred_cdf": 0.99,
        "trend_intact": True,
        "trend_since": "2026-07-24",
        "trend_label": "未破@2026-07-24",
        "trend_hugging": False,
    }
    c = candidates_from_today_triangle(row, trade_s="2026-07-31", intraday=False)
    assert c is not None
    assert c.decision == "绿三角观察"
    assert c.trend_intact is True
    assert ops_bucket(c.decision, trend_intact=c.trend_intact) == "up"

    down = dict(row, trend_intact=False, trend_label="已破MA20")
    c1 = candidates_from_today_triangle(down, trade_s="2026-07-31", intraday=False)
    assert c1.trend_intact is False
    assert ops_bucket(c1.decision, trend_intact=c1.trend_intact) == "down"

    hug = dict(row, trend_hugging=True)
    c2 = candidates_from_today_triangle(hug, trade_s="2026-07-31", intraday=False)
    assert c2.trend_intact is None
    assert ops_bucket(c2.decision, trend_intact=c2.trend_intact) == "range"


def test_top_zone_expired_bucket_follows_trend():
    """顶部区已失效是历史结构事件，五类桶跟当前趋势（SH516770 类：趋势未破→上涨）。"""
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    assert ops_bucket("顶部区已失效", trend_intact=True) == "up"
    assert ops_bucket("顶部区已失效", trend_intact=False) == "down"
    assert ops_bucket("顶部区已失效", trend_intact=None) == "range"


def test_pick_latest_deep_red_no_zombie_fallback_after_newer_reclaim():
    """SH588850-style: 07-30 reclaimed on 07-31 → do not fall back to 07-28."""
    ann = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-28"),
                "side": "down",
                "prior10": -0.188,
                "high": 1.686,
                "close": 1.592,
            },
            {
                "as_of": pd.Timestamp("2026-07-30"),
                "side": "down",
                "prior10": -0.202,
                "high": 1.580,
                "close": 1.480,
            },
        ]
    )
    px = pd.DataFrame(
        [
            {"as_of": pd.Timestamp("2026-07-28"), "$close": 1.592, "$high": 1.686},
            {"as_of": pd.Timestamp("2026-07-29"), "$close": 1.587, "$high": 1.612},
            {"as_of": pd.Timestamp("2026-07-30"), "$close": 1.480, "$high": 1.580},
            {"as_of": pd.Timestamp("2026-07-31"), "$close": 1.591, "$high": 1.651},
            {"as_of": pd.Timestamp("2026-08-03"), "$close": 1.554, "$high": 1.602},
        ]
    )
    as_of = pd.Timestamp("2026-08-03")
    buy_from = pd.Timestamp("2026-07-22")
    picked = pick_latest_deep_red_for_reclaim_watch(
        ann, px, as_of=as_of, buy_from=buy_from, buy_watch_dates=set()
    )
    assert picked is None
    recl = pick_latest_deep_red_reclaimed(
        ann, px, as_of=as_of, buy_from=buy_from, buy_watch_dates=set()
    )
    assert recl is not None
    assert pd.Timestamp(recl["as_of"]).normalize() == pd.Timestamp("2026-07-30")

    # Latest still unreclaimed → keep that bar (not an older lower high).
    px_no_reclaim = px.copy()
    px_no_reclaim.loc[px_no_reclaim["as_of"] == "2026-07-31", "$close"] = 1.570
    picked2 = pick_latest_deep_red_for_reclaim_watch(
        ann, px_no_reclaim, as_of=as_of, buy_from=buy_from, buy_watch_dates=set()
    )
    assert picked2 is not None
    assert pd.Timestamp(picked2["as_of"]).normalize() == pd.Timestamp("2026-07-30")
    assert float(picked2["high"]) == 1.580
    assert (
        pick_latest_deep_red_reclaimed(
            ann, px_no_reclaim, as_of=as_of, buy_from=buy_from, buy_watch_dates=set()
        )
        is None
    )


def test_sell_continuation_reclaim_becomes_buy_sh513650_style():
    """07-30 卖/看跌延续红 → 07-31 收盘收复 → 08-03 开盘买点。"""
    from etf_daily.scripts.backtest_triangle_decision_rules import RuleConfig

    ann = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-30"),
                "side": "down",
                "action": "卖/看跌延续",
                "prior10": -0.03856,
                "high": 1.885,
                "low": 1.866,
                "close": 1.870,
                "vol5_pct_120d": 0.50,
                "strength": "",
            }
        ]
    )
    px = pd.DataFrame(
        [
            {"as_of": pd.Timestamp("2026-07-30"), "$close": 1.870, "$high": 1.885, "$open": 1.882, "$low": 1.866},
            {"as_of": pd.Timestamp("2026-07-31"), "$close": 1.897, "$high": 1.907, "$open": 1.905, "$low": 1.894},
            {"as_of": pd.Timestamp("2026-08-03"), "$close": 1.911, "$high": 1.918, "$open": 1.907, "$low": 1.905},
            {"as_of": pd.Timestamp("2026-08-04"), "$close": 1.956, "$high": 1.956, "$open": 1.937, "$low": 1.937},
        ]
    )
    cfg = RuleConfig()
    # As-of reclaim day: next open still ahead → pending_open
    plans_731 = plan_sell_continuation_reclaim_buys(
        ann,
        px,
        as_of=pd.Timestamp("2026-07-31"),
        buy_from=pd.Timestamp("2026-07-20"),
        cfg=cfg,
        last_px=1.897,
    )
    assert len(plans_731) == 1
    assert plans_731[0]["phase"] == "pending_open"
    assert plans_731[0]["reclaim_d"] == pd.Timestamp("2026-07-31")
    # Next open may already be in px (test fixture) or absent in live EOD clip.
    assert plans_731[0]["exec_d"] in {
        None,
        pd.Timestamp("2026-08-03"),
    }
    # Reclaim-day-only clip: next open missing → still pending_open
    plans_731_clip = plan_sell_continuation_reclaim_buys(
        ann,
        px[px["as_of"] <= "2026-07-31"],
        as_of=pd.Timestamp("2026-07-31"),
        buy_from=pd.Timestamp("2026-07-20"),
        cfg=cfg,
        last_px=1.897,
    )
    assert len(plans_731_clip) == 1
    assert plans_731_clip[0]["phase"] == "pending_open"
    assert plans_731_clip[0]["exec_d"] is None

    # As-of exec day: still near exec
    plans_803 = plan_sell_continuation_reclaim_buys(
        ann,
        px,
        as_of=pd.Timestamp("2026-08-03"),
        buy_from=pd.Timestamp("2026-07-20"),
        cfg=cfg,
        last_px=1.911,
    )
    assert plans_803[0]["phase"] == "near_exec"

    # Resolve into buy bucket (not 趋势上涨-only silence)
    hist = pd.DataFrame(
        [
            {
                "code": "SH513650",
                "name": "标普500ETF南方",
                "decision": "买观察",
                "status": "待下一交易日开盘",
                "signal_date": "2026-07-30",
                "side": "红",
                "signal_px": 1.870,
                "signal_high": 1.885,
                "last_px": 1.897,
                "entry_state": "forward",
                "operation": (
                    "卖/看跌延续红@2026-07-30高点1.885已于2026-07-31收盘收复；"
                    "2026-08-03开盘试多（小仓）"
                ),
                "trend_intact": False,
                "trend_label": "已破MA20",
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-08-03",
        asof_s="2026-07-31",
        intraday=False,
    )
    main = main_table(resolved)
    assert main.iloc[0]["decision"] == "买观察"
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    assert ops_bucket(main.iloc[0]["decision"]) == "buy"


def test_soft_red_ignore_reclaim_sh513080_style():
    """忽略红（浅跌）07-24 → 07-27 收复 → 07-28 开盘买；08-04 仍应识别为已过窗口。"""
    from etf_daily.scripts.backtest_triangle_decision_rules import RuleConfig

    ann = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-24"),
                "side": "down",
                "action": "忽略",
                "prior10": -0.021,
                "high": 1.723,
                "low": 1.713,
                "close": 1.717,
                "vol5_pct_120d": 0.417,
                "strength": "",
            }
        ]
    )
    px = pd.DataFrame(
        [
            {"as_of": pd.Timestamp("2026-07-24"), "$close": 1.717, "$high": 1.723, "$open": 1.722, "$low": 1.713},
            {"as_of": pd.Timestamp("2026-07-27"), "$close": 1.749, "$high": 1.753, "$open": 1.730, "$low": 1.730},
            {"as_of": pd.Timestamp("2026-07-28"), "$close": 1.737, "$high": 1.744, "$open": 1.735, "$low": 1.733},
            {"as_of": pd.Timestamp("2026-08-04"), "$close": 1.835, "$high": 1.835, "$open": 1.818, "$low": 1.809},
        ]
    )
    cfg = RuleConfig()
    plans = plan_sell_continuation_reclaim_buys(
        ann,
        px,
        as_of=pd.Timestamp("2026-08-04"),
        buy_from=pd.Timestamp("2026-07-23"),
        cfg=cfg,
        last_px=1.835,
    )
    assert len(plans) == 1
    assert plans[0]["reclaim_d"] == pd.Timestamp("2026-07-27")
    assert plans[0]["exec_d"] == pd.Timestamp("2026-07-28")
    assert abs(plans[0]["exec_px"] - 1.735) < 1e-6
    assert plans[0]["phase"] == "expired"


def test_sell_continuation_ignores_false_reclaim_sz159509_style():
    """07-29 假收复、07-30 跌破后，以 07-31 再收复为买点。"""
    from etf_daily.scripts.backtest_triangle_decision_rules import RuleConfig

    ann = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-28"),
                "side": "down",
                "action": "卖/看跌延续",
                "prior10": -0.0426,
                "high": 2.531,
                "low": 2.510,
                "close": 2.519,
                "vol5_pct_120d": 0.325,
                "strength": "",
            }
        ]
    )
    px = pd.DataFrame(
        [
            {"as_of": pd.Timestamp("2026-07-28"), "$close": 2.519, "$high": 2.531, "$open": 2.521, "$low": 2.510},
            {"as_of": pd.Timestamp("2026-07-29"), "$close": 2.547, "$high": 2.549, "$open": 2.515, "$low": 2.512},
            {"as_of": pd.Timestamp("2026-07-30"), "$close": 2.502, "$high": 2.512, "$open": 2.510, "$low": 2.496},
            {"as_of": pd.Timestamp("2026-07-31"), "$close": 2.602, "$high": 2.617, "$open": 2.605, "$low": 2.587},
            {"as_of": pd.Timestamp("2026-08-03"), "$close": 2.630, "$high": 2.641, "$open": 2.612, "$low": 2.612},
            {"as_of": pd.Timestamp("2026-08-04"), "$close": 2.722, "$high": 2.722, "$open": 2.702, "$low": 2.694},
        ]
    )
    cfg = RuleConfig()
    # As-of 07-30: false reclaim lost → no active buy plan
    plans_730 = plan_sell_continuation_reclaim_buys(
        ann,
        px,
        as_of=pd.Timestamp("2026-07-30"),
        buy_from=pd.Timestamp("2026-07-20"),
        cfg=cfg,
        last_px=2.502,
    )
    assert plans_730 == []

    # As-of 07-31: real reclaim → next open 08-03
    plans_731 = plan_sell_continuation_reclaim_buys(
        ann,
        px,
        as_of=pd.Timestamp("2026-07-31"),
        buy_from=pd.Timestamp("2026-07-20"),
        cfg=cfg,
        last_px=2.602,
    )
    assert len(plans_731) == 1
    assert plans_731[0]["phase"] == "pending_open"
    assert plans_731[0]["reclaim_d"] == pd.Timestamp("2026-07-31")
    assert plans_731[0]["exec_d"] == pd.Timestamp("2026-08-03")
    assert abs(plans_731[0]["exec_px"] - 2.612) < 1e-6


def test_zhudi_observation_stays_in_range_not_down():
    """Deep-red reclaim → 筑底观察；仍破 MA20 也不进趋势下跌。"""
    from etf_daily.lib.triangle_ops_resolve import ops_bucket

    assert ops_bucket("筑底观察", trend_intact=False) == "range"
    assert ops_bucket("筑底观察", trend_intact=True) == "range"

    hist = pd.DataFrame(
        [
            {
                "code": "SH588850",
                "name": "科创机械ETF嘉实",
                "decision": "筑底观察",
                "status": "筑底（深跌红已收复）",
                "signal_date": "2026-07-30",
                "side": "RGR",
                "signal_px": 1.744,
                "signal_high": 1.580,
                "last_px": 1.554,
                "entry_state": "forward",
                "operation": (
                    "2026-08-04筑底观察；2026-07-30深跌红高点1.580已收复；"
                    "虽仍破MA20，按筑底而非趋势下跌处理"
                ),
                "invalidation": "再度跌破最近深跌红低点或放量破位则取消筑底解读",
                "trend_intact": False,
                "trend_label": "已破MA20",
                "above_ma20": False,
                "prior10": -0.202,
            }
        ]
    )
    resolved = collect_candidates(
        today=pd.DataFrame(),
        dec=hist,
        trade_s="2026-08-04",
        asof_s="2026-08-03",
        intraday=False,
    )
    main = main_table(resolved)
    assert main.iloc[0]["decision"] == "筑底观察"
    assert (
        ops_bucket(main.iloc[0]["decision"], trend_intact=main.iloc[0]["trend_intact"])
        == "range"
    )
    md = build_ops_markdown(
        as_of=pd.Timestamp("2026-08-03"),
        trade_date=pd.Timestamp("2026-08-04"),
        html_dir=Path("h"),
        val=Path("v"),
        n_codes=1,
        live_captured=None,
        main_df=main,
        all_cand=main,
        today=pd.DataFrame(),
        intraday=False,
    )
    assert "SH588850" in md
    assert "筑底观察" in md
    # Must not land in 趋势下跌 section.
    down_sec = md.split("## 4. 趋势下跌中")[1].split("## 5.")[0]
    assert "SH588850" not in down_sec
    range_sec = md.split("## 6. 无趋势横盘震荡")[1].split("## 附录")[0]
    assert "SH588850" in range_sec
