"""Replay report markdown rendering."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_daily.scripts.replay_symbol_warnings import (
    render_replay_index_md,
    render_symbol_replay_report_md,
    replay_md_filename,
    summarize_replay,
)


def test_render_symbol_replay_report_includes_chinese_name():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-06-04",
                "alert_level": "yellow",
                "suggest_reduce_pct": 0.3,
                "pnl_pct": -0.5,
                "dd_20d_high": -0.024,
                "dist_ma20_pct": 3.2,
                "reasons": "dd_soft|mom_fade",
            }
        ]
    )
    summary = summarize_replay(frame)
    md = render_symbol_replay_report_md(
        code="SH561560",
        name="电力ETF华泰柏瑞",
        frame=frame,
        summary=summary,
        start="20260529",
        end="20260610",
    )
    assert "SH561560 电力ETF华泰柏瑞" in md
    assert "首次黄灯" in md
    assert "2026-06-04" in md


def test_replay_md_filename_includes_chinese_name():
    assert replay_md_filename("SH513080", "法国ETF") == "SH513080_法国ETF_replay.md"


def test_render_replay_index_links_named_md():
    md = render_replay_index_md(
        [
            ("SH561560", "电力ETF华泰柏瑞", {"first_yellow": "2026-06-04", "first_red": "2026-06-05", "yellow_before_pnl7": True, "red_before_pnl7": True}),
        ],
        start="20260529",
        end="20260610",
        output_dir=Path("/tmp/replay"),
    )
    assert "电力ETF华泰柏瑞" in md
    assert "SH561560_电力ETF华泰柏瑞_replay.md" in md
