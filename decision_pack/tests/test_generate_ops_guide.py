from __future__ import annotations

from pathlib import Path

from decision_pack.scripts.generate_ops_guide import build_ops_guide_markdown


def test_build_ops_guide_markdown_20260612():
    pack_dir = Path("~/etf-daily-output/temp/decision_packs/20260612")
    notes = Path(
        "~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/workspace/notes/20260612.md"
    )
    if not (pack_dir / "tier_decision.json").exists():
        return

    md = build_ops_guide_markdown(pack_dir=pack_dir, notes_path=notes)
    assert "# T+1 操作指引" in md
    assert "Tier **1**" in md
    assert "SH561560" in md
    assert "保留观察" in md
    assert "留现金" in md
