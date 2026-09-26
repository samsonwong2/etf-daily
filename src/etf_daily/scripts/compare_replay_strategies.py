#!/usr/bin/env python3
"""Compare replay-based reduce strategies vs buy-and-hold on a date window."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.replay_symbol_warnings import load_holdings_name_map
from etf_daily.lib.replay_simulation import (
    PortfolioSimResult,
    StrategySpec,
    load_holdings_for_sim,
    load_replay_frames,
    simulate_portfolio,
)


def _iso_from_compact(compact: str) -> str:
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"


def build_strategy_grid() -> list[StrategySpec]:
    skip_529 = frozenset({"2026-05-29"})
    pnl_dd = frozenset({"pnl7", "dd5_20d"})

    grid: list[StrategySpec] = [
        StrategySpec(name="hold", mode="hold"),
        StrategySpec(name="warning_A"),
        StrategySpec(
            name="warning_B",
            yellow_reduce_pct=0.30,
            red_reduce_pct=0.50,
            max_yellow_actions=1,
            max_red_actions=1,
            warning_b_branch=True,
        ),
        StrategySpec(
            name="red_only_first",
            respond_yellow=False,
            red_reduce_pct=0.50,
            max_red_actions=1,
        ),
        StrategySpec(
            name="red_pnl_dd_only",
            red_reasons_allow=pnl_dd,
        ),
        StrategySpec(
            name="B_losing_only",
            yellow_reduce_pct=0.30,
            red_reduce_pct=0.50,
            max_yellow_actions=1,
            max_red_actions=1,
            require_pnl_lt=0.0,
            warning_b_branch=True,
        ),
        StrategySpec(
            name="critical_only",
            respond_yellow=False,
            critical_min_suggest=1.0,
        ),
        StrategySpec(
            name="one_shot_worst",
            one_shot_prefer_red=True,
            yellow_reduce_pct=0.30,
            red_reduce_pct=0.50,
        ),
        StrategySpec(
            name="cap50_cumulative",
            max_cumulative_reduce=0.50,
        ),
        StrategySpec(
            name="cooldown_2d",
            cooldown_days=2,
        ),
        StrategySpec(
            name="skip_20260529",
            skip_dates=skip_529,
        ),
        StrategySpec(
            name="pnl7_once_50",
            mode="metric_rule",
            metric_pnl_lte=-7.0,
            metric_reduce_pct=0.50,
        ),
        StrategySpec(
            name="pnl5_once_30",
            mode="metric_rule",
            metric_pnl_lte=-5.0,
            metric_reduce_pct=0.30,
        ),
        StrategySpec(
            name="dd5_once_50",
            mode="metric_rule",
            metric_dd_lte=-0.05,
            metric_reduce_pct=0.50,
        ),
        StrategySpec(
            name="B_light",
            yellow_reduce_pct=0.20,
            red_reduce_pct=0.40,
            max_yellow_actions=1,
            max_red_actions=1,
            warning_b_branch=True,
        ),
        StrategySpec(
            name="red_only_30",
            respond_yellow=False,
            red_reduce_pct=0.30,
            max_red_actions=1,
        ),
    ]

    for yellow, red in ((0.2, 0.4), (0.25, 0.5), (0.3, 0.5), (0.3, 0.6)):
        grid.append(
            StrategySpec(
                name=f"B_grid_y{int(yellow * 100)}_r{int(red * 100)}",
                yellow_reduce_pct=yellow,
                red_reduce_pct=red,
                max_yellow_actions=1,
                max_red_actions=1,
                warning_b_branch=True,
            )
        )

    return grid


def render_strategy_search_md(
    results: list[PortfolioSimResult],
    *,
    start: str,
    end: str,
    hold_pnl: float,
) -> str:
    ranked = sorted(results, key=lambda r: r.total_pnl, reverse=True)
    lines = [
        "# 5/29–6/10 持仓策略搜索",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 回放区间：**{_iso_from_compact(start)} ~ {_iso_from_compact(end)}**",
        f"- 一直持有基准：**{hold_pnl:,.0f} 元**",
        "- 假设：信号日按 replay 中 `pnl_pct` 推导价格减仓；不计佣金/滑点",
        "",
        "> 样本仅 9 个交易日、16 只标的，排名靠前也可能是过拟合，需扩窗验证。",
        "",
        "## 策略排行榜（按 6/10 合计盈亏）",
        "",
        "| 排名 | 策略 | 6/10 总盈亏 | vs 持有 | 过程最低市值合计 | vs 持有(过程) | 优/劣/接近 | beat 持有 |",
        "|------|------|------------|---------|------------------|---------------|-----------|-----------|",
    ]
    for idx, result in enumerate(ranked, start=1):
        beat = "是" if result.total_pnl > hold_pnl else "否"
        lines.append(
            f"| {idx} | {result.strategy.name} | **{result.total_pnl:,.0f}** | "
            f"{result.vs_hold:+,.0f} | {result.total_min_mv:,.0f} | "
            f"{result.total_min_mv - result.hold_total_min_mv:+,.0f} | "
            f"{result.win_count}/{result.lose_count}/{result.tie_count} | {beat} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_best_detail_md(
    top_results: list[PortfolioSimResult],
    holdings_meta: dict[str, dict[str, float | str]],
    name_map: dict[str, str],
    *,
    hold_result: PortfolioSimResult,
) -> str:
    hold_by_code = {s.code: s for s in hold_result.symbols}
    lines = [
        "# Top 策略逐标贡献",
        "",
    ]
    for result in top_results:
        lines.extend(
            [
                f"## {result.strategy.name}",
                "",
                f"- 合计盈亏：**{result.total_pnl:,.0f} 元**（vs 持有 {result.vs_hold:+,.0f}）",
                "",
                "| code | 中文名 | vs 持有 | 策略盈亏 | 持有盈亏 | 6/10 仓位 |",
                "|------|--------|---------|----------|----------|----------|",
            ]
        )
        rows = []
        for sym in result.symbols:
            diff = sym.pnl - sym.hold_pnl
            name = name_map.get(sym.code, str(holdings_meta[sym.code].get("name", sym.code)))
            rows.append((diff, sym.code, name, diff, sym.pnl, sym.hold_pnl, sym.final_position_pct))
        for _, code, name, diff, pnl, hold_pnl, pos in sorted(rows, key=lambda x: x[0]):
            lines.append(
                f"| {code} | {name} | {diff:+,.0f} | {pnl:,.0f} | {hold_pnl:,.0f} | {pos:.0f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def results_to_csv(results: list[PortfolioSimResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "strategy": result.strategy.name,
                "total_pnl": round(result.total_pnl, 2),
                "hold_pnl": round(result.total_hold_pnl, 2),
                "vs_hold": round(result.vs_hold, 2),
                "total_min_mv": round(result.total_min_mv, 2),
                "hold_total_min_mv": round(result.hold_total_min_mv, 2),
                "vs_hold_min_mv": round(result.total_min_mv - result.hold_total_min_mv, 2),
                "win_count": result.win_count,
                "lose_count": result.lose_count,
                "tie_count": result.tie_count,
                "beat_hold": result.total_pnl > result.total_hold_pnl,
            }
        )
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare replay reduce strategies vs hold")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--live-holdings", type=Path, required=True)
    parser.add_argument("--start", default="20260529")
    parser.add_argument("--end", default="20260610")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    replay_dir = args.replay_dir.expanduser().resolve()
    live_holdings = args.live_holdings.expanduser().resolve()
    output_dir = (args.output or replay_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    holdings = load_holdings_for_sim(live_holdings)
    codes = tuple(sorted(holdings.keys()))
    replay_frames = load_replay_frames(replay_dir, codes)
    name_map = load_holdings_name_map(live_holdings)

    strategies = build_strategy_grid()
    results: list[PortfolioSimResult] = []
    hold_result: PortfolioSimResult | None = None

    for spec in strategies:
        result = simulate_portfolio(holdings, replay_frames, spec)
        results.append(result)
        if spec.name == "hold":
            hold_result = result

    assert hold_result is not None
    hold_pnl = hold_result.total_pnl

    ranked = sorted(results, key=lambda r: r.total_pnl, reverse=True)
    top3 = [r for r in ranked if r.strategy.name != "hold"][:3]

    search_md = render_strategy_search_md(results, start=args.start, end=args.end, hold_pnl=hold_pnl)
    (output_dir / "STRATEGY_SEARCH.md").write_text(search_md, encoding="utf-8")

    detail_md = render_best_detail_md(top3, holdings, name_map, hold_result=hold_result)
    (output_dir / "strategy_best_detail.md").write_text(detail_md, encoding="utf-8")

    csv_path = output_dir / "strategy_results.csv"
    results_to_csv(results).to_csv(csv_path, index=False)

    best = ranked[0]
    print(f"[OK] strategies={len(results)} -> {output_dir}")
    print(f"     hold={hold_pnl:,.0f} best={best.strategy.name} pnl={best.total_pnl:,.0f} vs_hold={best.vs_hold:+,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
