---
name: ask-etf
description: Answers "can I buy this ETF" for one fund in the etf-daily pool from the latest rules7 output, in four fixed parts (conclusion, veto chain, flip conditions, confidence). Use when the user says "问 SH520870", "SH520870 能不能买", "XX ETF 可以买吗", or asks about buying, holding or selling a pooled ETF.
---

# ask-etf

Read-only research answer. It never places orders and is not investment advice. Do not run the daily pipeline here; only read what it already wrote.

## Steps

1. Run the facts script (stdlib only, any `python3`):
   `python3 .cursor/skills/ask-etf/scripts/ask_etf.py <CODE>`
   `CODE` can be `SH520870`, `520870`, or a unique name fragment like `巴西`. It finds the newest `*_from_listing` directory under `PLOTLY_ROOT` (env or `config.env`).
2. If it says the directory is not today, or the 日报 says `读数: 盘中`, say so at the top of the answer. Offer the daily commands from `README_zh.md` only if the user wants fresher data.
3. If the code is not in the pool, say that and stop. Do not guess from other data.
4. Answer in the four parts below. Use only numbers the script printed. Never invent a number.

## Output format

```
**<名称> <代码>（收盘价 X，<盘中/收盘>读数，<日期>）**

1. 结论：买 / 不买，观望 / 持有 / 卖。一句话。
2. 否决链：哪些买入规则亮了，谁否决了，否决证据有多硬。
3. 翻转条件：什么情况答案会变，写具体价格或天数。
4. 可信度：高 / 中 / 低，加 2-3 条依据。

数据外：<仓库里没有、需要自己核对的东西>
```

## How to fill each part

**Conclusion** from `今日状态` and `今日动作`:
- 空仓 + 观望 → 不买。
- `买入信号` 是，且 `推荐买入` 有值 → 可以按规则次日收盘买；写出止损规则（收盘 ≤ 买入价 × 0.90 次日止损）和 `回落幅度`。
- `买入信号` 是但日报说"单独出现"（全池买点占比 < 5%）→ 不自动买，降到观察。
- 持仓 + 持有 → 持有，不加仓。持仓 + 卖 → 次日收盘卖，写 `卖出原因`。
- `规则不适用` → 这套规则对它没有结论，只能说信号仅供参考。

**Veto chain**: list which of `近下轨上跳沿`、`辅助买像`、`aux_edge买点`、`B1` are 是, then the `买点状态`:
- `快形态挡住`: 买点真的出现了，被快形态（近 60 日有 ≥20% 的 40 日涨幅）否决。
- `快形态中，未到买点`: 在快形态里，但今天没有买点。不是被挡住的买点，别说成"买点被挡"。
- `今天已是买点`、`贴轨即可`、`还差这个价`、`辅助买像未齐`、`已在带内，明天同价不算`: 照字面解释，并给 `买点价` 和 `距图9下轨`。
- 买规则里没有 B1 的类型，`B1` 为是也不算买点。以 `买规则` 为准。

Evidence strength of the fast-shape veto: `docs/rules7.md` says "快形态时不买" improved only 3 of the 7 study ETFs, so call it a conservative rule, not a proven one. The doc lists 快形态不买 for only 黄金、通信、酒、恒生医药 (恒生医药 has no profitable rule). For any other ETF the veto is carried over from the common rule, so say so.

**Flip conditions**:
- 快形态: `快形态剩余K线` bars from now, assuming no new 20% run.
- 价格: `买点价` (P95 下轨 ×1.01) and `EWMA买点价`, with the distance from the close. EWMA-only buys are observation grade, win rate about 41%, so do not present them as a buy basis.
- `辅助买像未齐`: MA5 < MA20、gap5 ≤ 0、gap20 ≤ 0 and 收盘 < MA20 must all hold.
- 已持仓: 说明 `回落止损价` and `距止损`.

**Confidence** (take the worst one that applies):
- 低: `类型` 为"历史不足", or `bars` < 250, or `skipped_train` 为 True, or `n_closed` < 10.
- 中: 样本够，但标的不在 rules7 研究的 7 只里，或 edge 为负。
- 高: 只在 7 只研究标的之一，样本够，edge 为正。
Always give `compound`、`bh`、`n_closed`、`win` from the backtest, and mention the most recent trades from the script when they explain the move (for example a recent big loss that triggered the fast pattern).

**数据外** always lists what the repo does not have. For QDII and overseas ETFs: 溢价率、额度和限购公告、汇率、标的市场。 For domestic ETFs: 申赎和流动性。

## Style

- Chinese, short. State the position and what would change it. No hedging filler.
- Say plainly when evidence is weak. A veto with weak evidence is "保守", not "数据证明".
- Do not edit repo files or regenerate outputs unless the user asks.

## Example

User: 问 SH520870

> **巴西ETF易方达 SH520870（收盘价 0.9670，盘中读数，2026-11-06）**
>
> 1. 结论：不买，观望。
> 2. 否决链：近下轨上跳沿、辅助买像、aux_edge、B1 都亮了，现价在图9下轨下方 0.4%，买点是真的；近 60 日 40 日涨幅 27.3% 触发快形态，买点被挡。这个否决证据弱：规则只在 7 只研究标的里的 3 只有效，巴西 ETF 不在其中，所以是保守，不是被证明过。
> 3. 翻转条件：约 50 根 K 线后快形态解除（假设无新的 20% 涨幅）；或价格先跌到 EWMA 下轨 0.8772（距现价 10.2%），但 EWMA 买点只作观察。
> 4. 可信度：低。类型"历史不足"，238 根 K 线；模型训练 91 根且 `skipped_train`；4 笔已平仓胜率 50%，复利 -17.4% 对买入持有 -6.2%。10 月 9 日至 27 日那笔 -20.6%，就是触发快形态的冲高回落。
>
> 数据外：QDII 溢价率、额度或限购公告、雷亚尔汇率、巴西股市。
