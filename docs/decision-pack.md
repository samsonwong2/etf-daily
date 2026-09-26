# decision_pack

T 日收盘后生成 **Decision Pack**：档位、目标暴露、T+1 同比例减仓建议、对账与恢复状态。  
业务规则见 `DESIGN.md` 与 `config/default.yaml`。

## 快速开始

本模块依赖 **qlib** 拉取 ETF 收盘价（实盘 CSV 的 `price` 为成本时）。请使用 conda `py312` 环境：

```bash
PY=~/etf-daily-output/python/envs/py312/bin/python
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
```
# 标准：pipeline 已跑完当日
$PY -m decision_pack generate \
  --date 2026-06-05 \
  --run-dir temp/trend_sleeve_ablation_20260605/2025_regime_switch_ewma_shrink_no_sleeve

# 实盘 CSV：code,name,shares,price — price=持仓成本，市值由 qlib 收盘价计算
$PY -m decision_pack generate \
  --date 2026-06-05 \
  --run-dir <run_dir> \
  --live-holdings ~/etf-daily-output/temp/live_holdings/20260605.csv
```

### 实盘 CSV 列说明

| 列 | 含义 |
|----|------|
| `code` | 标的代码，如 `SH561560` |
| `name` | 可选，名称 |
| `shares` | 持有份额 |
| `price` | **默认当作持仓成本**；市值 = `shares × qlib 收盘价` |
| `cost` / `cost_price` | 显式成本列（与 `price` 二选一或并存） |
| `market_price` / `收盘价` | 若提供则直接用于算市值，不再读 qlib |
| `market_value` / `市值` | 若提供则直接用，忽略 price |

默认配置见 `config/default.yaml` → `live_holdings.price_column_semantics: cost`。  
若你的 `price` 列是**现价**而非成本，改为 `market`。

### v0 展示模式（默认）

`config/default.yaml` 中 `orders.mode: display_only` 时：

- 输出 `holdings_trend_board.csv`（权益 ETF 趋势指标，12 列）
- 报告含「持仓趋势板（v0 · 人工决策）」一节
- **不生成自动 SELL**；`orders_T+1.csv` 仅为 HOLD 占位

切回同比例减仓：`orders.mode: proportional_legacy`。

```bash
# 回放测试：不写 recovery state
$PY -m decision_pack generate --date 2026-06-05 --run-dir <run_dir> --dry-run

# 查看 recovery 状态
$PY -m decision_pack status
```

默认输出目录：`runtime/decision_packs/YYYYMMDD/`。

## 产物

| 文件 | 说明 |
|------|------|
| `DECISION_PACK_REPORT.md` | 人类可读主报告（display_only 为机构化 L1–L3 结构） |
| `action_plan.json` | 优先级行动清单、T+1 执行行、PM 摘要（`display_only` 模式） |
| `tier_decision.json` | 档位决策 |
| `orders_T+1.csv` | T+1 清单（展示模式为 HOLD 占位；legacy 为同比例减仓） |
| `holdings_trend_board.csv` | v0 持仓趋势板（`display_only` 模式） |
| `early_warning.csv` | 黄/红灯预警（`display_only` 模式） |
| `cluster_trend_board.csv` | 待选池趋势板（`display_only` 模式） |
| `portfolio_snapshot.csv` | 持仓快照 |
| `market_snapshot.json` | 基准与组合指标 |
| `reconcile_report.md` | 实盘 vs pipeline（有 `--live-holdings` 时） |
| `system_vs_manual.json` | L3 exposure_scale 对比 |
| `recovery_state.json` | 当日恢复状态快照 |
| `inputs_manifest.json` | 输入文件审计 |

跨日状态持久化：`decision_pack/state/recovery_state.json`。

## 短周期板与明日 digest（旁路）

GARCH / HMM 主审计板（h5/h10/h20，列不变）信息较密。若只需 **待选池明日涨跌区间 + 校准可信度**，用 digest 旁路：

```bash
# 每日：主审计板（列不变）
$PY src/etf_daily/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260715
$PY src/etf_daily/scripts/generate_hmm_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260715

# 每日：明日候选一张表（H=1）
$PY src/etf_daily/scripts/generate_tomorrow_digest.py \
  --pack-dir ~/temp/decision_packs/20260715
# -> tomorrow_candidates_digest.csv + tomorrow_topk_picks.csv + TOMORROW_DIGEST.md
# 可选：--topk-k 5 或 --no-topk

# 每日：左侧波动候选（H=5 · 压缩 × spike × 触轨↓）
$PY src/etf_daily/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260716 --horizon 5

# 每日：3-state 马尔科夫跃迁板（独立于 GARCH/HMM 短周期板）
$PY src/etf_daily/scripts/generate_regime_transition_board.py \
  --pack-dir ~/temp/decision_packs/20260716
# -> cluster_mapping_selected_regime_transition.csv

$PY src/etf_daily/scripts/generate_left_side_digest.py \
  --pack-dir ~/temp/decision_packs/20260716
# -> left_side_candidates_digest.csv + left_side_topk_picks.csv + LEFT_SIDE_DIGEST.md
```

**左侧 Top-K**：`left_side_topk_picks.csv` 对**全池标的**按 `compression_x_spike_x_down`（**(1−vol5分位) × p_vol_spike × 触轨↓ × 综合可信度**）排序；可选 `transition_x_spike_x_down`（**transition_score × p_vol_spike × 触轨↓ × transition_cred**，跃迁 CSV 存在时跳过内联 2-state HMM；`transition_cred = clip(score)×校准`）。`left_side_candidates_digest.csv` 在 transition 板存在时新增 `transition_score` / `regime_now` / `跃迁方向` 列。`equal_weight_pct` 仅标注满足门槛的前 K（默认 `vol5_pct_120d ≤ 35%`、`触轨↓ ≥ 15%`、排除 `live_price_gap`/`recent_jump`；`transition_x_spike_x_down` 规则不卡 vol5 上限）。GARCH 看板 H=5 CSV 新增 `vol5_pct_120d`、`p_vol_spike_h5` 列。配置见 `default.yaml` → `left_side_digest` / `garch_short_horizon.spike_mult`。

```bash
$PY src/etf_daily/regime/backtest_regime_transition_signals.py \
  --pack-dir ~/temp/decision_packs/20260716 --horizon 5 --top-pct 0.10
```

```bash
$PY src/etf_daily/scripts/backtest_left_side_vol_signals.py \
  --pack-dir ~/temp/decision_packs/20260716 --horizon 5 --k 8
```

**Top-K 等权买入**：`tomorrow_topk_picks.csv` 按可配置 **选股分** 对**全部标的**降序排名；`等权%` 列仅标注前 K 只（默认 K=8）的等权参考。默认 `rank_rule: upside_x_cred`（**upside_skew × 综合可信度**，偏「涨得多」），并过滤 `garch_方向偏 > 0`。**不是预测明日收益率**；高波动 avoid 腿可能霸榜，请结合 `barbell_leg` 与 `garch_跌5%`/`garch_涨95%`/`upside_skew_pct` 自行筛选。配置见 `default.yaml` → `tomorrow_digest.topk`。

### 可买排名（推荐读法 + 旁路脚本）

包内趋势 / 风险 / 明日信号是分开的。**不要直接拿 `tomorrow_topk_picks.csv` 当前排名当买入清单**。建议读序：

1. **趋势** — `cluster_trend_board.csv`：`ma_stack` / `risk_flags`（红灯：`bear_stack`、`deep_below_ma20`、`below_ma20_3d+`）
2. **风险腿** — `cluster_mapping_selected_pool_risk_252.csv`：`leg`（`avoid` 不做多）、`状态=破位`
3. **急跌/跳空** — `garch_jump_diagnostics.csv`：`recent_jump` / `live_price_gap`
4. **明日分（过门禁后再排）** — `tomorrow_topk_picks.pick_score`

一键合并（硬过滤后按明日 `pick_score` 排序）：

```bash
# 依赖：已生成 tomorrow_topk_picks.csv + cluster_trend_board + pool_risk + jump 诊断
$PY src/etf_daily/scripts/generate_buyable_rank.py \
  --pack-dir ~/temp/decision_packs/20260717
# -> buyable_topk_picks.csv + BUYABLE_RANK.md

# 按回测最优规则（gate / rank_rule / k / min_cred）生成当日榜
$PY src/etf_daily/scripts/generate_buyable_rank.py \
  --pack-dir ~/temp/decision_packs/20260717 \
  --from-best-strategy ~/temp/decision_packs/20260717/buyable_rank_backtest/best_strategy.json
# -> buyable_topk_picks_best.csv + BUYABLE_RANK_BEST.md

# 或显式指定（与上等价示例）
$PY src/etf_daily/scripts/generate_buyable_rank.py \
  --pack-dir ~/temp/decision_packs/20260717 \
  --gate core_risk --rank-rule dir_x_cred --topk-k 1 --min-credibility 0
```

配置见 `default.yaml` → `buyable_rank`。默认剔除 avoid / 破位 / 上述趋势与 jump 旗标；`equal_weight_pct` 只标注过门禁的前 K。CSV 仍保留被拒行（`pass_filters=false` + `reject_reason`）便于审计。`--rank-rule` 会用 `direction_bias` / `combined_cred` / `upside_skew` **重算** `pick_score`（不重跑 GARCH）。

### Buyable 逐日回测选优

因果重建每日信号（不复用期末 pack CSV），成交口径：**信号日收盘后生成排名 → 次日开盘买入、当日收盘卖出**，双边各 10 bp。按验证集（6 月）Sharpe 选最优；7 月 holdout 只作报告。

```bash
$PY src/etf_daily/scripts/backtest_buyable_rank.py \
  --pack-dir ~/temp/decision_packs/20260717 \
  --eval-start 2026-04-01 \
  --eval-end 2026-07-17 \
  --cost-bps-per-side 10
# -> buyable_rank_backtest/{strategy_summary,daily_returns,daily_picks,best_strategy.json,BUYABLE_RANK_BACKTEST.md}
```

比较网格：`rank_rule` × `K∈{1,2,3,5,8}` × 门禁（`strict` / `no_path_gate` / `core_risk` / `positive_score`）× `min_credibility∈{0,0.003,0.005}`。

**局限**：固定当前 `cluster_mapping_selected` 候选池（幸存者偏差）；无历史 H=1 校准缓存时用池中位可信度；短样本 holdout **不能**当作生产保证。

| `rank_rule` | 公式 |
|-------------|------|
| `dir_x_cred` | garch_方向偏 × 综合可信度（旧默认） |
| `upside_x_cred` | upside_skew × 综合可信度（`upside_skew = 涨95% − \|跌5%\|`，对数收益） |
| `composite` | max(0, 方向偏) × 综合可信度 × upside_skew |

H=1 三条规则 walk-forward 对比（GARCH-only，不含 buyable 门禁）：

```bash
$PY src/etf_daily/scripts/backtest_garch_topk_selection.py \
  --pack-dir ~/temp/decision_packs/20260715 --horizon 1 --k 8
```

**每周**刷新 walk-forward 校准缓存（供可信度缩放）：

```bash
mkdir -p ~/temp/decision_packs/calibration
$PY src/etf_daily/scripts/validate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260715 --horizon 1 \
  --output-cache ~/temp/decision_packs/calibration/garch_h1_per_symbol.csv
$PY src/etf_daily/scripts/validate_hmm_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260715 --horizon 1 \
  --output-cache ~/temp/decision_packs/calibration/hmm_h1_per_symbol.csv
```

| 文件 | 说明 |
|------|------|
| `tomorrow_candidates_digest.csv` | 候选池；GARCH/HMM 明日 q05~q95 区间与综合可信度 |
| `tomorrow_topk_picks.csv` | 全部标的选股排名（方向偏×可信度）；等权% 仅标注前 K；**未经趋势/avoid 门禁** |
| `buyable_topk_picks.csv` | 可买排名：硬过滤后再按 pick_score；含 `pass_filters` / `reject_reason` |
| `BUYABLE_RANK.md` | 可买 Top 摘要 + 被拒样例 |
| `left_side_topk_picks.csv` | 左侧波动全池排名（压缩×spike×触轨↓）；等权% 仅标注过门槛的前 K |
| `TOMORROW_DIGEST.md` | Top 10 可信度摘要 + Top-K 买入段 |
| `calibration/garch_h1_per_symbol.csv` | 周期性校准（Brier、分位覆盖率） |
| `calibration/hmm_h1_per_symbol.csv` | 同上 |

## 测试

```bash
$PY -m pytest decision_pack/tests -q
```

## 模块

- `src/tier.py` — 阶梯表 + 情绪降级
- `src/market.py` — qlib 基准 + daily_nav 回撤
- `src/portfolio.py` — 实盘 / pipeline 持仓
- `src/orders.py` — 同比例卖单 / v0 展示占位
- `src/symbol_trend_board.py` — v0 持仓趋势板
- `src/early_warning.py` — 黄/红灯预警
- `src/action_plan.py` — 机构化优先级行动清单（L2/L3/T+1）
- `src/indicators.py` — 共享因果指标（MA、vol 分位等）
- `src/reconcile.py` — 对账告警
- `src/recovery.py` — T+N 恢复状态机
- `src/tomorrow_digest.py` — 明日 H=1 digest / topk
- `src/buyable_rank.py` — 可买排名（趋势/肥尾/急跌门禁 + pick_score）
- `src/buyable_rank_backtest.py` — buyable 逐日回测与选优
- `src/pack.py` — 编排入口











```
cd ~/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python
# 1) 先生成 decision pack（若还没有）
$PY -m decision_pack generate \
  --date 2026-06-12 \
  --run-dir ~/etf-daily-output/temp/trend_sleeve_ablation_20260612/2025_regime_switch_ewma_shrink_no_sleeve \
  --live-holdings ~/etf-daily-output/temp/live_holdings/20260612.csv \
  --output ~/temp/decision_packs/20260612
# 2) 再生成操作指引 MD
$PY src/etf_daily/scripts/generate_ops_guide.py \
  --pack-dir ~/temp/decision_packs/20260612 \
  --notes workspace/notes/20260612.md \
  --output ~/temp/decision_packs/20260612/OPS_GUIDE.md
```

