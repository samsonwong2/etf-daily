



# fund_pool_builder

> 工程化、模块化重构自旧 `2_filter/`（已归档），负责生成 canonical 的
> `cluster_mapping_selected.txt` 待选池文件，供上层 pipeline 的 `--cluster-mapping-path` 参数消费。
>
> 上游直接以手工维护的 `~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/fund_list.csv` 为输入，
> **不再从深证/上证 ETF 快照 Excel 自动合并**（这类快照更新不及时，易滞后）。
> 若 `fund_list.csv` 需增减 ETF，直接编辑该 CSV 即可。

---

## 1. 目录结构

```
fund_pool_builder/
├── 生成cluster_mapping_selected_最短命令清单.py    # 选池 + canonical txt（thin launcher）
├── 每日审查cluster_mapping_最短命令清单.py          # 手工池每日审查（thin launcher）
├── shared_filter_config.json                       # qlib provider / 测试期 / fund_list 路径
├── README.md
│
└── pool_builder/                                   # 重构后的 Python 包
    ├── __init__.py
    ├── cli.py                  # 顶层 orchestrator：select → filter_txt [→ eval/apply]
    ├── constants.py            # 输出路径 + 白/黑名单 + 聚类阈值
    ├── config.py               # shared_filter_config.json 加载 + metadata json 写入
    ├── code_utils.py           # 基金代码归一化/别名/手工白名单加载
    ├── data_loading.py         # qlib init + fund_list CSV → code_name_map + close/volume
    ├── clustering.py           # 窗口/重叠/多代表池/trim/ward 聚类核心
    ├── dendrogram.py           # 选中代表树状图（自动加载 CJK 字体）
    ├── export.py               # cluster_mapping*.csv 导出器
    ├── selector.py             # 高层 select_pool()（对应老 0.2 的 main）
    ├── filter_txt.py           # CSV → canonical all.txt 格式 txt（对应老 0_all_filter_txt.py）
    ├── audit.py                # 缺失赢家审计（对应老 0.5_audit_missed_winners.py）
    ├── compare.py              # 双池差异 Markdown 报告（对应老 compare_cluster_pool.py）
    └── daily_review.py         # 手工池每日审查（漏族 / 代表落后 / vs 机器池 diff）
```

---

## 2. 数据流概览

```
~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/fund_list.csv    （手工维护；code / name / inception_date / ...）
         │
         ▼  pool_builder.selector.select_pool()
   ├─ cluster_mapping.csv            （身份/簇字段 + fund_list 业务列）
   ├─ cluster_mapping_selected.csv   （mainline 规则下入选的 ETF）
   ├─ cluster_mapping_selected_metadata.json
   └─ dendrogram_selected_reps.png / .svg
         │
         ▼  pool_builder.filter_txt.filter_all_txt()
  ~/etf-daily-output/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt
         │
         ▼  上游 pipeline 消费
   python run.py pipeline
     --cluster-mapping-path .../cluster_mapping_selected.txt
```

---

## 3. 最常用运行命令

```bash
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/fund_pool_builder
```

### 3.1 默认一键生成 canonical txt（规则输出版）

```bash
python 生成cluster_mapping_selected_最短命令清单.py
```

相当于顺序执行：

1. `pool_builder.selector.select_pool()` 按 `shared_filter_config.json` 指定的
   `test_period` 和 `fund_list.csv` 跑聚类 + 选代表池
2. `pool_builder.filter_txt.filter_all_txt()` 用上一步得到的
   `cluster_mapping_selected.csv` 过滤 qlib 的 `all.txt`，写入 canonical
   `cluster_mapping_selected.txt`

### 3.2 只做聚类/选池，不写 canonical txt

```bash
python 生成cluster_mapping_selected_最短命令清单.py --skip-filter-txt
```

### 3.3 只更新 canonical txt（复用已有 CSV）

```bash
python 生成cluster_mapping_selected_最短命令清单.py --skip-select
```

### 3.4 评估后输出（one-shot forward 评估自动择池）

本包不再默认依赖旧项目的 `cluster_pool_rolling_eval/`，而是把它作为可选外部
命令挂钩（保持解耦）。如需该链路，把评估/应用脚本路径通过参数传入：

```bash
python 生成cluster_mapping_selected_最短命令清单.py \
  --production-eval-cmd "python /path/to/cluster_pool_rolling_eval/production_cluster_pool_eval.py \
                           --eval-start 2025-04-01 --eval-end 2026-03-31" \
  --production-apply-cmd "python /path/to/cluster_pool_rolling_eval/production_cluster_pool_apply.py"
```

### 3.5 缺失赢家审计（0.5 的重构版）

```bash
python -m pool_builder.audit \
  --mapping_csv ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/etf_filter_checks/cluster_mapping_etf.csv \
  --future_end 2026-04-22
```

### 3.6 对比两次 canonical txt

```bash
python -m pool_builder.compare \
  --old 20250101_20251231 \
  --new ~/etf-daily-output/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt \
  --cluster-map ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping.csv
```

---

## 4. 关键参数（`pool_builder.cli`）

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--shared-config` | `shared_filter_config.json` 路径 | 本包下的同名 JSON |
| `--inception-cutoff` | 剔除成立日晚于此日期的新基金 | `test_period[1]` |
| `--skip-select` | 跳过聚类/选池阶段 | False |
| `--skip-filter-txt` | 跳过 canonical txt 阶段 | False |
| `--auto-selected-suffix` | 机器池写入 `cluster_mapping_selected_{suffix}.csv`，不覆盖手工 CSV | 无 |
| `--csv-path` | filter-txt 读取的 CSV | `cluster_mapping_selected.csv` |
| `--all-path` | qlib `all.txt` 路径 | `~/etf-daily-output/data/qlib_data/.../all.txt` |
| `--output-path` | 输出 canonical txt 路径 | `~/etf-daily-output/data/qlib_data/.../cluster_mapping_selected.txt` |
| `--production-eval-cmd` | 可选：一键评估命令 | 无 |
| `--production-apply-cmd` | 可选：一键 apply 命令 | 无 |

---

## 5. 关键常量与白/黑名单

所有常量集中在 [pool_builder/constants.py](pool_builder/constants.py) 中：

- **路径**：`OUT_DIR`, `CSV_OUT`, `CSV_SELECTED_OUT`, `METADATA_OUT`, `IMG_OUT`,
  `SVG_OUT`, `DEFAULT_ALL_TXT`, `DEFAULT_CANONICAL_TXT`
- **聚类阈值**：`DIST_T=0.33`, `PER_CLUSTER=1`, `TARGET_SELECTED_COUNT=80`,
  `CLUSTER_LOOKBACK_DAYS=252`, `MIN_CLUSTER_HISTORY_DAYS=120`,
  `MIN_PAIR_OVERLAP_DAYS=120`, `MIN_SELECTION_HISTORY_DAYS=180`,
  `FINAL_MAX_ABS_CORR_THRESHOLDS=(0.80, 0.85, 0.90)`
- **规则**：`RULE="multi_rep_equal_mainline"`（`BENCHMARK_RULE` / exp mapping 已停用）,
  `MULTI_REP_EXP_BASE=0.65`
- **白名单**：`ALWAYS_KEEP_CODES`（含 `SH518880` 黄金及固收锚点等）
- **黑名单**：`ALWAYS_DROP_CODES` + `ALWAYS_DROP_CLUSTER_CODES`（按种子代码剔除整簇）
- **覆盖关系**：`ALWAYS_KEEP_CODES` / 手工白名单 **覆盖** `EXCLUDE_TYPES` 与成立日 cutoff（可强制保留 `指数型-固收`）；`ALWAYS_DROP_*` 仍优先于白名单

另有手工白名单文件 `MANUAL_KEEP_CODES_FILE`
（`~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/etf_filter_checks/etf_manual_keep_whitelist.csv`），
存在即自动并入 `ALWAYS_KEEP_CODES`。

---

## 6. 审核生成的 `cluster_mapping_selected.csv` 质量

**目的**：用同一次 select 产出的**全量** `cluster_mapping.csv`（含同族未选兄弟），检查**已生成**的 `cluster_mapping_selected.csv` 里每只代表是否仍是簇内近期最强、是否有漏族/强但未选。

| 文件 | 路径 |
| --- | --- |
| 全量 mapping（同族对照） | `~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping.csv` |
| 待审核的入选池 | `~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping_selected.csv` |

```bash
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/fund_pool_builder

# 1) 生成 mapping + selected（不写 canonical txt 可加快）
python 生成cluster_mapping_selected_最短命令清单.py --skip-filter-txt

# 2) 质量审查（默认即上述两个 CSV；不做双 selected 对比）
python 每日审查cluster_mapping_最短命令清单.py --future-end $(date +%Y-%m-%d) --skip-diff
```

审查输出：`~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/etf_filter_checks/daily_{YYYYMMDD}/`

| 输出 | 含义 |
| --- | --- |
| `rep_lag.csv` | 已入选标的在簇内 5/20 日收益排名落后（核心） |
| `missed_clusters.csv` | mapping 有簇、selected 无代表（含刻意整族剔除说明） |
| `top_regret_same_cluster_etf.csv` | 同族更强但未入选（audit，`--skip-audit` 可关） |
| `daily_cluster_review_{date}.md` | 一页摘要 |

**检查表**：`rep_lag` 无高 regret；`missed_clusters` 仅预期整族剔除；需要时再 `filter_txt` 写 canonical txt。

可选：若你另有一份手工定稿 CSV，把 `--selected-csv` 指过去即可；`--auto-selected-suffix` / dated CSV 仅用于「保留历史机器快照」，与质量审查无关。

**同日 μ 层健康（pipeline temp-dir，与选池审查互补）**：

```bash
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean && \
~/etf-daily-output/python/envs/py312/bin/python 每日μ健康检查_最短命令清单.py \
  --temp-dir ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/trend_sleeve_ablation_20260521/2025_regime_switch_ewma_shrink_no_sleeve
```

说明见 [`workspace/history/daily_layer_quality_audit_runbook_20260517.md`](../workspace/history/daily_layer_quality_audit_runbook_20260517.md) §5.1（`mu_monitor` + `daily_layer_quality` 各最近 5 行；需 production `--audit-level full` 才有 layer_quality）。

---

## 7. 模块方式调用

```python
from pool_builder.selector import select_pool
from pool_builder.filter_txt import filter_all_txt

# 1) 生成 mapping CSV 和 metadata
select_pool(shared_config_path="/path/to/shared_filter_config.json")

# 2) 写 canonical txt
filter_all_txt(
  csv_path="~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping_selected.csv",
  all_path="~/etf-daily-output/data/qlib_data/all_fund_data/instruments/all.txt",
)
```

---

## 8. 与旧 `2_filter/` 的对应关系

| 旧脚本 | 新位置 |
| --- | --- |
| `2_filter/0.1run_all_fund_herc.py` | [pool_builder/cli.py](pool_builder/cli.py) |
| `2_filter/0.2_cluster_select_from_dendrogram.py` | [pool_builder/selector.py](pool_builder/selector.py) + [clustering.py](pool_builder/clustering.py) + [data_loading.py](pool_builder/data_loading.py) + [code_utils.py](pool_builder/code_utils.py) + [export.py](pool_builder/export.py) + [dendrogram.py](pool_builder/dendrogram.py) + [constants.py](pool_builder/constants.py) |
| `2_filter/0_all_filter_txt.py` | [pool_builder/filter_txt.py](pool_builder/filter_txt.py) |
| `2_filter/0.5_audit_missed_winners.py` | [pool_builder/audit.py](pool_builder/audit.py) |
| `2_filter/0.5b_count_miss_types.py` | 使用 `pool_builder.audit` 的函数手写脚本即可（逻辑已拆出） |
| `2_filter/compare_cluster_pool.py` | [pool_builder/compare.py](pool_builder/compare.py) |
| `2_filter/shared_filter_config.json` | [shared_filter_config.json](shared_filter_config.json) |
| `2_filter/first_filer/*.py` | 已移除（不再从 Excel 合并；直接以 `fund_list.csv` 为输入） |

所有聚类/选池的数值阈值、白黑名单、规则字符串均原样保留，因此新包的默认输出
应与旧 `2_filter` 一致。

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/fund_pool_builder
# 生成全量 mapping + selected
python 生成cluster_mapping_selected_最短命令清单.py --skip-filter-txt
# 审 selected 质量（对全量 mapping 做同族对照）
python 每日审查cluster_mapping_最短命令清单.py \
  --future-end $(date +%Y-%m-%d) \
  --skip-diff
```

| 文件                              | 回答的问题                                                   |
| :-------------------------------- | :----------------------------------------------------------- |
| `rep_lag.csv`                     | 池里某只入选，同簇里是否有人近 5/20 日更强？差多少？是否可能因流动性保留现代表？ |
| `missed_clusters.csv`             | 全量 mapping 里有簇，但 selected 里一只都没有（整族黑名单会标原因） |
| `top_regret_same_cluster_etf.csv` | 同族里强、却没进 selected 的标的（audit 扩展）               |
| `daily_cluster_review_*.md`       | 上面几项的摘要                                               |



```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/fund_pool_builder
python 每日审查cluster_mapping_最短命令清单.py \
  --future-end 2026-07-31 \
  --selected-csv ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping_selected.csv \
  --mapping-csv ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping.csv
```

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean && \
~/etf-daily-output/python/envs/py312/bin/python workspace/scripts/generate_daily_mu_position_report.py \
  --base-dir ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/trend_sleeve_ablation_20260731/2025_regime_switch_ewma_shrink_no_sleeve \
  --date 20260731
```

```
cd ~/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python
$PY -m decision_pack generate \
  --date 2026-07-23 \
  --run-dir ~/etf-daily-output/temp/trend_sleeve_ablation_20260723/2025_regime_switch_ewma_shrink_no_sleeve \
  --live-holdings ~/etf-daily-output/temp/live_holdings/20260717.csv \
  --output ~/temp/decision_packs/20260723
```

```

PY=~/etf-daily-output/python/envs/py312/bin/python
$PY decision_pack/scripts/replay_symbol_warnings.py \
  --symbol SH561560 \
  --live-holdings ~/etf-daily-output/temp/live_holdings/20260610.csv
```

上面是单个标的的预警

下面是所有标的的预警

```
PY=~/etf-daily-output/python/envs/py312/bin/python
$PY decision_pack/scripts/replay_symbol_warnings.py \
  --all-holdings \
  --live-holdings ~/etf-daily-output/temp/live_holdings/20260610.csv \
  --temp-root ~/etf-daily-output/temp \
  --start 20260529 \
  --end 20260610 \
  --output ~/temp/decision_packs_replay/all_holdings
```

操作指引



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
$PY decision_pack/scripts/generate_ops_guide.py \
  --pack-dir ~/temp/decision_packs/20260612 \
  --notes workspace/notes/20260612.md \
  --output ~/temp/decision_packs/20260612/OPS_GUIDE.md
```

层级聚类脚本

```
conda activate py312
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
python workspace/scripts/generate_hrp_dendrogram_html.py \
  --asof-date 2026-08-1 \
  --holdings-csv ~/etf-daily-output/temp/live_holdings/20260717.csv \
  --output ~/etf-daily-output/temp/decision_packs/20260811/hrp_dendrogram_20260811.html
```

fat_tails风险模型

```
PY=~/etf-daily-output/python/envs/py312/bin/python
cd ~/gitee/skfolio_csi300/etf_strategy_clean
$PY decision_pack/scripts/generate_fat_tail_risk_report.py \
  --pack-dir ~/temp/decision_packs/20260723
```

```
PY=~/etf-daily-output/python/envs/py312/bin/python
cd ~/gitee/skfolio_csi300/etf_strategy_clean
$PY decision_pack/scripts/generate_fat_tail_risk_report.py \
  --pack-dir ~/temp/decision_packs/20260721 \
  --lookback-days 252
```

MLFAM

```
PY=~/etf-daily-output/python/envs/py312/bin/python
$PY decision_pack/scripts/generate_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260721
```

`garch_short_horizon` 块

```
PY=~/etf-daily-output/python/envs/py312/bin/python
cd ~/gitee/skfolio_csi300/etf_strategy_clean

# 默认 h5 + h10 + h20
$PY decision_pack/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260721

# 仅 h5
$PY decision_pack/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260708 --horizon 5
```

HMM块

```
PY=~/etf-daily-output/python/envs/py312/bin/python
# GARCH（CSV 列不变；多写诊断旁路）
$PY decision_pack/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260721 
# HMM（独立）
$PY decision_pack/scripts/generate_hmm_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260721
  
# 明日一张表
$PY decision_pack/scripts/generate_tomorrow_digest.py --pack-dir ~/temp/decision_packs/20260721 

$PY decision_pack/scripts/generate_buyable_rank.py \
  --pack-dir ~/temp/decision_packs/20260721
```

生成每日涨跌幅分布预测

```
$PY decision_pack/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260714 --with-price-targets
```

明日一张表

```
# 生成明日排名（默认 upside_x_cred）
$PY decision_pack/scripts/generate_tomorrow_digest.py \
  --pack-dir ~/temp/decision_packs/20260715
```

```
$PY decision_pack/scripts/generate_buyable_rank.py \
  --pack-dir ~/temp/decision_packs/20260709 \
  --gate core_risk --rank-rule dir_x_cred --topk-k 1 --min-credibility 0
```



## 每周校准（提升可信度精度）

```
mkdir -p ~/temp/decision_packs/calibration
$PY decision_pack/scripts/validate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260715 --horizon 1 \
  --output-cache ~/temp/decision_packs/calibration/garch_h1_per_symbol.csv
$PY decision_pack/scripts/validate_hmm_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260715 --horizon 1 \
  --output-cache ~/temp/decision_packs/calibration/hmm_h1_per_symbol.csv
```



*# 左侧 digest*

```
$PY decision_pack/scripts/generate_garch_short_horizon_board.py \
  --pack-dir ~/temp/decision_packs/20260716 --horizon 5

$PY decision_pack/scripts/generate_regime_transition_board.py \
  --pack-dir ~/temp/decision_packs/20260716

# default.yaml 中设置 rank_rule: transition_x_spike_x_down
$PY decision_pack/scripts/generate_left_side_digest.py \
  --pack-dir ~/temp/decision_packs/20260716
```







HMM状态跳跃

**A. 正式 EOD（Qlib 已有收盘）——不调 AkShare**

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python

# ① 增量验证到 AS_OF（只写正式目录，不抓实时）
AS_OF=2026-08-06 PY=$PY ./scripts/daily_regime_transition_validation.sh

# ② 重建反转标签 + 事件匹配
$PY decision_pack/scripts/rebuild_reversal_event_metrics.py \
  --validation-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720

# ③ 待选池全部 HTML
$PY decision_pack/scripts/plot_regime_transition_example.py \
  --validation-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720 \
  --start-date 2026-04-01 \
  --end-date 2026-08-04 \
  --html-out-dir workspace/plotly_outputs/20260806all/
```

**B. 盘中 provisional（Qlib 尚未有当日 K 线 + AkShare 新浪实时快照）——推荐一条命令**

前提：
- `AS_OF` **必须等于机器今天**（抓快照会校验 `as_of==today`）
- 输出目录为独立路径，**不会**覆盖正式 `q90_to0720`

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python

# ①②③ 全自动：抓快照 → 信号 → 反转标签 → HTML
# 默认目录：workspace/decision_packs/intraday/YYYYMMDD/HHMMSS/
# （HHMMSS 是运行时刻，不要默认当成 1200）
INTRADAY=1 AS_OF=2026-08-05 PY=$PY ./scripts/daily_regime_transition_validation.sh

# 若要固定子目录名（例如 1200），加 OUT_STAMP：
INTRADAY=1 AS_OF=2026-08-05 OUT_STAMP=1200 PY=$PY ./scripts/daily_regime_transition_validation.sh
```

跑完后看终端打印的 **`输出目录: .../OUT_DIR`**，以它为准：

| 产物 | 路径 |
|---|---|
| 信号 | `$OUT_DIR/signals_oos.csv`（应含当日 `AS_OF` 行） |
| 快照 | `$OUT_DIR/live_snapshot.csv` |
| HTML | `$OUT_DIR/html/regime_transition_*.html` |

可选：改 HTML 起始日 `PLOT_START=2026-05-01 INTRADAY=1 AS_OF=... ./scripts/...`

**C. 盘中只补 ②③（一般不必；B 已自动跑完）**

仅当 B 已成功、你只想重跑反转/作图时。**`$OUT_DIR` 必须是 B 打印的那个目录**（含 `signals_oos.csv` + `live_snapshot.csv`），不要手写不存在的 `…/1200/`。

```
# 例：上次 OUT_STAMP=1200 且已跑通
OUT_DIR=workspace/decision_packs/intraday/20260723/1200

$PY decision_pack/scripts/rebuild_reversal_event_metrics.py \
  --validation-dir "$OUT_DIR" \
  --live-snapshot "$OUT_DIR/live_snapshot.csv"

$PY decision_pack/scripts/plot_regime_transition_example.py \
  --validation-dir "$OUT_DIR" \
  --live-snapshot "$OUT_DIR/live_snapshot.csv" \
  --start-date 2026-06-01 \
  --end-date 2026-07-23 \
  --html-out-dir "$OUT_DIR/html/"
```

说明：
- **Qlib 已有该日收盘**：用 **A**，不调 AkShare。
- **当日尚未进 Qlib、要盘中看图**：用 **B**（`INTRADAY=1`）。instruments/`end_date` 可能仍停在 Qlib 日历末日；有 live 快照收盘价时，eligibility 允许算到 `AS_OF`（provisional）。
- 盘中 **B 一条命令即可**；不要在注释里写死 `…/1200/` 再手动 ②③（目录对不上会报 `missing signals_oos.csv` / `as_of mismatch`）。
- 手动 ②③ 必须带 **`--live-snapshot`**，且与该次 run 的快照为同一文件；rebuild 以快照自身的 `as_of` 为准（可略超前于信号末日，仅 WARN）。

- 若还要正式目录评估表，可在 EOD **A** 的 ② 后加：

```
$PY decision_pack/scripts/assess_regime_transition_validation.py \
  --validation-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720 \
  --eval-end 2026-07-22 \
  --horizon 10 \
  --output-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720/assessment
```





单一标的运行代码

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python
$PY decision_pack/scripts/plot_regime_transition_example.py \
  --validation-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720 \
  --code SZ159647 \
  --start-date 2025-04-01 \
  --end-date 2026-07-24 \
  --html-out-dir workspace/plotly_outputs/20260724all/
```



标的预判判断

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python

$PY decision_pack/scripts/scan_triangle_decision_ops.py \
  --html-dir workspace/plotly_outputs/20260805all \
  --validation-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720 \
  --as-of 2026-08-05 \
  --continue-on-error
```

盘中运行标的的判断代码

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python

$PY decision_pack/scripts/scan_triangle_decision_ops.py \
  --pack-dir workspace/decision_packs/intraday/20260805/134749 \
  --intraday \
  --continue-on-error
```

### D. 每日 `hybrid_ma_adx` 分阶段买卖 HTML（全池）

与 `20260806all_hybrid_ma_adx` 相同规则：因果分段 + 固定买卖点；**不画期末强平卖点**。

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python

# 默认：PLOT_START=2026-04-01，输出 workspace/plotly_outputs/{AS_OF}all_hybrid_ma_adx/
AS_OF=2026-08-06 PY=$PY ./scripts/daily_hybrid_ma_adx_html.sh

# 单票 / 改起点 / 指定目录
CODES=SH588710 AS_OF=2026-08-06 PY=$PY ./scripts/daily_hybrid_ma_adx_html.sh
PLOT_START=2026-05-01 AS_OF=2026-08-06 PY=$PY ./scripts/daily_hybrid_ma_adx_html.sh

# 等价 Python
$PY decision_pack/scripts/plot_hybrid_ma_adx_stage_pool.py \
  --start-date 2026-04-01 \
  --end-date 2026-08-06 \
  --html-out-dir workspace/plotly_outputs/20260806all_hybrid_ma_adx
```

### E. 每日「按标的自适应」阶段 + 贝叶斯收缩规则 HTML（全池）

每标的用 **TRAIN_CUTOFF 之前**（默认 = `PLOT_START`=2026-04-01）的历史自选分段法与买卖规则；图窗仅样本外执行。后验分：

`posterior = (n·own + k·prior) / (n + k)`，默认 `k=5`。

默认打分 **`SCORE_MODE=expect_no_trade`**（变体回测 V2 胜出）：`own/prior` 用单笔净期望 `mean_ret_net`（扣双边 0.1% 成本），并允许阶段选 `NO_TRADE`。变体对比见 `workspace/plotly_outputs/{AS_OF}all_adaptive_variants/variants_report.md`。

分段模式默认 **`REGIME_MODE=legacy_score_method`（B0）**。交易默认 **`TRADE_MODE=hold_up`**：只做多——上涨阶段开始买入、上涨结束卖出；下跌/横盘不交易。旧贝叶斯分阶段规则用 `TRADE_MODE=bayes_rules`。

缓存键：`TRAIN_CUTOFF` + `TRADE_MODE` + `SCORE_MODE` + `REGIME_MODE` + `truth_version` + `method_impl`；命中则复用 `configs/`；`RETRAIN=1` 强制重训。

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python

# 默认输出 workspace/plotly_outputs/{AS_OF}all_adaptive/
AS_OF=2026-08-06 PY=$PY ./scripts/daily_adaptive_stage_html.sh

# 单票 / 强制重训 / 切换模式
CODES=SH588710 AS_OF=2026-08-06 PY=$PY ./scripts/daily_adaptive_stage_html.sh
RETRAIN=1 AS_OF=2026-08-06 PY=$PY ./scripts/daily_adaptive_stage_html.sh
TRADE_MODE=bayes_rules RETRAIN=1 AS_OF=2026-08-06 PY=$PY ./scripts/daily_adaptive_stage_html.sh
REGIME_MODE=ensemble_cv RETRAIN=1 AS_OF=2026-08-06 PY=$PY ./scripts/daily_adaptive_stage_html.sh
```



用每日脚本，把 `AS_OF` 设成 2026-08-07 即可；输出会到新目录 `20260807all_adaptive`。

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
AS_OF=2026-08-07 \
PY=~/etf-daily-output/python/envs/py312/bin/python \
./scripts/daily_adaptive_stage_html.sh
```



生成

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
# 1) 先出 adaptive HTML（若还没有）
AS_OF=2026-08-14 PY=~/etf-daily-output/python/envs/py312/bin/python \
  SKIP_HTML=1 \
  ./scripts/daily_regime_transition_validation.sh
AS_OF=2026-08-14 PY=~/etf-daily-output/python/envs/py312/bin/python \
  ./scripts/daily_adaptive_stage_html.sh

# 2) 生成下一交易日执行方案 MD/CSV/JSON
AS_OF=2026-08-14 PY=~/etf-daily-output/python/envs/py312/bin/python \
  ./scripts/daily_adaptive_execution_plan.sh
```





### 1)日中处理 先抓当日实时快照

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python
STAMP=$(date +%H%M%S)
SNAP="workspace/decision_packs/intraday/20260812/${STAMP}/live_snapshot.csv"
"${PY}" -m decision_pack.src.regime_transition_live_snapshot \
  --as-of 2026-08-12 \
  --output "${SNAP}"
```

（AkShare 新浪 ETF 现货；`as_of` 须等于机器当天，否则加 `--allow-stale-as-of` 仅作回放。）

### 2) 再出 HTML（叠实时 bar）

推荐：新建 0810 目录，复用 0807 的 frozen configs（不重训）

```
cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
PY=~/etf-daily-output/python/envs/py312/bin/python \
  ./scripts/daily_adaptive_stage_html_intraday.sh
```



```
mkdir -p workspace/plotly_outputs/20260812all_adaptive
cp -a workspace/plotly_outputs/20260811all_adaptive/configs \
      workspace/plotly_outputs/20260811all_adaptive/.train_cache_* \
      workspace/plotly_outputs/20260812all_adaptive/ 2>/dev/null || true

AS_OF=2026-08-12 \
TRAIN_CUTOFF=2026-04-01 \
PLOT_START=2026-04-01 \
HTML_OUT_DIR=workspace/plotly_outputs/20260812all_adaptive \
LIVE_SNAPSHOT="${SNAP}" \
PY=~/etf-daily-output/python/envs/py312/bin/python \
  ./scripts/daily_adaptive_stage_html.sh
```



### 正确跑法（日中 / 收盘前）

语义拆开：

- `AS_OF` = 上一正式收盘（Qlib 已确认，这里 08-07）

- `TRADE_DATE` = 今天（08-10）

- `SNAPSHOT` = 今日盘中 OHLC（你已有的 live_snapshot）

- `ADAPTIVE_DIR` = `20260810all_adaptive`（用它的 configs）                   

  ```
  AS_OF=2026-08-11 TRADE_DATE=2026-08-12 \
  ADAPTIVE_DIR=workspace/plotly_outputs/20260812all_adaptive \
  SNAPSHOT=workspace/decision_packs/intraday/20260810/162147/live_snapshot.csv \
  OUT_DIR=workspace/decision_packs/adaptive_execution/20260810_intraday \
  PY=~/etf-daily-output/python/envs/py312/bin/python \
    ./scripts/daily_adaptive_execution_plan.sh
  ```




波动率补偿

```
python decision_pack/scripts/build_vol_return_board.py --as-of 2026-08-10
```

- 结果：`workspace/plotly_outputs/20260810_vol_return_board/`（CSV + README）

按
