#!/usr/bin/env bash
# 每日增量 / 盘中 provisional regime transition 验证
#
# EOD 完整链路（正式目录，不抓实时；①验证 + ②反转标签 + ③全池 HTML）:
#   PY=~/etf-daily-output/python/envs/py312/bin/python ./scripts/daily_regime_transition_validation.sh
#   AS_OF=2026-07-27 PY=... ./scripts/daily_regime_transition_validation.sh
#
# 仅跑 ① 验证（跳过 ②③）——最快日更；不需要刷新反转标签/事件匹配时用:
#   SKIP_REBUILD=1 SKIP_HTML=1 AS_OF=2026-07-27 PY=... ./scripts/daily_regime_transition_validation.sh
# 说明: SKIP_REBUILD=1 跳过 ② rebuild_reversal_event_metrics（第二次加载 panel +
#   重算反转标签）。信号打分/go_nogo 仍更新。若下游 HTML/OPS 依赖反转列，勿跳过。
#
# EOD 当月按天增量（默认）: 保留 ${MONTH}.signals.csv，只清 .done，backtest 只打
#   as_of > 已有最大日 的新交易日。整月重算兜底:
#   FORCE_MONTH_REBUILD=1 AS_OF=... PY=... ./scripts/daily_regime_transition_validation.sh
#
# 盘中用法（独立 timestamp 目录，Qlib 历史 + AkShare 实时快照）:
#   INTRADAY=1 AS_OF=2026-07-23 PY=... ./scripts/daily_regime_transition_validation.sh
#   # 可选固定子目录名（默认 HHMMSS）:
#   INTRADAY=1 AS_OF=2026-07-23 OUT_STAMP=1200 PY=... ./scripts/daily_regime_transition_validation.sh
#
# 可选：生成三角操作清单 OPS md（④ scan_triangle_decision_ops）:
#   RUN_OPS_SCAN=1 AS_OF=2026-07-27 PY=... ./scripts/daily_regime_transition_validation.sh
# 盘中完整链路（含 ④ OPS）:
#   INTRADAY=1 RUN_OPS_SCAN=1 OUT_STAMP=103630 AS_OF=2026-07-28 PY=... ./scripts/daily_regime_transition_validation.sh
# 已有盘中包，只补跑 ④:
#   $PY src/etf_daily/regime/scan_triangle_decision_ops.py \\
#     --pack-dir runtime/decision_packs/intraday/20260728/103630 --intraday --continue-on-error
#
# AS_OF: 信号截止日（含）。未设置时默认用系统今天。所有日期相关路径/参数均由此推导。
# INTRADAY=1: 抓取冻结快照并写入 runtime/decision_packs/intraday/YYYYMMDD/<OUT_STAMP|HHMMSS>/
set -euo pipefail

# ── 路径配置（按需修改）──────────────────────────────────────
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/daily_env.sh"
FORMAL_OUT_DIR="${PROJECT_ROOT}/runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
CACHE_DIR="${PROJECT_ROOT}/runtime/decision_packs/regime_transition_model_cache"
INTRADAY="${INTRADAY:-0}"
PLOT_START="${PLOT_START:-2026-04-01}"
SKIP_REBUILD="${SKIP_REBUILD:-0}"
SKIP_HTML="${SKIP_HTML:-0}"
RUN_OPS_SCAN="${RUN_OPS_SCAN:-0}"
JOBS="${JOBS:-8}"
FORCE_MONTH_REBUILD="${FORCE_MONTH_REBUILD:-0}"
HTML_POOL_DIR="${HTML_POOL_DIR:-}"  # 默认 ~/etf-daily-output/temp/plotly_outputs/${AS_OF_TAG}all/

# ── 日期（可用 AS_OF=YYYY-MM-DD 覆盖系统今天）────────────────
AS_OF="${AS_OF:-$(date +%Y-%m-%d)}"
if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[ERROR] AS_OF must be YYYY-MM-DD, got: ${AS_OF}" >&2
  exit 2
fi
if ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "[ERROR] AS_OF is not a valid date: ${AS_OF}" >&2
  exit 2
fi
MONTH="${AS_OF:0:7}"
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"

cd "${PROJECT_ROOT}"

LIVE_SNAPSHOT=""
HTML_OUT_DIR="${HTML_POOL_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all}"

if [[ "${INTRADAY}" == "1" ]]; then
  STAMP="${OUT_STAMP:-$(date +%H%M%S)}"
  OUT_DIR="${PROJECT_ROOT}/runtime/decision_packs/intraday/${AS_OF_TAG}/${STAMP}"
  HTML_OUT_DIR="${OUT_DIR}/html"
  LIVE_SNAPSHOT="${OUT_DIR}/live_snapshot.csv"
  mkdir -p "${OUT_DIR}/shards" "${HTML_OUT_DIR}"

  echo "[INFO] INTRADAY mode AS_OF=${AS_OF} OUT_DIR=${OUT_DIR}"
  echo "[INFO] (do not assume OUT_DIR=.../1200 unless OUT_STAMP=1200 was set)"

  # Seed historical shards from formal dir (exclude current month) for fast resume.
  if [[ -d "${FORMAL_OUT_DIR}/shards" ]]; then
    shopt -s nullglob
    for f in "${FORMAL_OUT_DIR}/shards/"*; do
      base="$(basename "${f}")"
      if [[ "${base}" == "${MONTH}."* ]]; then
        continue
      fi
      cp -a "${f}" "${OUT_DIR}/shards/"
    done
    shopt -u nullglob
    echo "[INFO] seeded historical shards from ${FORMAL_OUT_DIR}/shards (excluded ${MONTH})"
  else
    echo "[WARN] formal shards missing at ${FORMAL_OUT_DIR}/shards; full backtest will run"
  fi

  # Force-rebuild AS_OF month even when reusing OUT_STAMP (e.g. 1200).
  # Otherwise a stale ${MONTH}.done keeps the previous cluster universe
  # (e.g. SH516970) while the new live snapshot already follows the updated pool.
  rm -f "${OUT_DIR}/shards/${MONTH}.done"
  rm -f "${OUT_DIR}/shards/${MONTH}.signals.csv"
  echo "[INFO] cleared ${MONTH} shards under ${OUT_DIR}/shards (force recompute for current pool)"

  # Capture frozen snapshot BEFORE any shard recomputation.
  echo "[INFO] capturing AkShare live snapshot -> ${LIVE_SNAPSHOT}"
  "${PY}" -m etf_daily.lib.regime_transition_live_snapshot \
    --as-of "${AS_OF}" \
    --output "${LIVE_SNAPSHOT}"

  LIVE_ARGS=(--live-snapshot "${LIVE_SNAPSHOT}")
else
  OUT_DIR="${FORMAL_OUT_DIR}"
  LIVE_ARGS=()
  echo "[INFO] EOD mode AS_OF=${AS_OF} MONTH=${MONTH} OUT_DIR=${OUT_DIR}"
  echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR} PLOT_START=${PLOT_START}"

  if [[ "${FORCE_MONTH_REBUILD}" == "1" || "${FORCE_MONTH_REBUILD}" == "true" || "${FORCE_MONTH_REBUILD}" == "yes" ]]; then
    # 整月重算兜底（例如池变更 / 怀疑当月 shard 损坏）
    rm -f "${OUT_DIR}/shards/${MONTH}.done"
    rm -f "${OUT_DIR}/shards/${MONTH}.signals.csv"
    echo "[INFO] FORCE_MONTH_REBUILD=1: cleared ${MONTH} shards under ${OUT_DIR}/shards"
  else
    # 按天增量：保留当月 signals.csv，只清 .done，让 backtest 追加新交易日
    rm -f "${OUT_DIR}/shards/${MONTH}.done"
    echo "[INFO] day-resume: cleared ${MONTH}.done (kept ${MONTH}.signals.csv if present)"
  fi
fi

# ── ① 增量跑验证（resume 默认开启，历史月份跳过）────────────
echo ""
echo "=== ① regime transition validation ==="
"${PY}" src/etf_daily/regime/backtest_regime_transition_signals.py \
  --output-dir "${OUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --eval-start 2024-06-01 \
  --eval-end "${AS_OF}" \
  --horizon 10 \
  --jobs "${JOBS}" \
  "${LIVE_ARGS[@]+"${LIVE_ARGS[@]}"}"

run_rebuild_and_plot() {
  if [[ "${SKIP_REBUILD}" != "1" ]]; then
    echo ""
    echo "=== ② rebuild reversal labels + event match ==="
    "${PY}" src/etf_daily/regime/rebuild_reversal_event_metrics.py \
      --validation-dir "${OUT_DIR}" \
      "${LIVE_ARGS[@]+"${LIVE_ARGS[@]}"}"
  else
    echo "[SKIP] ② rebuild_reversal_event_metrics (SKIP_REBUILD=1)"
  fi

  if [[ "${SKIP_HTML}" != "1" ]]; then
    echo ""
    echo "=== ③ plot pool HTML ==="
    mkdir -p "${HTML_OUT_DIR}"
    "${PY}" src/etf_daily/plots/plot_regime_transition_example.py \
      --validation-dir "${OUT_DIR}" \
      --start-date "${PLOT_START}" \
      --end-date "${AS_OF}" \
      --html-out-dir "${HTML_OUT_DIR}" \
      --continue-on-error \
      "${LIVE_ARGS[@]+"${LIVE_ARGS[@]}"}"
  else
    echo "[SKIP] ③ plot HTML (SKIP_HTML=1)"
  fi
}

if [[ "${INTRADAY}" == "1" ]]; then
  run_rebuild_and_plot

  if [[ "${RUN_OPS_SCAN}" == "1" ]]; then
    echo ""
    echo "=== ④ triangle decision ops scan (intraday) ==="
    "${PY}" src/etf_daily/regime/scan_triangle_decision_ops.py \
      --pack-dir "${OUT_DIR}" \
      --as-of "${AS_OF}" \
      --intraday \
      --continue-on-error
  fi
else
  run_rebuild_and_plot

  if [[ "${RUN_OPS_SCAN}" == "1" ]]; then
    echo ""
    echo "=== ④ triangle decision ops scan (guide next session) ==="
    "${PY}" src/etf_daily/regime/scan_triangle_decision_ops.py \
      --html-dir "${HTML_OUT_DIR}" \
      --validation-dir "${OUT_DIR}" \
      --as-of "${AS_OF}" \
      --continue-on-error
  fi
fi

# ── 快速检查结果 ──────────────────────────────────────────
echo ""
echo "=== 运行完成 ==="
echo "mode:     $([ "${INTRADAY}" = "1" ] && echo INTRADAY || echo EOD)"
echo "AS_OF:    ${AS_OF}"
echo "输出目录: ${OUT_DIR}"
echo "信号文件: ${OUT_DIR}/signals_oos.csv"
echo "报告:     ${OUT_DIR}/REGIME_TRANSITION_VALIDATION.md"
if [[ -n "${LIVE_SNAPSHOT}" && -f "${LIVE_SNAPSHOT}" ]]; then
  echo "快照:     ${LIVE_SNAPSHOT}"
  "${PY}" - <<PY
import pandas as pd
p = r"""${LIVE_SNAPSHOT}"""
df = pd.read_csv(p)
print(f"snapshot_codes={df['code'].nunique()} captured_at={df['captured_at'].iloc[0]}")
PY
fi
if [[ -n "${HTML_OUT_DIR}" && -d "${HTML_OUT_DIR}" ]]; then
  echo "HTML目录: ${HTML_OUT_DIR}"
  echo "HTML数量: $(find "${HTML_OUT_DIR}" -maxdepth 1 -name 'regime_transition_*.html' 2>/dev/null | wc -l)"
fi
if [[ "${RUN_OPS_SCAN}" == "1" ]]; then
  echo "OPS扫描:  ${OUT_DIR}/triangle_decision_scan_${AS_OF_TAG}/"
fi
echo "go/no-go: $("${PY}" -c "import json; print(json.load(open('${OUT_DIR}/go_nogo.json'))['pass'])" 2>/dev/null || echo 'N/A')"
