#!/usr/bin/env bash
# 每日生成待选池「按标的自适应」趋势阶段 + 贝叶斯收缩买卖规则 HTML
#
# EOD 示例:
#   PY=~/etf-daily-output/python/envs/py312/bin/python ./scripts/daily_adaptive_stage_html.sh
#   AS_OF=2026-08-06 PY=... ./scripts/daily_adaptive_stage_html.sh
#
# 训练截止默认 = PLOT_START（样本外起点）。同一 TRAIN_CUTOFF 复用 configs/，不重训：
#   RETRAIN=1 AS_OF=2026-08-06 PY=... ./scripts/daily_adaptive_stage_html.sh
# 跨日共享训练缓存（默认 ~/etf-daily-output/temp/plotly_outputs/_train_cache；关: TRAIN_CACHE_DIR=off）:
#   TRAIN_CACHE_DIR=/path/to/cache AS_OF=... PY=... ./scripts/daily_adaptive_stage_html.sh
#
# 单票调试:
#   CODES=SH588710 AS_OF=2026-08-06 PY=... ./scripts/daily_adaptive_stage_html.sh
#
# 多尺度公平路径（默认图8=1y / 图9=6m / 图10=3m / 图11=1m；关: FAIR_PATH_EXTRA_TRAIL_YEARS=off）:
#   FAIR_PATH_EXTRA_TRAIL_YEARS=1,0.5,0.25,1/12 AS_OF=... PY=... ./scripts/daily_adaptive_stage_html.sh
#
# 研究票卡 CSV（不改 hold_up）默认写 HTML 目录 ticket_card.csv；关掉:
#   TICKET_CARD=0 PY=... ./scripts/daily_adaptive_stage_html.sh
#
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/daily_env.sh"
VALIDATION_DIR="${VALIDATION_DIR:-${PROJECT_ROOT}/workspace/decision_packs/20260720/regime_transition_validation_q90_to0720}"
PLOT_START="${PLOT_START:-2026-04-01}"
TRAIN_CUTOFF="${TRAIN_CUTOFF:-${PLOT_START}}"
LIVE_SNAPSHOT="${LIVE_SNAPSHOT:-}"
CODES="${CODES:-}"
MAX_CODES="${MAX_CODES:-}"
RETRAIN="${RETRAIN:-0}"
JOBS="${JOBS:-8}"
PRIOR_K="${PRIOR_K:-5}"
SCORE_MODE="${SCORE_MODE:-expect_no_trade}"
# default legacy (B0): B4 ensemble_cv improves dual-truth labels but failed trade-edge gate
REGIME_MODE="${REGIME_MODE:-legacy_score_method}"
# hold_up = long-only enter/exit with up regime; bayes_rules = old per-stage sweep
TRADE_MODE="${TRADE_MODE:-hold_up}"
# HRP 切簇 CSV。未设置或为空时不传 --hrp-membership-csv。
DISABLE_RU_DIAG="${DISABLE_RU_DIAG:-0}"
# Fig7=2y + fig8..11 multi-scale trails (1y / 6m / 3m / 1m). Set off|0|none to disable.
FAIR_PATH_EXTRA_TRAIL_YEARS="${FAIR_PATH_EXTRA_TRAIL_YEARS:-1,0.5,0.25,1/12}"
# Shared pass1 configs across AS_OF out dirs (same TRAIN_CUTOFF + version tag).
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${PLOTLY_ROOT}/_train_cache}"

AS_OF="${AS_OF:-$(date +%Y-%m-%d)}"
if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[ERROR] AS_OF must be YYYY-MM-DD, got: ${AS_OF}" >&2
  exit 2
fi
if ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "[ERROR] AS_OF is not a valid date: ${AS_OF}" >&2
  exit 2
fi
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"
HTML_OUT_DIR="${HTML_OUT_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all_adaptive}"

cd "${PROJECT_ROOT}"

ARGS=(
  decision_pack/scripts/plot_adaptive_stage_pool.py
  --start-date "${PLOT_START}"
  --end-date "${AS_OF}"
  --train-cutoff "${TRAIN_CUTOFF}"
  --validation-dir "${VALIDATION_DIR}"
  --html-out-dir "${HTML_OUT_DIR}"
  --prior-k "${PRIOR_K}"
  --score-mode "${SCORE_MODE}"
  --regime-mode "${REGIME_MODE}"
  --trade-mode "${TRADE_MODE}"
  --jobs "${JOBS}"
  --continue-on-error
)
if [[ -n "${HRP_MEMBERSHIP_CSV:-}" ]]; then
  ARGS+=(--hrp-membership-csv "${HRP_MEMBERSHIP_CSV}")
fi

_tc="$(echo "${TRAIN_CACHE_DIR}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${TRAIN_CACHE_DIR}" && "${_tc}" != "off" && "${_tc}" != "none" && "${_tc}" != "0" ]]; then
  ARGS+=(--train-cache-dir "${TRAIN_CACHE_DIR}")
fi
if [[ "${RETRAIN}" == "1" || "${RETRAIN}" == "true" || "${RETRAIN}" == "yes" ]]; then
  ARGS+=(--retrain)
fi
if [[ "${DISABLE_RU_DIAG}" == "1" || "${DISABLE_RU_DIAG}" == "true" || "${DISABLE_RU_DIAG}" == "yes" ]]; then
  ARGS+=(--disable-ru-diag)
fi
if [[ -n "${LIVE_SNAPSHOT}" ]]; then
  ARGS+=(--live-snapshot "${LIVE_SNAPSHOT}")
fi
if [[ -n "${MAX_CODES}" ]]; then
  ARGS+=(--max-codes "${MAX_CODES}")
fi
if [[ -n "${CODES}" ]]; then
  for c in ${CODES}; do
    ARGS+=(--code "${c}")
  done
fi
_fp_extra="$(echo "${FAIR_PATH_EXTRA_TRAIL_YEARS}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${FAIR_PATH_EXTRA_TRAIL_YEARS}" && "${_fp_extra}" != "off" && "${_fp_extra}" != "none" && "${_fp_extra}" != "0" ]]; then
  ARGS+=(--fair-path-extra-trail-years "${FAIR_PATH_EXTRA_TRAIL_YEARS}")
fi

echo "[INFO] AS_OF=${AS_OF} PLOT_START=${PLOT_START} TRAIN_CUTOFF=${TRAIN_CUTOFF} RETRAIN=${RETRAIN} SCORE_MODE=${SCORE_MODE} REGIME_MODE=${REGIME_MODE} TRADE_MODE=${TRADE_MODE}"
echo "[INFO] FAIR_PATH_EXTRA_TRAIL_YEARS=${FAIR_PATH_EXTRA_TRAIL_YEARS}"
echo "[INFO] TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}"
echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR}"
"${PY}" "${ARGS[@]}"
echo "[OK] 输出目录: ${HTML_OUT_DIR}"
if [[ "${TICKET_CARD:-1}" != "0" ]]; then
  echo "[INFO] building research ticket_card.csv (TICKET_CARD=0 to skip)"
  "${PY}" decision_pack/scripts/build_daily_ticket_card.py \
    --as-of "${AS_OF}" \
    --adaptive-dir "${HTML_OUT_DIR}" \
    --out "${HTML_OUT_DIR}/ticket_card.csv"
fi
