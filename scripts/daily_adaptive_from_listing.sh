#!/usr/bin/env bash
# 每日生成 from_listing 自适应 HTML（上市日 → AS_OF）
#
# EOD 增量（默认：复用前一天 *_from_listing HTML，只补一根 K 线）:
#   etf-daily listing --as-of 2026-09-04
#   PY=~/etf-daily-output/python/envs/py312/bin/python ./scripts/daily_adaptive_from_listing.sh
#   AS_OF=2026-09-04 JOBS=8 PY=... ./scripts/daily_adaptive_from_listing.sh
#
# 盘中：自动用今天最新一份含 signals_oos.csv 的盘中包，不用写 OUT_STAMP:
#   etf-daily listing --intraday
#
# 强制全量重画:
#   INCREMENTAL=0 AS_OF=2026-09-04 PY=... ./scripts/daily_adaptive_from_listing.sh
#
# 手动指定昨日目录（跳过自动发现）:
#   PREV_LISTING_DIR=~/etf-daily-output/temp/plotly_outputs/20260903_from_listing \
#     AS_OF=2026-09-04 PY=... ./scripts/daily_adaptive_from_listing.sh
#
# 单票调试:
#   CODES=SH588710 AS_OF=2026-09-04 PY=... ./scripts/daily_adaptive_from_listing.sh
#
# 共享训练缓存 / 上市日缓存（默认开启）:
#   TRAIN_CACHE_DIR=... LISTING_CACHE=... AS_OF=... PY=... ./scripts/daily_adaptive_from_listing.sh
#   TRAIN_CACHE_DIR=off LISTING_CACHE=off 可关
#
# 图12 六状态背景（默认开，后处理重画；FIG12=0 可关）:
#   FIG12=0 AS_OF=... PY=... ./scripts/daily_adaptive_from_listing.sh
#
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/daily_env.sh"
AS_OF="${AS_OF:-$(date +%Y-%m-%d)}"
TRAIN_CUTOFF="${TRAIN_CUTOFF:-2026-04-01}"
VALIDATION_DIR="${VALIDATION_DIR:-${PROJECT_ROOT}/runtime/decision_packs/20260720/regime_transition_validation_q90_to0720}"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-}"
HTML_OUT_DIR="${HTML_OUT_DIR:-}"
CODES="${CODES:-}"
MAX_CODES="${MAX_CODES:-}"
RETRAIN="${RETRAIN:-0}"
JOBS="${JOBS:-8}"
LIVE_SNAPSHOT="${LIVE_SNAPSHOT:-}"
DISABLE_RU_DIAG="${DISABLE_RU_DIAG:-0}"
# INCREMENTAL=1 (default) → html mode; INCREMENTAL=0 → full rebuild.
# Legacy INCREMENTAL_MODE=html|off still honored when INCREMENTAL unset explicitly via MODE.
INCREMENTAL="${INCREMENTAL:-1}"
INCREMENTAL_MODE="${INCREMENTAL_MODE:-}"
INCREMENTAL_FROM="${INCREMENTAL_FROM:-}"
PREV_LISTING_DIR="${PREV_LISTING_DIR:-}"
FAIR_PATH_EXTRA_TRAIL_YEARS="${FAIR_PATH_EXTRA_TRAIL_YEARS:-1,0.5,0.25,1/12}"
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${PLOTLY_ROOT}/_train_cache}"
LISTING_CACHE="${LISTING_CACHE:-${PLOTLY_ROOT}/_listing_dates.csv}"
FIG12="${FIG12:-1}"

if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[ERROR] AS_OF must be YYYY-MM-DD, got: ${AS_OF}" >&2
  exit 2
fi
if ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "[ERROR] AS_OF is not a valid date: ${AS_OF}" >&2
  exit 2
fi
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all_adaptive}"
HTML_OUT_DIR="${HTML_OUT_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}_from_listing}"
INTRADAY="${INTRADAY:-0}"
if [[ "${INTRADAY}" == "1" || "${INTRADAY}" == "true" || "${INTRADAY}" == "yes" ]]; then
  if [[ -z "${LIVE_SNAPSHOT}" || "${VALIDATION_DIR}" == "${PROJECT_ROOT}/runtime/decision_packs/20260720/regime_transition_validation_q90_to0720" ]]; then
    _PACK="$(latest_intraday_pack "${PROJECT_ROOT}/runtime/decision_packs/intraday/${AS_OF_TAG}")"
    if [[ -z "${_PACK}" ]]; then
      echo "[ERROR] 今天还没有盘中验证包：runtime/decision_packs/intraday/${AS_OF_TAG}/<HHMMSS>/" >&2
      echo "  先跑：etf-daily regime --skip-html --intraday" >&2
      exit 3
    fi
    if [[ "${VALIDATION_DIR}" == "${PROJECT_ROOT}/runtime/decision_packs/20260720/regime_transition_validation_q90_to0720" ]]; then
      VALIDATION_DIR="${_PACK}"
    fi
    if [[ -z "${LIVE_SNAPSHOT}" ]]; then
      LIVE_SNAPSHOT="${_PACK}/live_snapshot.csv"
    fi
    echo "[INFO] intraday pack: ${VALIDATION_DIR}"
    echo "[INFO] intraday snapshot: ${LIVE_SNAPSHOT}"
  fi
fi

# Resolve incremental mode: INCREMENTAL_MODE wins if set; else INCREMENTAL=1 → html.
if [[ -z "${INCREMENTAL_MODE}" ]]; then
  if [[ "${INCREMENTAL}" == "0" || "${INCREMENTAL}" == "false" || "${INCREMENTAL}" == "off" || "${INCREMENTAL}" == "no" ]]; then
    INCREMENTAL_MODE="off"
  else
    INCREMENTAL_MODE="html"
  fi
fi

find_prev_listing_dir() {
  local as_of_tag="$1"
  local best=""
  local best_tag=""
  shopt -s nullglob
  for d in "${PLOTLY_ROOT}"/*_from_listing; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/listing_dates.csv" ]] || continue
    local base
    base="$(basename "${d}")"
    # expect YYYYMMDD_from_listing
    local tag="${base%_from_listing}"
    [[ "${tag}" =~ ^[0-9]{8}$ ]] || continue
    if [[ "${tag}" < "${as_of_tag}" ]]; then
      if [[ -z "${best_tag}" || "${tag}" > "${best_tag}" ]]; then
        best_tag="${tag}"
        best="${d}"
      fi
    fi
  done
  shopt -u nullglob
  echo "${best}"
}

if [[ "${INCREMENTAL_MODE}" == "html" ]]; then
  if [[ -n "${INCREMENTAL_FROM}" ]]; then
    PREV_LISTING_DIR="${INCREMENTAL_FROM}"
  elif [[ -n "${PREV_LISTING_DIR}" ]]; then
    :
  else
    PREV_LISTING_DIR="$(find_prev_listing_dir "${AS_OF_TAG}")"
  fi
  if [[ -z "${PREV_LISTING_DIR}" || ! -d "${PREV_LISTING_DIR}" ]]; then
    echo "[WARN] no previous *_from_listing found; falling back to full rebuild"
    INCREMENTAL_MODE="off"
    PREV_LISTING_DIR=""
  fi
fi

cd "${PROJECT_ROOT}"

ARGS=(
  src/etf_daily/plots/plot_adaptive_from_listing.py
  --as-of "${AS_OF}"
  --train-cutoff "${TRAIN_CUTOFF}"
  --validation-dir "${VALIDATION_DIR}"
  --config-source-dir "${CONFIG_SOURCE_DIR}"
  --html-out-dir "${HTML_OUT_DIR}"
  --jobs "${JOBS}"
  --incremental-mode "${INCREMENTAL_MODE}"
)

_tc="$(echo "${TRAIN_CACHE_DIR}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${TRAIN_CACHE_DIR}" && "${_tc}" != "off" && "${_tc}" != "none" && "${_tc}" != "0" ]]; then
  ARGS+=(--train-cache-dir "${TRAIN_CACHE_DIR}")
fi
_lc="$(echo "${LISTING_CACHE}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${LISTING_CACHE}" && "${_lc}" != "off" && "${_lc}" != "none" && "${_lc}" != "0" ]]; then
  ARGS+=(--listing-cache "${LISTING_CACHE}")
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
if [[ "${INCREMENTAL_MODE}" == "html" && -n "${PREV_LISTING_DIR}" ]]; then
  ARGS+=(--incremental-from "${PREV_LISTING_DIR}")
fi
_fp_extra="$(echo "${FAIR_PATH_EXTRA_TRAIL_YEARS}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${FAIR_PATH_EXTRA_TRAIL_YEARS}" && "${_fp_extra}" != "off" && "${_fp_extra}" != "none" && "${_fp_extra}" != "0" ]]; then
  ARGS+=(--fair-path-extra-trail-years "${FAIR_PATH_EXTRA_TRAIL_YEARS}")
fi

echo "[INFO] AS_OF=${AS_OF} TRAIN_CUTOFF=${TRAIN_CUTOFF} JOBS=${JOBS} RETRAIN=${RETRAIN}"
echo "[INFO] CONFIG_SOURCE_DIR=${CONFIG_SOURCE_DIR}"
echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR}"
echo "[INFO] INCREMENTAL_MODE=${INCREMENTAL_MODE} PREV_LISTING_DIR=${PREV_LISTING_DIR:-}"
echo "[INFO] TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}"
echo "[INFO] LISTING_CACHE=${LISTING_CACHE}"
echo "[INFO] FIG12=${FIG12}"
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" "${ARGS[@]}"

_fig12="$(echo "${FIG12}" | tr '[:upper:]' '[:lower:]')"
if [[ "${_fig12}" != "0" && "${_fig12}" != "false" && "${_fig12}" != "off" && "${_fig12}" != "no" ]]; then
  FIG12_ARGS=(
    src/etf_daily/plots/add_fig12_six_states.py
    --html-dir "${HTML_OUT_DIR}"
    --jobs "${JOBS}"
  )
  if [[ -n "${CODES}" ]]; then
    for c in ${CODES}; do
      FIG12_ARGS+=(--code "${c}")
    done
  fi
  echo "[INFO] fig12 post-process html-dir=${HTML_OUT_DIR} jobs=${JOBS}"
  PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" "${FIG12_ARGS[@]}"
fi
echo "[OK] 输出目录: ${HTML_OUT_DIR}"
