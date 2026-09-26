#!/usr/bin/env bash
# 日中自适应 HTML 一键入口：抓快照 → 复用上一批 configs → 叠 live bar 出图
#
# 用法（一般不用改日期）:
#   PY=~/etf-daily-output/python/envs/py312/bin/python \
#     ./scripts/daily_adaptive_stage_html_intraday.sh
#
# 指定日（回放）:
#   AS_OF=2026-08-12 ./scripts/daily_adaptive_stage_html_intraday.sh
#   ./scripts/daily_adaptive_stage_html_intraday.sh 2026-08-12
#
# 常用覆盖:
#   PREV_ADAPTIVE_DIR=~/etf-daily-output/temp/plotly_outputs/20260811all_adaptive
#   LIVE_SNAPSHOT=/path/to/live_snapshot.csv   # 已有快照则跳过抓取
#   SKIP_SNAPSHOT=1                            # 强制不抓（须已设 LIVE_SNAPSHOT）
#   CODES=SH515050 MAX_CODES=3
#   TRAIN_CUTOFF=2026-04-01 PLOT_START=2026-04-01
#   RETRAIN=0   # 日中默认不重训；要重训才设 RETRAIN=1
#   ALLOW_STALE=1  # as_of≠今天时抓快照（脚本在日期不一致时也会自动加）
#
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/daily_env.sh"

PLOT_START="${PLOT_START:-2026-04-01}"
TRAIN_CUTOFF="${TRAIN_CUTOFF:-${PLOT_START}}"
RETRAIN="${RETRAIN:-0}"
ALLOW_STALE="${ALLOW_STALE:-0}"
SKIP_SNAPSHOT="${SKIP_SNAPSHOT:-0}"
CODES="${CODES:-}"
MAX_CODES="${MAX_CODES:-}"
PREV_ADAPTIVE_DIR="${PREV_ADAPTIVE_DIR:-}"
LIVE_SNAPSHOT="${LIVE_SNAPSHOT:-}"
HTML_OUT_DIR="${HTML_OUT_DIR:-}"

# 位置参数优先：./script.sh 2026-08-12
if [[ "${1:-}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  AS_OF="$1"
  shift
fi
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
TODAY_TAG="$(date +%Y%m%d)"
HTML_OUT_DIR="${HTML_OUT_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all_adaptive}"

find_prev_adaptive_dir() {
  local as_of_tag="$1"
  local best_tag="" best_dir=""
  local d base tag
  shopt -s nullglob
  for d in "${PLOTLY_ROOT}"/*all_adaptive; do
    [[ -d "${d}/configs" ]] || continue
    base="$(basename "${d}")"
    tag="${base%all_adaptive}"
    [[ "${tag}" =~ ^[0-9]{8}$ ]] || continue
    if [[ "${tag}" < "${as_of_tag}" ]]; then
      if [[ -z "${best_tag}" || "${tag}" > "${best_tag}" ]]; then
        best_tag="${tag}"
        best_dir="${d}"
      fi
    fi
  done
  shopt -u nullglob
  echo "${best_dir}"
}

if [[ -z "${PREV_ADAPTIVE_DIR}" ]]; then
  PREV_ADAPTIVE_DIR="$(find_prev_adaptive_dir "${AS_OF_TAG}")"
fi
if [[ -z "${PREV_ADAPTIVE_DIR}" || ! -d "${PREV_ADAPTIVE_DIR}/configs" ]]; then
  echo "[ERROR] 找不到可复用的上一批 adaptive configs。" >&2
  echo "  请先跑过至少一天 EOD：AS_OF=... ./scripts/daily_adaptive_stage_html.sh" >&2
  echo "  或手动指定 PREV_ADAPTIVE_DIR=~/etf-daily-output/temp/plotly_outputs/YYYYMMDDall_adaptive" >&2
  exit 3
fi
if [[ "${PREV_ADAPTIVE_DIR}" != /* ]]; then
  PREV_ADAPTIVE_DIR="${PROJECT_ROOT}/${PREV_ADAPTIVE_DIR}"
fi

mkdir -p "${HTML_OUT_DIR}"
echo "[INFO] 复用 configs: ${PREV_ADAPTIVE_DIR} → ${HTML_OUT_DIR}"
rm -rf "${HTML_OUT_DIR}/configs"
cp -a "${PREV_ADAPTIVE_DIR}/configs" "${HTML_OUT_DIR}/"
shopt -s nullglob
for cache in "${PREV_ADAPTIVE_DIR}"/.train_cache_*; do
  cp -a "${cache}" "${HTML_OUT_DIR}/"
done
shopt -u nullglob

if [[ -n "${LIVE_SNAPSHOT}" ]]; then
  if [[ "${LIVE_SNAPSHOT}" != /* ]]; then
    LIVE_SNAPSHOT="${PROJECT_ROOT}/${LIVE_SNAPSHOT}"
  fi
  if [[ ! -f "${LIVE_SNAPSHOT}" ]]; then
    echo "[ERROR] LIVE_SNAPSHOT not found: ${LIVE_SNAPSHOT}" >&2
    exit 4
  fi
  echo "[INFO] 使用已有快照: ${LIVE_SNAPSHOT}"
elif [[ "${SKIP_SNAPSHOT}" == "1" || "${SKIP_SNAPSHOT}" == "true" ]]; then
  echo "[ERROR] SKIP_SNAPSHOT=1 但未提供 LIVE_SNAPSHOT" >&2
  exit 4
else
  STAMP="$(date +%H%M%S)"
  SNAP_DIR="${PROJECT_ROOT}/workspace/decision_packs/intraday/${AS_OF_TAG}/${STAMP}"
  mkdir -p "${SNAP_DIR}"
  LIVE_SNAPSHOT="${SNAP_DIR}/live_snapshot.csv"
  SNAP_ARGS=(
    -m decision_pack.src.regime_transition_live_snapshot
    --as-of "${AS_OF}"
    --output "${LIVE_SNAPSHOT}"
  )
  if [[ "${ALLOW_STALE}" == "1" || "${ALLOW_STALE}" == "true" || "${AS_OF_TAG}" != "${TODAY_TAG}" ]]; then
    if [[ "${AS_OF_TAG}" != "${TODAY_TAG}" && "${ALLOW_STALE}" != "1" && "${ALLOW_STALE}" != "true" ]]; then
      echo "[WARN] AS_OF=${AS_OF} ≠ 今天；自动加 --allow-stale-as-of（回放）。正式日中请用当天日期。" >&2
    fi
    SNAP_ARGS+=(--allow-stale-as-of)
  fi
  echo "[INFO] 抓取 live snapshot → ${LIVE_SNAPSHOT}"
  "${PY}" "${SNAP_ARGS[@]}"
fi

export AS_OF PLOT_START TRAIN_CUTOFF HTML_OUT_DIR LIVE_SNAPSHOT PY RETRAIN PLOTLY_ROOT
export CODES MAX_CODES
echo "[INFO] AS_OF=${AS_OF} PLOT_START=${PLOT_START} TRAIN_CUTOFF=${TRAIN_CUTOFF}"
echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR}"
echo "[INFO] LIVE_SNAPSHOT=${LIVE_SNAPSHOT}"
"${PROJECT_ROOT}/scripts/daily_adaptive_stage_html.sh"
echo "[OK] 日中 adaptive HTML: ${HTML_OUT_DIR}"
echo "[OK] snapshot: ${LIVE_SNAPSHOT}"
