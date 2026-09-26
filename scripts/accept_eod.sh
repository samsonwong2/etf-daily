#!/usr/bin/env bash
# Run the eight end-of-day commands in handbook order. Both dates are required.
# Missing config.env or the local json exits before any Python process starts.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_HERE}/.." && pwd)"
AS_OF=""
NEXT_DAY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --as-of)
      AS_OF="${2:-}"
      shift 2
      ;;
    --next-day)
      NEXT_DAY="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: scripts/accept_eod.sh --as-of YYYY-MM-DD --next-day YYYY-MM-DD" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${AS_OF}" || -z "${NEXT_DAY}" ]]; then
  echo "usage: scripts/accept_eod.sh --as-of YYYY-MM-DD --next-day YYYY-MM-DD" >&2
  exit 2
fi
if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "AS_OF must be a real YYYY-MM-DD date, got: ${AS_OF}" >&2
  exit 2
fi
if [[ ! "${NEXT_DAY}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "${NEXT_DAY}" >/dev/null 2>&1; then
  echo "NEXT_DAY must be a real YYYY-MM-DD date, got: ${NEXT_DAY}" >&2
  exit 2
fi

CONFIG_ENV="${PROJECT_ROOT}/config.env"
LOCAL_JSON="${PROJECT_ROOT}/configs/production_regime_switch_ewma_shrink.json"
if [[ ! -f "${CONFIG_ENV}" || ! -f "${LOCAL_JSON}" ]]; then
  echo "cp config.env.example config.env" >&2
  echo "cp configs/production_regime_switch_ewma_shrink.json.example configs/production_regime_switch_ewma_shrink.json" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${_HERE}/daily_env.sh"
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"
: "${TEMP_DIR:?Set TEMP_DIR in config.env to the same directory as paths.temp_dir}"
: "${PLOTLY_ROOT:?Set PLOTLY_ROOT in config.env}"
: "${HRP_OUTPUT_DIR:?Set HRP_OUTPUT_DIR in config.env}"

echo "accept 1/8 etl"
"$PY" run.py etl

echo "accept 2/8 pool"
"$PY" run.py pool

echo "accept 3/8 cluster review"
"$PY" src/etf_daily/pool/每日审查cluster_mapping_最短命令清单.py \
  --future-end "${AS_OF}" \
  --selected-csv "${TEMP_DIR}/cluster_mapping_selected.csv" \
  --mapping-csv "${TEMP_DIR}/cluster_mapping.csv"

echo "accept 4/8 regime validation"
AS_OF="${AS_OF}" JOBS=8 PY="${PY}" SKIP_REBUILD=1 SKIP_HTML=1 \
  ./scripts/daily_regime_transition_validation.sh

echo "accept 5/8 adaptive html"
AS_OF="${AS_OF}" JOBS=8 PY="${PY}" \
  ./scripts/daily_adaptive_stage_html.sh

echo "accept 6/8 from_listing"
AS_OF="${AS_OF}" JOBS=8 PY="${PY}" \
  ./scripts/daily_adaptive_from_listing.sh

echo "accept 7/8 hrp"
"$PY" src/etf_daily/hrp/dendrogram.py \
  --lookback-days 252 \
  --asof-date "${AS_OF}" \
  --dist-t 0.8 \
  --output "${HRP_OUTPUT_DIR}/${AS_OF_TAG}/hrp_dendrogram_${AS_OF_TAG}_d080.html"

echo "accept 8/8 next-day triggers"
PYTHONPATH="${PROJECT_ROOT}/src" "$PY" src/etf_daily/triggers/scan.py \
  --listing-dir "${PLOTLY_ROOT}/${AS_OF_TAG}_from_listing" \
  --as-of "${AS_OF}" \
  --next-day "${NEXT_DAY}" \
  --jobs 8
