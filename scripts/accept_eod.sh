#!/usr/bin/env bash
# Run the end-of-day commands in handbook order, then the morning receipt.
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
LOCAL_JSON="${PROJECT_ROOT}/configs/paths.json"
LEGACY_JSON="${PROJECT_ROOT}/configs/production_regime_switch_ewma_shrink.json"
if [[ ! -f "${CONFIG_ENV}" || ( ! -f "${LOCAL_JSON}" && ! -f "${LEGACY_JSON}" ) ]]; then
  echo "cp config.env.example config.env" >&2
  echo "cp configs/paths.example.json configs/paths.json" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${_HERE}/daily_env.sh"
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"
: "${TEMP_DIR:?Set TEMP_DIR in config.env to the same directory as paths.temp_dir}"
: "${PLOTLY_ROOT:?Set PLOTLY_ROOT in config.env}"
: "${HRP_OUTPUT_DIR:?Set HRP_OUTPUT_DIR in config.env}"

echo "accept 1/8 etl"
"$PY" -m etf_daily etl

echo "accept 2/8 pool"
"$PY" -m etf_daily pool

echo "accept 3/8 cluster review"
"$PY" -m etf_daily cluster-review \
  --future-end "${AS_OF}" \
  --selected-csv "${TEMP_DIR}/cluster_mapping_selected.csv" \
  --mapping-csv "${TEMP_DIR}/cluster_mapping.csv"

echo "accept 4/8 regime validation"
"$PY" -m etf_daily regime --as-of "${AS_OF}" --skip-rebuild --skip-html

echo "accept 5/8 from_listing"
"$PY" -m etf_daily listing --as-of "${AS_OF}"

echo "accept 6/8 hrp"
"$PY" -m etf_daily hrp --dist-t 0.8 --as-of "${AS_OF}"

echo "accept 7/8 next-day triggers"
"$PY" -m etf_daily triggers \
  --listing-dir "${PLOTLY_ROOT}/${AS_OF_TAG}_from_listing" \
  --as-of "${AS_OF}" \
  --next-day "${NEXT_DAY}" \
  --jobs 8

echo "accept 8/8 morning receipt"
"$PY" -m etf_daily morning --as-of "${AS_OF}" --next-day "${NEXT_DAY}"
