#!/usr/bin/env bash
# Run the eight close commands below, then stop. This is not the research list in README.md.
# That list also draws the default-distance HRP page and runs rules7. This script does neither.
# Missing config.env or the local json exits before any Python process starts.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_HERE}/.." && pwd)"
AS_OF=""
NEXT_DAY=""
PRINT_PLAN=0

usage() {
  cat >&2 <<'EOF'
usage: scripts/accept_eod.sh --as-of YYYY-MM-DD --next-day YYYY-MM-DD [--print-plan]

--next-day is the next trading session. This script does not know exchange holidays.
It does not rewrite --next-day to the next weekday. Pass the real session when that
session is later than the next weekday.

--print-plan prints the eight commands and exits. It does not start Python.
rules7 is not one of the eight. The default-distance HRP page is not one of the eight.
EOF
}

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
    --print-plan)
      PRINT_PLAN=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${AS_OF}" || -z "${NEXT_DAY}" ]]; then
  usage
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

# One ordered list. --print-plan and the live run both go through dispatch.
dispatch() {
  local n="$1"
  local label="$2"
  shift 2
  if [[ "${PRINT_PLAN}" == 1 ]]; then
    printf '%s etf-daily' "${n}"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  echo "accept ${n}/8 ${label}"
  "$PY" -m etf_daily "$@"
}

dispatch 1 etl etl
dispatch 2 pool pool
dispatch 3 "cluster review" cluster-review \
  --future-end "${AS_OF}" \
  --selected-csv "${TEMP_DIR}/cluster_mapping_selected.csv" \
  --mapping-csv "${TEMP_DIR}/cluster_mapping.csv"
dispatch 4 "regime validation" regime --as-of "${AS_OF}" --skip-rebuild --skip-html
dispatch 5 from_listing listing --as-of "${AS_OF}"
dispatch 6 hrp hrp --dist-t 0.8 --as-of "${AS_OF}"
dispatch 7 "next-day triggers" triggers \
  --listing-dir "${PLOTLY_ROOT}/${AS_OF_TAG}_from_listing" \
  --as-of "${AS_OF}" \
  --next-day "${NEXT_DAY}" \
  --jobs 8
dispatch 8 "morning receipt" morning --as-of "${AS_OF}" --next-day "${NEXT_DAY}"

if [[ "${PRINT_PLAN}" == 1 ]]; then
  exit 0
fi
