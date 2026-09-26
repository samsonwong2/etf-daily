# Sourced by the four daily shells. One place for the repo root and local config.
# Missing config.env keeps the public defaults. scripts/accept_eod.sh refuses that case.
# HTML follows PLOTLY_ROOT. Regime shards are symlinks, not variables in this file.
_DAILY_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_DAILY_ENV_DIR}/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/config.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/config.env"
  set +a
fi
PLOTLY_ROOT="${PLOTLY_ROOT:-${HOME}/etf-daily-output/plotly_outputs}"
HRP_OUTPUT_DIR="${HRP_OUTPUT_DIR:-${HOME}/etf-daily-output/decision_packs}"
PY="${PY:-python3}"
if [[ -z "${HRP_MEMBERSHIP_CSV:-}" ]]; then
  unset HRP_MEMBERSHIP_CSV || true
fi
cd "${PROJECT_ROOT}"
