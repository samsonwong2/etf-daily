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
_SRC="${PROJECT_ROOT}/src"
case ":${PYTHONPATH:-}:" in
  *":${_SRC}:"*) ;;
  *) PYTHONPATH="${_SRC}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export PYTHONPATH
if [[ -z "${HRP_MEMBERSHIP_CSV:-}" ]]; then
  unset HRP_MEMBERSHIP_CSV || true
fi
# Latest intraday pack for one day: both the signal file and the frozen snapshot.
# Stamp directories are HHMMSS names; the newest name is the newest capture.
latest_intraday_pack() {
  local day_dir="$1"
  local best="" d base
  [[ -d "${day_dir}" ]] || return 0
  shopt -s nullglob
  for d in "${day_dir}"/*; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/live_snapshot.csv" && -f "${d}/signals_oos.csv" ]] || continue
    base="$(basename "${d}")"
    if [[ -z "${best}" || "${base}" > "$(basename "${best}")" ]]; then
      best="${d}"
    fi
  done
  shopt -u nullglob
  printf '%s\n' "${best}"
}

cd "${PROJECT_ROOT}"
