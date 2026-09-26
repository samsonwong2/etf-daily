#!/usr/bin/env bash
# Machine checks for daily_env, accept_eod, and the link script. No qlib.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash -n scripts/daily_env.sh
bash -n scripts/accept_eod.sh
bash -n scripts/link_decision_packs.sh
for shell in \
  scripts/daily_regime_transition_validation.sh \
  scripts/daily_adaptive_stage_html.sh \
  scripts/daily_adaptive_from_listing.sh \
  scripts/daily_adaptive_stage_html_intraday.sh
do
  bash -n "${shell}"
  grep -q 'daily_env.sh' "${shell}"
done

# The membership flag is appended only inside the non-empty check.
if grep -n -- '--hrp-membership-csv' scripts/daily_adaptive_stage_html.sh | grep -v 'ARGS+=' | grep -v ':#'; then
  echo "stage shell passes --hrp-membership-csv outside the empty check" >&2
  exit 1
fi

# Empty membership omits the flag. Checked below with a temporary config.env,
# so a real local config.env does not leak into this assertion.

cfg="${ROOT}/config.env"
json="${ROOT}/configs/production_regime_switch_ewma_shrink.json"
cfg_bak=""
json_bak=""
restore() {
  if [[ -n "${cfg_bak}" ]]; then
    mv "${cfg_bak}" "${cfg}"
  else
    rm -f "${cfg}"
  fi
  if [[ -n "${json_bak}" ]]; then
    mv "${json_bak}" "${json}"
  fi
}
if [[ -f "${cfg}" ]]; then
  cfg_bak="$(mktemp)"
  mv "${cfg}" "${cfg_bak}"
fi
if [[ -f "${json}" ]]; then
  json_bak="$(mktemp)"
  mv "${json}" "${json_bak}"
fi
trap restore EXIT

# Public defaults when config.env is absent.
(
  unset PY PLOTLY_ROOT HRP_OUTPUT_DIR HRP_MEMBERSHIP_CSV
  # shellcheck disable=SC1091
  source scripts/daily_env.sh
  [[ "${PY}" == "python3" ]]
  [[ "${PLOTLY_ROOT}" == "${HOME}/etf-daily-output/plotly_outputs" ]]
  [[ "${HRP_OUTPUT_DIR}" == "${HOME}/etf-daily-output/decision_packs" ]]
  [[ -z "${HRP_MEMBERSHIP_CSV+x}" ]]
)

set +e
PY=/bin/false ./scripts/accept_eod.sh >/tmp/etf-accept-out 2>/tmp/etf-accept-err
code=$?
set -e
if [[ "${code}" -eq 0 ]]; then
  echo "accept_eod without dates exited 0" >&2
  exit 1
fi

set +e
PY=/bin/false ./scripts/accept_eod.sh --as-of 2026-09-23 --next-day 2026-09-24 \
  >/tmp/etf-accept-out 2>/tmp/etf-accept-err
code=$?
set -e
if [[ "${code}" -ne 2 ]]; then
  echo "accept_eod missing config exited ${code}, want 2" >&2
  cat /tmp/etf-accept-err >&2
  exit 1
fi
grep -q 'cp config.env.example config.env' /tmp/etf-accept-err
grep -q 'cp configs/production_regime_switch_ewma_shrink.json.example' /tmp/etf-accept-err

tmp="$(mktemp -d)"
mkdir -p "${tmp}/src/20260720" "${tmp}/src/regime_transition_model_cache" "${tmp}/dest"
if OLD_DECISION_PACK_ROOT="${tmp}/missing" LINK_DEST="${tmp}/dest" \
  ./scripts/link_decision_packs.sh >/tmp/etf-link-out 2>/tmp/etf-link-err
then
  echo "link script accepted a missing target" >&2
  exit 1
fi
grep -q 'missing target:' /tmp/etf-link-err

mkdir -p "${tmp}/real/20260720"
if OLD_DECISION_PACK_ROOT="${tmp}/src" LINK_DEST="${tmp}/real" \
  ./scripts/link_decision_packs.sh >/tmp/etf-link-out 2>/tmp/etf-link-err
then
  echo "link script replaced a real directory" >&2
  exit 1
fi
grep -q 'refusing real directory:' /tmp/etf-link-err

ok="${tmp}/ok"
mkdir -p "${ok}"
OLD_DECISION_PACK_ROOT="${tmp}/src" LINK_DEST="${ok}" ./scripts/link_decision_packs.sh
test -L "${ok}/20260720"
test -d "${ok}/20260720"
test -L "${ok}/regime_transition_model_cache"
test -d "${ok}/regime_transition_model_cache"

# A present config.env is sourced. An empty membership value is unset and adds no flag.
printf 'PLOTLY_ROOT=/tmp/etf-daily-sourced-plotly\nHRP_MEMBERSHIP_CSV=\n' > "${cfg}"
(
  # shellcheck disable=SC1091
  source scripts/daily_env.sh
  [[ "${PLOTLY_ROOT}" == "/tmp/etf-daily-sourced-plotly" ]]
  [[ -z "${HRP_MEMBERSHIP_CSV+x}" ]]
  args=()
  if [[ -n "${HRP_MEMBERSHIP_CSV:-}" ]]; then
    args+=(--hrp-membership-csv "${HRP_MEMBERSHIP_CSV}")
  fi
  [[ ${#args[@]} -eq 0 ]]
)
rm -f "${cfg}"

echo "daily_env checks passed"
