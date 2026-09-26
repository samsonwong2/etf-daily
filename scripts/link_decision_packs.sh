#!/usr/bin/env bash
# Create the two regime-shard symlinks. Pass the old decision_packs directory
# as the first argument or as OLD_DECISION_PACK_ROOT. Does not link plotly or temp.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_HERE}/.." && pwd)"
OLD_ROOT="${1:-${OLD_DECISION_PACK_ROOT:-}}"
LINK_DEST="${LINK_DEST:-${PROJECT_ROOT}/runtime/decision_packs}"
NAMES=(20260720 regime_transition_model_cache)

if [[ -z "${OLD_ROOT}" ]]; then
  echo "usage: OLD_DECISION_PACK_ROOT=/path/to/decision_packs $0" >&2
  exit 2
fi

mkdir -p "${LINK_DEST}"
for name in "${NAMES[@]}"; do
  target="${OLD_ROOT}/${name}"
  link="${LINK_DEST}/${name}"
  if [[ ! -d "${target}" ]]; then
    echo "missing target: ${target}" >&2
    exit 1
  fi
  if [[ -e "${link}" && ! -L "${link}" ]]; then
    echo "refusing real directory: ${link}" >&2
    exit 1
  fi
  ln -sfn "${target}" "${link}"
done
