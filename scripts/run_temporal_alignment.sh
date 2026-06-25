#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${ROOT_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/results/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" src/estimate_temporal_offset_ncc.py \
  --odom-in data/Odom.jsonl \
  --gt-in data/pose_GT_by_mocap.jsonl \
  --odom-out data/Odom_temporal_aligned.jsonl \
  --gt-out data/pose_GT_by_mocap_temporal_aligned.jsonl \
  --summary-dir results/summaries \
  --diagnostics-dir results/diagnostics
