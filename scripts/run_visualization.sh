#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${ROOT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/results/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/results/.cache}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" src/visualize_tum.py \
  --odom data/Odom.tum \
  --gt data/pose_GT_by_mocap.tum \
  --out-dir results
