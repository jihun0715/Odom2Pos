#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${ROOT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" src/export_jsonl_odometry.py \
  --odom-in data/Odom.tum \
  --gt-in data/pose_GT_by_mocap.tum \
  --odom-out data/Odom.jsonl \
  --gt-out data/pose_GT_by_mocap.jsonl
