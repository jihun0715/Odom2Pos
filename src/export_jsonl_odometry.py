#!/usr/bin/env python3
"""Export TUM trajectories as first-frame 2D velocity JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TUM_COLUMNS = ("timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw")
ZERO_EPSILON = 1e-12


def load_tum(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(TUM_COLUMNS):
        raise ValueError(
            f"{path} must have {len(TUM_COLUMNS)} columns "
            f"({', '.join(TUM_COLUMNS)}), but has {data.shape[1]}."
        )

    finite_mask = np.isfinite(data).all(axis=1)
    data = data[finite_mask].copy()
    if data.size == 0:
        raise ValueError(f"{path} has no finite trajectory rows.")

    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]

    quat = data[:, 4:8]
    quat_norm = np.linalg.norm(quat, axis=1)
    valid_quat = quat_norm > 0.0
    data = data[valid_quat]
    quat_norm = quat_norm[valid_quat]
    data[:, 4:8] = data[:, 4:8] / quat_norm[:, None]
    return data


def unique_by_timestamp(data: np.ndarray) -> np.ndarray:
    """Keep the last row for each duplicate timestamp."""
    reversed_timestamps = data[::-1, 0]
    _, reversed_indices = np.unique(reversed_timestamps, return_index=True)
    keep_indices = data.shape[0] - 1 - reversed_indices
    keep_indices.sort()
    return data[keep_indices]


def yaw_from_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    qx = quat_xyzw[:, 0]
    qy = quat_xyzw[:, 1]
    qz = quat_xyzw[:, 2]
    qw = quat_xyzw[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def to_first_frame_pose(data: np.ndarray) -> np.ndarray:
    """Return [timestamp, x, y, theta] in the first pose frame."""
    xy = data[:, 1:3]
    yaw = np.unwrap(yaw_from_quaternion(data[:, 4:8]))

    origin_xy = xy[0]
    origin_yaw = yaw[0]
    delta_xy = xy - origin_xy

    c = np.cos(origin_yaw)
    s = np.sin(origin_yaw)
    first_frame_x = c * delta_xy[:, 0] + s * delta_xy[:, 1]
    first_frame_y = -s * delta_xy[:, 0] + c * delta_xy[:, 1]
    first_frame_theta = yaw - origin_yaw

    rows = np.column_stack((data[:, 0], first_frame_x, first_frame_y, first_frame_theta))
    rows[:, 1:4][np.abs(rows[:, 1:4]) < ZERO_EPSILON] = 0.0
    return rows


def to_first_frame_velocity(data: np.ndarray) -> np.ndarray:
    """Return [timestamp, vx, vy, vtheta] in the first pose frame."""
    unique_pose = to_first_frame_pose(unique_by_timestamp(data))
    if unique_pose.shape[0] < 2:
        raise ValueError("At least two unique timestamps are required to compute velocity.")

    t_unique = unique_pose[:, 0]
    vx_unique = np.gradient(unique_pose[:, 1], t_unique)
    vy_unique = np.gradient(unique_pose[:, 2], t_unique)
    vtheta_unique = np.gradient(unique_pose[:, 3], t_unique)

    timestamps = data[:, 0]
    rows = np.column_stack(
        (
            timestamps,
            np.interp(timestamps, t_unique, vx_unique),
            np.interp(timestamps, t_unique, vy_unique),
            np.interp(timestamps, t_unique, vtheta_unique),
        )
    )
    rows[:, 1:4][np.abs(rows[:, 1:4]) < ZERO_EPSILON] = 0.0
    return rows


def write_jsonl(rows: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for timestamp, vx, vy, vtheta in rows:
            record = {
                "timestamp": float(timestamp),
                "vx": float(vx),
                "vy": float(vy),
                "vtheta": float(vtheta),
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def export_one(in_path: Path, out_path: Path) -> int:
    data = load_tum(in_path)
    rows = to_first_frame_velocity(data)
    write_jsonl(rows, out_path)
    return rows.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TUM trajectories to JSONL with timestamp, vx, vy, "
            "and vtheta in each trajectory's first-pose frame."
        )
    )
    parser.add_argument("--odom-in", type=Path, default=Path("data/Odom.tum"))
    parser.add_argument("--gt-in", type=Path, default=Path("data/pose_GT_by_mocap.tum"))
    parser.add_argument("--odom-out", type=Path, default=Path("data/Odom.jsonl"))
    parser.add_argument(
        "--gt-out", type=Path, default=Path("data/pose_GT_by_mocap.jsonl")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (
        (args.odom_in, args.odom_out),
        (args.gt_in, args.gt_out),
    )
    for in_path, out_path in outputs:
        row_count = export_one(in_path, out_path)
        print(f"Wrote {row_count} rows: {out_path}")


if __name__ == "__main__":
    main()
