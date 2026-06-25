#!/usr/bin/env python3
"""Create comparison plots for two TUM-format trajectory files."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TUM_COLUMNS = ("timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw")


@dataclass(frozen=True)
class Trajectory:
    name: str
    path: Path
    data: np.ndarray
    unique: np.ndarray
    yaw_unwrapped: np.ndarray
    yaw_unique_unwrapped: np.ndarray


def load_tum(path: Path) -> np.ndarray:
    """Load a TUM trajectory file as [timestamp, tx, ty, tz, qx, qy, qz, qw]."""
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
    """Return yaw in radians from TUM quaternions ordered as qx, qy, qz, qw."""
    qx = quat_xyzw[:, 0]
    qy = quat_xyzw[:, 1]
    qz = quat_xyzw[:, 2]
    qw = quat_xyzw[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def build_trajectory(name: str, path: Path) -> Trajectory:
    data = load_tum(path)
    unique = unique_by_timestamp(data)
    return Trajectory(
        name=name,
        path=path,
        data=data,
        unique=unique,
        yaw_unwrapped=np.unwrap(yaw_from_quaternion(data[:, 4:8])),
        yaw_unique_unwrapped=np.unwrap(yaw_from_quaternion(unique[:, 4:8])),
    )


def downsample_rows(data: np.ndarray, max_points: int) -> np.ndarray:
    if data.shape[0] <= max_points:
        return data
    step = int(math.ceil(data.shape[0] / max_points))
    return data[::step]


def downsample_series(
    t: np.ndarray, values: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if t.shape[0] <= max_points:
        return t, values
    step = int(math.ceil(t.shape[0] / max_points))
    return t[::step], values[::step]


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_xy(odom: Trajectory, gt: Trajectory, out_path: Path, max_points: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for traj, color in ((odom, "#1f77b4"), (gt, "#d62728")):
        sampled = downsample_rows(traj.data, max_points)
        ax.plot(sampled[:, 1], sampled[:, 2], linewidth=1.4, color=color, label=traj.name)
        ax.scatter(traj.data[0, 1], traj.data[0, 2], marker="o", s=45, color=color)
        ax.scatter(traj.data[-1, 1], traj.data[-1, 2], marker="x", s=55, color=color)
    style_axis(ax, "XY trajectory", "x [m]", "y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    save_figure(fig, out_path)


def plot_position_time(
    odom: Trajectory, gt: Trajectory, out_path: Path, max_points: int
) -> None:
    t0 = min(odom.unique[0, 0], gt.unique[0, 0])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    labels = ("x [m]", "y [m]", "z [m]")
    for axis_index, ax in enumerate(axes, start=1):
        for traj, color in ((odom, "#1f77b4"), (gt, "#d62728")):
            t_rel = traj.unique[:, 0] - t0
            t_plot, v_plot = downsample_series(t_rel, traj.unique[:, axis_index], max_points)
            ax.plot(t_plot, v_plot, linewidth=1.1, color=color, label=traj.name)
        style_axis(ax, f"{labels[axis_index - 1]} over time", "", labels[axis_index - 1])
        ax.legend(loc="best")
    axes[-1].set_xlabel("time from earliest timestamp [s]")
    save_figure(fig, out_path)


def plot_yaw_time(
    odom: Trajectory, gt: Trajectory, out_path: Path, max_points: int
) -> None:
    t0 = min(odom.unique[0, 0], gt.unique[0, 0])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for traj, color in ((odom, "#1f77b4"), (gt, "#d62728")):
        t_rel = traj.unique[:, 0] - t0
        yaw_deg = np.rad2deg(traj.yaw_unique_unwrapped)
        t_plot, yaw_plot = downsample_series(t_rel, yaw_deg, max_points)
        ax.plot(t_plot, yaw_plot, linewidth=1.1, color=color, label=traj.name)
    style_axis(ax, "Unwrapped yaw over time", "time from earliest timestamp [s]", "yaw [deg]")
    ax.legend(loc="best")
    save_figure(fig, out_path)


def plot_sampling(
    trajectories: Iterable[Trajectory], out_path: Path, max_points: int
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for traj, color in zip(trajectories, ("#1f77b4", "#d62728"), strict=True):
        t = traj.unique[:, 0]
        dt = np.diff(t)
        if dt.size == 0:
            continue
        t_rel = t[1:] - t[0]
        t_plot, dt_plot = downsample_series(t_rel, dt, max_points)
        axes[0].plot(t_plot, dt_plot * 1000.0, linewidth=0.9, color=color, label=traj.name)
        axes[1].hist(
            dt * 1000.0,
            bins=60,
            alpha=0.45,
            color=color,
            label=traj.name,
            log=True,
        )
    style_axis(axes[0], "Sampling interval over time", "time from each start [s]", "dt [ms]")
    style_axis(axes[1], "Sampling interval histogram", "dt [ms]", "count")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    save_figure(fig, out_path)


def plot_aligned_time(
    odom: Trajectory, gt: Trajectory, out_path: Path, max_points: int
) -> bool:
    common_start = max(odom.unique[0, 0], gt.unique[0, 0])
    common_end = min(odom.unique[-1, 0], gt.unique[-1, 0])
    if common_start >= common_end:
        return False

    gt_mask = (gt.unique[:, 0] >= common_start) & (gt.unique[:, 0] <= common_end)
    target_t = gt.unique[gt_mask, 0]
    if target_t.shape[0] < 2:
        return False

    odom_interp = np.column_stack(
        [np.interp(target_t, odom.unique[:, 0], odom.unique[:, col]) for col in (1, 2, 3)]
    )
    gt_xyz = gt.unique[gt_mask, 1:4]
    odom_yaw = np.interp(target_t, odom.unique[:, 0], odom.yaw_unique_unwrapped)
    gt_yaw = gt.yaw_unique_unwrapped[gt_mask]

    t_rel = target_t - common_start
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    labels = ("x [m]", "y [m]", "z [m]", "yaw [deg]")
    series = (
        (odom_interp[:, 0], gt_xyz[:, 0]),
        (odom_interp[:, 1], gt_xyz[:, 1]),
        (odom_interp[:, 2], gt_xyz[:, 2]),
        (np.rad2deg(odom_yaw), np.rad2deg(gt_yaw)),
    )
    for ax, label, (odom_values, gt_values) in zip(axes, labels, series, strict=True):
        t_plot, odom_plot = downsample_series(t_rel, odom_values, max_points)
        _, gt_plot = downsample_series(t_rel, gt_values, max_points)
        ax.plot(t_plot, odom_plot, linewidth=1.0, color="#1f77b4", label=f"{odom.name} interp.")
        ax.plot(t_plot, gt_plot, linewidth=1.0, color="#d62728", label=gt.name)
        style_axis(ax, f"Timestamp-aligned {label}", "", label)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time from common start [s]")
    save_figure(fig, out_path)
    return True


def trajectory_stats(traj: Trajectory) -> dict[str, object]:
    timestamps = traj.data[:, 0]
    unique_timestamps = traj.unique[:, 0]
    duration = float(timestamps[-1] - timestamps[0])
    dt = np.diff(unique_timestamps)
    quat_norm = np.linalg.norm(traj.data[:, 4:8], axis=1)
    return {
        "path": str(traj.path),
        "rows": int(traj.data.shape[0]),
        "unique_timestamps": int(traj.unique.shape[0]),
        "duplicate_rows": int(traj.data.shape[0] - traj.unique.shape[0]),
        "start_timestamp": float(timestamps[0]),
        "end_timestamp": float(timestamps[-1]),
        "duration_sec": duration,
        "mean_unique_rate_hz": float((traj.unique.shape[0] - 1) / duration) if duration > 0 else None,
        "median_unique_dt_ms": float(np.median(dt) * 1000.0) if dt.size else None,
        "min_xyz": traj.data[:, 1:4].min(axis=0).tolist(),
        "max_xyz": traj.data[:, 1:4].max(axis=0).tolist(),
        "normalized_quaternion_norm_min": float(quat_norm.min()),
        "normalized_quaternion_norm_max": float(quat_norm.max()),
    }


def write_summary(odom: Trajectory, gt: Trajectory, out_dir: Path) -> None:
    common_start = max(odom.unique[0, 0], gt.unique[0, 0])
    common_end = min(odom.unique[-1, 0], gt.unique[-1, 0])
    summary = {
        "format": "TUM trajectory: timestamp tx ty tz qx qy qz qw",
        "trajectories": {
            odom.name: trajectory_stats(odom),
            gt.name: trajectory_stats(gt),
        },
        "common_time_window": {
            "start_timestamp": float(common_start),
            "end_timestamp": float(common_end),
            "duration_sec": float(max(0.0, common_end - common_start)),
        },
    }
    json_path = out_dir / "tum_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "TUM trajectory summary",
        "======================",
        "",
        "Format: timestamp tx ty tz qx qy qz qw",
        "",
    ]
    for name, stats in summary["trajectories"].items():
        lines.extend(
            [
                f"{name}",
                f"  path: {stats['path']}",
                f"  rows: {stats['rows']}",
                f"  unique timestamps: {stats['unique_timestamps']}",
                f"  duplicate rows: {stats['duplicate_rows']}",
                f"  timestamp range: {stats['start_timestamp']:.9f} - {stats['end_timestamp']:.9f}",
                f"  duration: {stats['duration_sec']:.3f} sec",
                f"  median dt: {stats['median_unique_dt_ms']:.3f} ms",
                "",
            ]
        )
    lines.extend(
        [
            "Common time window",
            f"  start: {common_start:.9f}",
            f"  end: {common_end:.9f}",
            f"  duration: {max(0.0, common_end - common_start):.3f} sec",
            "",
        ]
    )
    (out_dir / "tum_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize odometry and mocap ground-truth TUM trajectory files."
    )
    parser.add_argument("--odom", type=Path, default=Path("data/Odom.tum"))
    parser.add_argument("--gt", type=Path, default=Path("data/pose_GT_by_mocap.tum"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-points", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    odom = build_trajectory("Odom", args.odom)
    gt = build_trajectory("pose_GT_by_mocap", args.gt)

    outputs = [
        args.out_dir / "trajectory_xy.png",
        args.out_dir / "position_vs_time.png",
        args.out_dir / "yaw_vs_time.png",
        args.out_dir / "sampling_intervals.png",
    ]
    plot_xy(odom, gt, outputs[0], args.max_points)
    plot_position_time(odom, gt, outputs[1], args.max_points)
    plot_yaw_time(odom, gt, outputs[2], args.max_points)
    plot_sampling((odom, gt), outputs[3], args.max_points)

    aligned_path = args.out_dir / "aligned_time_comparison.png"
    if plot_aligned_time(odom, gt, aligned_path, args.max_points):
        outputs.append(aligned_path)

    write_summary(odom, gt, args.out_dir)
    outputs.extend([args.out_dir / "tum_summary.json", args.out_dir / "tum_summary.txt"])

    print("Generated:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
