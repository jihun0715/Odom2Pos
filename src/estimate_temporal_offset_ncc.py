#!/usr/bin/env python3
"""Estimate and apply temporal offset between two odometry JSONL trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


JSONL_COLUMNS = ("timestamp", "odom_x", "odom_y", "odom_theta")
FEATURE_CHANNELS = ("speed", "omega")
STD_EPSILON = 1e-9
ZERO_EPSILON = 1e-12


@dataclass(frozen=True)
class Trajectory2D:
    name: str
    path: Path
    data: np.ndarray
    duplicate_rows_removed: int

    @property
    def t(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def x(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def y(self) -> np.ndarray:
        return self.data[:, 2]

    @property
    def theta(self) -> np.ndarray:
        return self.data[:, 3]


@dataclass(frozen=True)
class FeatureSeries:
    t: np.ndarray
    speed: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class NccResult:
    lags: np.ndarray
    scores: np.ndarray
    overlap_samples: np.ndarray
    channel_counts: np.ndarray
    best_lag_sec: float
    best_score: float
    best_overlap_samples: int
    is_boundary_peak: bool


def load_jsonl_trajectory(path: Path, name: str) -> Trajectory2D:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                rows.append([float(record[key]) for key in JSONL_COLUMNS])
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing key {exc.args[0]!r}") from exc

    if not rows:
        raise ValueError(f"{path} has no JSONL rows.")

    data = np.asarray(rows, dtype=float)
    finite_mask = np.isfinite(data).all(axis=1)
    data = data[finite_mask]
    if data.size == 0:
        raise ValueError(f"{path} has no finite trajectory rows.")

    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    before = data.shape[0]
    data = unique_by_timestamp(data)
    return Trajectory2D(
        name=name,
        path=path,
        data=data,
        duplicate_rows_removed=before - data.shape[0],
    )


def unique_by_timestamp(data: np.ndarray) -> np.ndarray:
    reversed_timestamps = data[::-1, 0]
    _, reversed_indices = np.unique(reversed_timestamps, return_index=True)
    keep_indices = data.shape[0] - 1 - reversed_indices
    keep_indices.sort()
    return data[keep_indices]


def median_dt(traj: Trajectory2D) -> float:
    dt = np.diff(traj.t)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError(f"{traj.path} has fewer than two unique timestamps.")
    return float(np.median(dt))


def make_uniform_time(start: float, end: float, dt: float) -> np.ndarray:
    count = int(np.floor((end - start) / dt)) + 1
    if count < 2:
        raise ValueError("Uniform grid would contain fewer than two samples.")
    return start + np.arange(count, dtype=float) * dt


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values
    if window_samples % 2 == 0:
        window_samples += 1
    kernel = np.ones(window_samples, dtype=float) / window_samples
    pad = window_samples // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_feature_series(traj: Trajectory2D, dt: float, smooth_sec: float) -> FeatureSeries:
    t_grid = make_uniform_time(traj.t[0], traj.t[-1], dt)
    x = np.interp(t_grid, traj.t, traj.x)
    y = np.interp(t_grid, traj.t, traj.y)
    theta = np.interp(t_grid, traj.t, traj.theta)

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.sqrt(vx * vx + vy * vy)
    omega = np.gradient(theta, dt)

    window_samples = max(1, int(round(smooth_sec / dt)))
    speed = moving_average(speed, window_samples)
    omega = moving_average(omega, window_samples)
    return FeatureSeries(t=t_grid, speed=speed, omega=omega)


def normalized_dot(a: np.ndarray, b: np.ndarray) -> float | None:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    a_std = float(np.std(a_centered))
    b_std = float(np.std(b_centered))
    if a_std < STD_EPSILON or b_std < STD_EPSILON:
        return None
    return float(np.mean((a_centered / a_std) * (b_centered / b_std)))


def score_lag(
    odom_features: FeatureSeries,
    gt_features: FeatureSeries,
    lag_sec: float,
    dt: float,
    min_overlap_samples: int,
) -> tuple[float, int, int]:
    # Convention: gt_feature(t + lag_sec) ~= odom_feature(t).
    start = max(odom_features.t[0], gt_features.t[0] - lag_sec)
    end = min(odom_features.t[-1], gt_features.t[-1] - lag_sec)
    if end <= start:
        return math.nan, 0, 0

    t_eval = make_uniform_time(start, end, dt)
    if t_eval.shape[0] < min_overlap_samples:
        return math.nan, int(t_eval.shape[0]), 0

    channel_scores: list[float] = []
    for channel in FEATURE_CHANNELS:
        odom_values = np.interp(t_eval, odom_features.t, getattr(odom_features, channel))
        gt_values = np.interp(
            t_eval + lag_sec, gt_features.t, getattr(gt_features, channel)
        )
        score = normalized_dot(odom_values, gt_values)
        if score is not None:
            channel_scores.append(score)

    if not channel_scores:
        return math.nan, int(t_eval.shape[0]), 0
    return float(np.mean(channel_scores)), int(t_eval.shape[0]), len(channel_scores)


def refine_peak(lags: np.ndarray, scores: np.ndarray, best_index: int) -> float:
    if best_index <= 0 or best_index >= scores.shape[0] - 1:
        return float(lags[best_index])
    y_prev = scores[best_index - 1]
    y_mid = scores[best_index]
    y_next = scores[best_index + 1]
    if not np.isfinite([y_prev, y_mid, y_next]).all():
        return float(lags[best_index])

    denominator = y_prev - 2.0 * y_mid + y_next
    if abs(denominator) < STD_EPSILON:
        return float(lags[best_index])
    delta_index = 0.5 * (y_prev - y_next) / denominator
    if abs(delta_index) > 1.0:
        return float(lags[best_index])
    return float(lags[best_index] + delta_index * (lags[1] - lags[0]))


def estimate_offset(
    odom_features: FeatureSeries,
    gt_features: FeatureSeries,
    dt: float,
    max_lag_sec: float,
    min_overlap_sec: float,
) -> NccResult:
    lags = np.arange(-max_lag_sec, max_lag_sec + 0.5 * dt, dt, dtype=float)
    scores = np.full(lags.shape, math.nan, dtype=float)
    overlap_samples = np.zeros(lags.shape, dtype=int)
    channel_counts = np.zeros(lags.shape, dtype=int)
    min_overlap_samples = max(2, int(math.ceil(min_overlap_sec / dt)))

    for index, lag_sec in enumerate(lags):
        score, samples, channels = score_lag(
            odom_features, gt_features, float(lag_sec), dt, min_overlap_samples
        )
        scores[index] = score
        overlap_samples[index] = samples
        channel_counts[index] = channels

    if not np.isfinite(scores).any():
        raise ValueError("No valid NCC scores were computed. Check overlap and signal variance.")

    best_index = int(np.nanargmax(scores))
    refined_lag = refine_peak(lags, scores, best_index)
    refined_score, refined_samples, refined_channels = score_lag(
        odom_features, gt_features, refined_lag, dt, min_overlap_samples
    )
    if not np.isfinite(refined_score):
        refined_score = float(scores[best_index])
        refined_samples = int(overlap_samples[best_index])
        refined_channels = int(channel_counts[best_index])

    return NccResult(
        lags=lags,
        scores=scores,
        overlap_samples=overlap_samples,
        channel_counts=channel_counts,
        best_lag_sec=refined_lag,
        best_score=float(refined_score),
        best_overlap_samples=int(refined_samples),
        is_boundary_peak=best_index == 0 or best_index == scores.shape[0] - 1,
    )


def interp_pose(traj: Trajectory2D, query_t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            query_t,
            np.interp(query_t, traj.t, traj.x),
            np.interp(query_t, traj.t, traj.y),
            np.interp(query_t, traj.t, traj.theta),
        )
    )


def zero_small_values(rows: np.ndarray) -> np.ndarray:
    rows = rows.copy()
    rows[np.abs(rows) < ZERO_EPSILON] = 0.0
    return rows


def aligned_pose_rows(
    odom: Trajectory2D, gt: Trajectory2D, lag_sec: float, dt: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    start = max(odom.t[0], gt.t[0] - lag_sec)
    end = min(odom.t[-1], gt.t[-1] - lag_sec)
    if end <= start:
        raise ValueError("No overlapping trajectory window after temporal alignment.")

    odom_query_t = make_uniform_time(start, end, dt)
    gt_query_t = odom_query_t + lag_sec
    rel_t = odom_query_t - odom_query_t[0]

    odom_rows = interp_pose(odom, odom_query_t)
    gt_rows = interp_pose(gt, gt_query_t)
    odom_rows[:, 0] = rel_t
    gt_rows[:, 0] = rel_t
    return (
        zero_small_values(odom_rows),
        zero_small_values(gt_rows),
        {
            "odom_time_start": float(start),
            "odom_time_end": float(odom_query_t[-1]),
            "gt_time_start": float(gt_query_t[0]),
            "gt_time_end": float(gt_query_t[-1]),
            "duration_sec": float(rel_t[-1]) if rel_t.size else 0.0,
        },
    )


def write_jsonl(rows: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for timestamp, odom_x, odom_y, odom_theta in rows:
            record = {
                "timestamp": float(timestamp),
                "odom_x": float(odom_x),
                "odom_y": float(odom_y),
                "odom_theta": float(odom_theta),
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_summary(
    odom: Trajectory2D,
    gt: Trajectory2D,
    ncc: NccResult,
    dt: float,
    max_lag_sec: float,
    smooth_sec: float,
    min_overlap_sec: float,
    aligned_window: dict[str, float],
    odom_rows: np.ndarray,
    summary_dir: Path,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    common_start = max(odom.t[0], gt.t[0])
    common_end = min(odom.t[-1], gt.t[-1])
    valid_score_count = int(np.isfinite(ncc.scores).sum())
    summary = {
        "offset_convention": "gt_motion(t + gt_time_offset_sec) ~= odom_motion(t)",
        "gt_time_offset_sec": ncc.best_lag_sec,
        "ncc_peak_score": ncc.best_score,
        "is_boundary_peak": ncc.is_boundary_peak,
        "dt_sec": dt,
        "max_lag_sec": max_lag_sec,
        "smooth_sec": smooth_sec,
        "min_overlap_sec": min_overlap_sec,
        "channels": list(FEATURE_CHANNELS),
        "lag_count": int(ncc.lags.shape[0]),
        "valid_score_count": valid_score_count,
        "best_overlap_samples": ncc.best_overlap_samples,
        "best_overlap_sec": float(ncc.best_overlap_samples * dt),
        "raw_common_window": {
            "start_timestamp": float(common_start),
            "end_timestamp": float(common_end),
            "duration_sec": float(max(0.0, common_end - common_start)),
        },
        "aligned_window": aligned_window,
        "output_rows": int(odom_rows.shape[0]),
        "inputs": {
            odom.name: {
                "path": str(odom.path),
                "rows_after_dedup": int(odom.data.shape[0]),
                "duplicate_rows_removed": odom.duplicate_rows_removed,
                "start_timestamp": float(odom.t[0]),
                "end_timestamp": float(odom.t[-1]),
                "median_dt_sec": median_dt(odom),
            },
            gt.name: {
                "path": str(gt.path),
                "rows_after_dedup": int(gt.data.shape[0]),
                "duplicate_rows_removed": gt.duplicate_rows_removed,
                "start_timestamp": float(gt.t[0]),
                "end_timestamp": float(gt.t[-1]),
                "median_dt_sec": median_dt(gt),
            },
        },
    }
    (summary_dir / "temporal_alignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "Temporal alignment summary",
        "==========================",
        "",
        "Convention: gt_motion(t + gt_time_offset_sec) ~= odom_motion(t)",
        f"gt_time_offset_sec: {ncc.best_lag_sec:.9f}",
        f"NCC peak score: {ncc.best_score:.6f}",
        f"boundary peak: {ncc.is_boundary_peak}",
        f"dt: {dt:.9f} sec",
        f"max lag: +/-{max_lag_sec:.3f} sec",
        f"smoothing: {smooth_sec:.3f} sec",
        f"best overlap: {ncc.best_overlap_samples} samples ({ncc.best_overlap_samples * dt:.3f} sec)",
        f"output rows: {odom_rows.shape[0]}",
        "",
        "Aligned window",
        f"  odom: {aligned_window['odom_time_start']:.9f} - {aligned_window['odom_time_end']:.9f}",
        f"  gt: {aligned_window['gt_time_start']:.9f} - {aligned_window['gt_time_end']:.9f}",
        f"  duration: {aligned_window['duration_sec']:.3f} sec",
        "",
        "Input cleanup",
        f"  {odom.name}: removed {odom.duplicate_rows_removed} duplicate timestamp rows",
        f"  {gt.name}: removed {gt.duplicate_rows_removed} duplicate timestamp rows",
        "",
    ]
    (summary_dir / "temporal_alignment_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def downsample(t: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if t.shape[0] <= max_points:
        return t, values
    step = int(math.ceil(t.shape[0] / max_points))
    return t[::step], values[::step]


def plot_diagnostics(
    odom_features: FeatureSeries,
    gt_features: FeatureSeries,
    ncc: NccResult,
    diagnostics_dir: Path,
    max_points: int,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    lag = ncc.best_lag_sec
    start = max(odom_features.t[0], gt_features.t[0] - lag)
    end = min(odom_features.t[-1], gt_features.t[-1] - lag)
    t_eval = make_uniform_time(start, end, ncc.lags[1] - ncc.lags[0])
    t_rel = t_eval - t_eval[0]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(ncc.lags, ncc.scores, linewidth=1.2, color="#1f77b4")
    axes[0].axvline(lag, color="#d62728", linestyle="--", label=f"offset={lag:.4f}s")
    axes[0].set_title("NCC score over candidate lag")
    axes[0].set_xlabel("gt_time_offset_sec [s]")
    axes[0].set_ylabel("NCC score")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    for ax, channel, ylabel in (
        (axes[1], "speed", "speed [m/s]"),
        (axes[2], "omega", "yaw rate [rad/s]"),
    ):
        odom_values = np.interp(t_eval, odom_features.t, getattr(odom_features, channel))
        gt_before = np.interp(t_eval, gt_features.t, getattr(gt_features, channel))
        gt_after = np.interp(t_eval + lag, gt_features.t, getattr(gt_features, channel))
        t_plot, odom_plot = downsample(t_rel, odom_values, max_points)
        _, before_plot = downsample(t_rel, gt_before, max_points)
        _, after_plot = downsample(t_rel, gt_after, max_points)
        ax.plot(t_plot, odom_plot, linewidth=1.0, color="#1f77b4", label="Odom")
        ax.plot(t_plot, before_plot, linewidth=0.8, color="#999999", alpha=0.65, label="GT before")
        ax.plot(t_plot, after_plot, linewidth=1.0, color="#d62728", label="GT shifted")
        ax.set_title(f"{channel} overlay")
        ax.set_xlabel("aligned time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(diagnostics_dir / "temporal_ncc.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate temporal offset with NCC and export temporally aligned JSONL."
    )
    parser.add_argument("--odom-in", type=Path, default=Path("data/Odom.jsonl"))
    parser.add_argument("--gt-in", type=Path, default=Path("data/pose_GT_by_mocap.jsonl"))
    parser.add_argument("--odom-out", type=Path, default=Path("data/Odom_temporal_aligned.jsonl"))
    parser.add_argument(
        "--gt-out", type=Path, default=Path("data/pose_GT_by_mocap_temporal_aligned.jsonl")
    )
    parser.add_argument("--summary-dir", type=Path, default=Path("results/summaries"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--max-lag-sec", type=float, default=5.0)
    parser.add_argument("--smooth-sec", type=float, default=0.15)
    parser.add_argument("--min-overlap-sec", type=float, default=20.0)
    parser.add_argument("--max-plot-points", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    odom = load_jsonl_trajectory(args.odom_in, "Odom")
    gt = load_jsonl_trajectory(args.gt_in, "pose_GT_by_mocap")
    dt = args.dt if args.dt is not None else max(median_dt(odom), median_dt(gt))
    if dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.max_lag_sec <= 0.0:
        raise ValueError("--max-lag-sec must be positive.")

    odom_features = build_feature_series(odom, dt, args.smooth_sec)
    gt_features = build_feature_series(gt, dt, args.smooth_sec)
    ncc = estimate_offset(
        odom_features=odom_features,
        gt_features=gt_features,
        dt=dt,
        max_lag_sec=args.max_lag_sec,
        min_overlap_sec=args.min_overlap_sec,
    )

    odom_rows, gt_rows, aligned_window = aligned_pose_rows(odom, gt, ncc.best_lag_sec, dt)
    write_jsonl(odom_rows, args.odom_out)
    write_jsonl(gt_rows, args.gt_out)
    write_summary(
        odom=odom,
        gt=gt,
        ncc=ncc,
        dt=dt,
        max_lag_sec=args.max_lag_sec,
        smooth_sec=args.smooth_sec,
        min_overlap_sec=args.min_overlap_sec,
        aligned_window=aligned_window,
        odom_rows=odom_rows,
        summary_dir=args.summary_dir,
    )
    plot_diagnostics(
        odom_features=odom_features,
        gt_features=gt_features,
        ncc=ncc,
        diagnostics_dir=args.diagnostics_dir,
        max_points=args.max_plot_points,
    )

    print(f"Estimated gt_time_offset_sec: {ncc.best_lag_sec:.9f}")
    print(f"NCC peak score: {ncc.best_score:.6f}")
    print(f"Wrote {odom_rows.shape[0]} rows: {args.odom_out}")
    print(f"Wrote {gt_rows.shape[0]} rows: {args.gt_out}")
    print(f"Wrote summary: {args.summary_dir / 'temporal_alignment_summary.txt'}")
    print(f"Wrote diagnostics: {args.diagnostics_dir / 'temporal_ncc.png'}")


if __name__ == "__main__":
    main()
