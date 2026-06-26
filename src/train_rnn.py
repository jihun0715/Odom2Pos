#!/usr/bin/env python3
"""Train a CPU RNN model that maps odom velocity to mocap GT velocity."""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config as cfg
from losses import robust_sequence_loss
from model import RNNRegressor


STD_EPSILON = 1e-8


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray, starts: Iterable[int], seq_len: int) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.starts = list(starts)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        end = start + self.seq_len
        return torch.from_numpy(self.x[start:end]), torch.from_numpy(self.y[start:end])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                timestamps.append(float(record["timestamp"]))
                values.append([float(record[key]) for key in keys])
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing key {exc.args[0]!r}") from exc
    if not values:
        raise ValueError(f"{path} has no rows.")
    return np.asarray(timestamps, dtype=float), np.asarray(values, dtype=float)


def load_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_t, x = load_jsonl(Path(cfg.INPUT_PATH), tuple(cfg.FEATURE_KEYS))
    target_t, y = load_jsonl(Path(cfg.TARGET_PATH), tuple(cfg.TARGET_KEYS))
    if input_t.shape != target_t.shape:
        raise ValueError("Input and target timestamp arrays have different lengths.")
    if not np.allclose(input_t, target_t, rtol=0.0, atol=1e-9):
        max_diff = float(np.max(np.abs(input_t - target_t)))
        raise ValueError(f"Input and target timestamps differ. max_diff={max_diff}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Input or target contains non-finite values.")
    return input_t, x, y


def make_normalizer(values: np.ndarray) -> Normalizer:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < STD_EPSILON, 1.0, std)
    return Normalizer(mean=mean, std=std)


def make_window_starts(start: int, end: int, seq_len: int, stride: int) -> list[int]:
    if end - start < seq_len:
        return []
    return list(range(start, end - seq_len + 1, stride))


def make_dataloaders(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    train_end: int,
) -> tuple[DataLoader, DataLoader, list[int], list[int]]:
    train_starts = make_window_starts(0, train_end, cfg.SEQ_LEN, cfg.STRIDE)
    val_starts = make_window_starts(train_end, x_norm.shape[0], cfg.SEQ_LEN, cfg.STRIDE)
    if not train_starts:
        raise ValueError("No training windows. Reduce SEQ_LEN or TRAIN_RATIO.")
    if not val_starts:
        raise ValueError("No validation windows. Reduce SEQ_LEN or increase validation span.")

    train_loader = DataLoader(
        SequenceDataset(x_norm, y_norm, train_starts, cfg.SEQ_LEN),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(x_norm, y_norm, val_starts, cfg.SEQ_LEN),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader, train_starts, val_starts


def config_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name in dir(cfg):
        if not name.isupper():
            continue
        value = getattr(cfg, name)
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, (str, int, float, bool, list)) or value is None:
            snapshot[name] = value
    return snapshot


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    metrics: dict[str, float] = {}
    for idx, key in enumerate(cfg.TARGET_KEYS):
        metrics[f"mae_{key}"] = float(np.mean(np.abs(error[..., idx])))
        metrics[f"rmse_{key}"] = float(np.sqrt(np.mean(error[..., idx] ** 2)))
    metrics["mae_all"] = float(np.mean(np.abs(error)))
    metrics["rmse_all"] = float(np.sqrt(np.mean(error**2)))
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = robust_sequence_loss(
            pred,
            y_batch,
            loss_type=cfg.LOSS_TYPE,
            berhu_c=cfg.BERHU_C,
            theta_loss_weight=cfg.THETA_LOSS_WEIGHT,
            theta_index=cfg.THETA_INDEX,
            wrap_theta_residual=cfg.WRAP_THETA_RESIDUAL,
        )
        if is_train:
            loss.backward()
            if cfg.GRAD_CLIP_NORM is not None and cfg.GRAD_CLIP_NORM > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()
        batch_size = x_batch.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


@torch.no_grad()
def predict_windows(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        pred = model(x_batch.to(device)).cpu().numpy()
        preds.append(pred)
        targets.append(y_batch.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def save_checkpoint(
    model: nn.Module,
    x_normalizer: Normalizer,
    y_normalizer: Normalizer,
    history: list[dict[str, float]],
    best_epoch: int,
) -> None:
    checkpoint_path = Path(cfg.CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_snapshot(),
            "input_normalizer": x_normalizer.to_dict(),
            "target_normalizer": y_normalizer.to_dict(),
            "history": history,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    Path(cfg.CONFIG_SNAPSHOT_PATH).write_text(
        json.dumps(config_snapshot(), indent=2), encoding="utf-8"
    )


def save_history(history: list[dict[str, float]]) -> None:
    history_path = Path(cfg.HISTORY_PATH)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def plot_loss_curve(history: list[dict[str, float]]) -> None:
    out_path = Path(cfg.LOSS_CURVE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("RNN training loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prediction_preview(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    input_norm: np.ndarray,
    y_normalizer: Normalizer,
    x_normalizer: Normalizer,
) -> None:
    out_path = Path(cfg.PREDICTION_PREVIEW_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred = y_normalizer.inverse(pred_norm[0])
    target = y_normalizer.inverse(target_norm[0])
    raw_input = x_normalizer.inverse(input_norm[0])
    t = np.arange(pred.shape[0])

    fig, axes = plt.subplots(len(cfg.TARGET_KEYS), 1, figsize=(10, 7), sharex=True)
    if len(cfg.TARGET_KEYS) == 1:
        axes = [axes]
    for idx, key in enumerate(cfg.TARGET_KEYS):
        axes[idx].plot(t, target[:, idx], label="GT", color="#d62728")
        axes[idx].plot(t, pred[:, idx], label="prediction", color="#1f77b4")
        axes[idx].plot(t, raw_input[:, idx], label="raw odom", color="#7f7f7f", alpha=0.7)
        axes[idx].set_ylabel(key)
        axes[idx].grid(True, alpha=0.25)
        axes[idx].legend(loc="best")
    axes[-1].set_xlabel("sequence sample")
    fig.suptitle("Validation prediction preview")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def stack_windows(values: np.ndarray, starts: Iterable[int], seq_len: int) -> np.ndarray:
    return np.stack([values[start : start + seq_len] for start in starts], axis=0)


def main() -> None:
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    timestamps, x, y = load_training_arrays()
    train_end = int(math.floor(timestamps.shape[0] * cfg.TRAIN_RATIO))

    x_normalizer = make_normalizer(x[:train_end])
    y_normalizer = make_normalizer(y[:train_end])
    x_norm = x_normalizer.transform(x)
    y_norm = y_normalizer.transform(y)

    train_loader, val_loader, train_starts, val_starts = make_dataloaders(
        x_norm, y_norm, train_end
    )
    model = RNNRegressor(
        input_dim=len(cfg.FEATURE_KEYS),
        output_dim=len(cfg.TARGET_KEYS),
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        rnn_type=cfg.RNN_TYPE,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    print(
        f"Training on {device} with {len(train_starts)} train windows and "
        f"{len(val_starts)} validation windows."
    )
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, None)
            pred_norm, target_norm = predict_windows(model, val_loader, device)
        pred = y_normalizer.inverse(pred_norm)
        target = y_normalizer.inverse(target_norm)
        metrics = compute_metrics(pred, target)
        raw_val = stack_windows(x, val_starts, cfg.SEQ_LEN)
        raw_metrics = compute_metrics(raw_val, target)
        raw_metrics = {f"raw_odom_{key}": value for key, value in raw_metrics.items()}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
            **raw_metrics,
        }
        history.append(row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.EPOCHS:
            print(
                f"epoch {epoch:04d} train={train_loss:.6f} val={val_loss:.6f} "
                f"rmse_all={metrics['rmse_all']:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, x_normalizer, y_normalizer, history, best_epoch)
    save_history(history)
    plot_loss_curve(history)
    with torch.no_grad():
        pred_norm, target_norm = predict_windows(model, val_loader, device)
    first_val_start = val_starts[0]
    input_preview_norm = x_norm[first_val_start : first_val_start + cfg.SEQ_LEN][None, ...]
    plot_prediction_preview(
        pred_norm,
        target_norm,
        input_preview_norm,
        y_normalizer,
        x_normalizer,
    )
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    print(f"Wrote checkpoint: {cfg.CHECKPOINT_PATH}")
    print(f"Wrote history: {cfg.HISTORY_PATH}")
    print(f"Wrote plots: {cfg.LOSS_CURVE_PATH}, {cfg.PREDICTION_PREVIEW_PATH}")


if __name__ == "__main__":
    main()
