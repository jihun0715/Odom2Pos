#!/usr/bin/env python3
"""Run a trained odom-to-GT RNN checkpoint and export predicted velocity JSONL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

import config as cfg
from model import RNNRegressor


STD_EPSILON = 1e-8


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "Normalizer":
        return cls(mean=np.asarray(data["mean"], dtype=float), std=np.asarray(data["std"], dtype=float))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


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


def write_jsonl(
    timestamps: np.ndarray, values: np.ndarray, out_path: Path, keys: tuple[str, ...]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for timestamp, row in zip(timestamps, values, strict=True):
            record = {"timestamp": float(timestamp)}
            for key, value in zip(keys, row, strict=True):
                record[key] = float(value)
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def make_window_starts(n_rows: int, seq_len: int, stride: int) -> list[int]:
    if n_rows <= seq_len:
        return [0]
    starts = list(range(0, n_rows - seq_len + 1, stride))
    last = n_rows - seq_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_model(config_data: dict[str, object]) -> RNNRegressor:
    return RNNRegressor(
        input_dim=len(config_data["FEATURE_KEYS"]),
        output_dim=len(config_data["TARGET_KEYS"]),
        hidden_size=int(config_data["HIDDEN_SIZE"]),
        num_layers=int(config_data["NUM_LAYERS"]),
        dropout=float(config_data["DROPOUT"]),
        rnn_type=str(config_data["RNN_TYPE"]),
    )


@torch.no_grad()
def predict_full_sequence(
    model: RNNRegressor,
    x_norm: np.ndarray,
    y_normalizer: Normalizer,
    seq_len: int,
    stride: int,
    output_dim: int,
    device: torch.device,
) -> np.ndarray:
    starts = make_window_starts(x_norm.shape[0], seq_len, stride)
    pred_sum = np.zeros((x_norm.shape[0], output_dim), dtype=float)
    pred_count = np.zeros((x_norm.shape[0], 1), dtype=float)
    model.eval()
    for start in starts:
        end = min(start + seq_len, x_norm.shape[0])
        window = x_norm[start:end]
        if window.shape[0] < seq_len:
            pad_count = seq_len - window.shape[0]
            window = np.vstack([window, np.repeat(window[-1:], pad_count, axis=0)])
        x_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
        pred_norm = model(x_tensor).squeeze(0).cpu().numpy()[: end - start]
        pred = y_normalizer.inverse(pred_norm)
        pred_sum[start:end] += pred
        pred_count[start:end] += 1.0
    if np.any(pred_count < 1.0):
        raise RuntimeError("Inference did not cover every input row.")
    return pred_sum / np.maximum(pred_count, STD_EPSILON)


def main() -> None:
    checkpoint_path = Path(cfg.CHECKPOINT_PATH)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    config_data = checkpoint["config"]
    x_normalizer = Normalizer.from_dict(checkpoint["input_normalizer"])
    y_normalizer = Normalizer.from_dict(checkpoint["target_normalizer"])
    feature_keys = tuple(str(key) for key in config_data["FEATURE_KEYS"])
    target_keys = tuple(str(key) for key in config_data["TARGET_KEYS"])

    input_path = Path(cfg.INFERENCE_INPUT_PATH)
    output_path = Path(cfg.INFERENCE_OUTPUT_PATH)
    timestamps, x = load_jsonl(input_path, feature_keys)
    x_norm = x_normalizer.transform(x)

    device = torch.device(cfg.DEVICE)
    model = build_model(config_data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pred = predict_full_sequence(
        model,
        x_norm,
        y_normalizer,
        seq_len=int(config_data["SEQ_LEN"]),
        stride=int(config_data["STRIDE"]),
        output_dim=len(target_keys),
        device=device,
    )
    write_jsonl(timestamps, pred, output_path, target_keys)
    print(f"Wrote {pred.shape[0]} rows: {output_path}")


if __name__ == "__main__":
    main()
