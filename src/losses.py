"""Robust regression losses."""

from __future__ import annotations

import torch


def wrap_angle_error(error: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(error), torch.cos(error))


def berhu_loss_from_error(error: torch.Tensor, c: float) -> torch.Tensor:
    if c <= 0.0:
        raise ValueError("BerHu c must be positive.")
    abs_error = torch.abs(error)
    linear = abs_error
    quadratic = (error * error + c * c) / (2.0 * c)
    return torch.where(abs_error <= c, linear, quadratic)


def robust_sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    berhu_c: float,
    theta_loss_weight: float,
    theta_index: int,
    wrap_theta_residual: bool,
) -> torch.Tensor:
    error = pred - target
    if wrap_theta_residual:
        error = error.clone()
        error[..., theta_index] = wrap_angle_error(error[..., theta_index])

    loss_type = loss_type.lower()
    if loss_type == "berhu":
        loss = berhu_loss_from_error(error, berhu_c)
    elif loss_type == "l1":
        loss = torch.abs(error)
    elif loss_type in {"mse", "l2"}:
        loss = error * error
    else:
        raise ValueError(f"Unsupported LOSS_TYPE: {loss_type!r}")

    if theta_loss_weight != 1.0:
        weights = torch.ones(loss.shape[-1], dtype=loss.dtype, device=loss.device)
        weights[theta_index] = theta_loss_weight
        loss = loss * weights
    return loss.mean()
