"""Shared C3VDv2 depth masking and metric helpers."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


C3VD_MIN_DEPTH = 0.001
C3VD_MAX_DEPTH = 0.100
C3VD_SATURATION_MARGIN = 0.001


DEPTH_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")


def build_c3vdv2_valid_mask_np(
    gt_depth: np.ndarray,
    occlusion: np.ndarray | None = None,
    *,
    min_depth: float = C3VD_MIN_DEPTH,
    max_depth: float = C3VD_MAX_DEPTH,
    saturation_margin: float = C3VD_SATURATION_MARGIN,
) -> np.ndarray:
    """Return the C3VDv2 valid-depth mask used for metric reporting."""
    mask = (gt_depth > min_depth) & (gt_depth < max_depth - saturation_margin)
    if occlusion is not None:
        mask = mask & (occlusion <= 0)
    return mask


def build_c3vdv2_valid_mask_torch(
    gt_depth: torch.Tensor,
    occlusion: torch.Tensor | None = None,
    *,
    min_depth: float = C3VD_MIN_DEPTH,
    max_depth: float = C3VD_MAX_DEPTH,
    saturation_margin: float = C3VD_SATURATION_MARGIN,
) -> torch.Tensor:
    """Torch equivalent of build_c3vdv2_valid_mask_np."""
    mask = (gt_depth > min_depth) & (gt_depth < max_depth - saturation_margin)
    if occlusion is not None:
        mask = mask & (occlusion <= 0)
    return mask


def _maybe_median_scale_np(
    gt: np.ndarray,
    pred: np.ndarray,
    median_scaling: bool,
    eps: float,
) -> tuple[np.ndarray, float]:
    if not median_scaling:
        return pred, 1.0
    median_pred = float(np.median(pred))
    if median_pred <= eps:
        return pred, 1.0
    scale = float(np.median(gt)) / median_pred
    return pred * scale, scale


def compute_depth_metrics_np(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_depth: float = C3VD_MIN_DEPTH,
    max_depth: float = C3VD_MAX_DEPTH,
    median_scaling: bool = False,
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Compute standard depth metrics on C3VDv2 metric depths.

    C3VDv2 has absolute metric depth supervision, so median_scaling defaults to
    False. Pass median_scaling=True only for self-supervised scale-aligned
    comparisons.
    """
    if mask is None:
        mask = build_c3vdv2_valid_mask_np(gt_depth, min_depth=min_depth, max_depth=max_depth)

    gt = gt_depth[mask].astype(np.float64)
    pred = pred_depth[mask].astype(np.float64)
    if gt.size == 0:
        return {key: float("nan") for key in DEPTH_METRIC_KEYS} | {
            "valid_pixels": 0.0,
            "scale": 1.0,
        }

    pred, scale = _maybe_median_scale_np(gt, pred, median_scaling, eps)
    pred = np.clip(pred, min_depth, max_depth)
    pred = np.maximum(pred, eps)
    gt = np.maximum(gt, eps)

    thresh = np.maximum(gt / pred, pred / gt)
    error = gt - pred
    log_error = np.log(gt) - np.log(pred)

    return {
        "abs_rel": float(np.mean(np.abs(error) / gt)),
        "sq_rel": float(np.mean((error ** 2) / gt)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "rmse_log": float(np.sqrt(np.mean(log_error ** 2))),
        "a1": float(np.mean(thresh < 1.25)),
        "a2": float(np.mean(thresh < 1.25**2)),
        "a3": float(np.mean(thresh < 1.25**3)),
        "valid_pixels": float(gt.size),
        "scale": float(scale),
    }


def depth_metric_sums_torch(
    gt_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    min_depth: float = C3VD_MIN_DEPTH,
    max_depth: float = C3VD_MAX_DEPTH,
    median_scaling: bool = False,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Return additive metric sums so batches can be aggregated correctly."""
    if mask is None:
        mask = build_c3vdv2_valid_mask_torch(
            gt_depth, min_depth=min_depth, max_depth=max_depth
        )

    gt = gt_depth[mask].float()
    pred = pred_depth[mask].float()
    if gt.numel() == 0:
        zero = torch.zeros((), device=gt_depth.device)
        return {
            "valid_pixels": zero,
            "abs_rel_sum": zero,
            "sq_rel_sum": zero,
            "sq_error_sum": zero,
            "log_sq_error_sum": zero,
            "a1_count": zero,
            "a2_count": zero,
            "a3_count": zero,
        }

    if median_scaling:
        median_pred = torch.median(pred)
        if median_pred > eps:
            pred = pred * (torch.median(gt) / median_pred)

    pred = torch.clamp(pred, min=min_depth, max=max_depth)
    pred = torch.clamp(pred, min=eps)
    gt = torch.clamp(gt, min=eps)

    thresh = torch.maximum(gt / pred, pred / gt)
    error = gt - pred
    log_error = torch.log(gt) - torch.log(pred)

    return {
        "valid_pixels": gt.new_tensor(float(gt.numel())),
        "abs_rel_sum": (torch.abs(error) / gt).sum(),
        "sq_rel_sum": ((error ** 2) / gt).sum(),
        "sq_error_sum": (error ** 2).sum(),
        "log_sq_error_sum": (log_error ** 2).sum(),
        "a1_count": (thresh < 1.25).float().sum(),
        "a2_count": (thresh < 1.25**2).float().sum(),
        "a3_count": (thresh < 1.25**3).float().sum(),
    }


def merge_depth_metric_sums(
    totals: dict[str, float],
    sums: Mapping[str, torch.Tensor | float],
) -> dict[str, float]:
    """Accumulate torch or numeric metric sums into a plain dict."""
    for key, value in sums.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu())
        totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def finalize_depth_metric_sums(totals: Mapping[str, float]) -> dict[str, float]:
    """Convert additive sums from depth_metric_sums_torch into metric means."""
    n = float(totals.get("valid_pixels", 0.0))
    if n <= 0:
        return {key: float("nan") for key in DEPTH_METRIC_KEYS} | {
            "valid_pixels": 0.0,
            "rmse_mm": float("nan"),
        }

    rmse = float(np.sqrt(totals.get("sq_error_sum", 0.0) / n))
    return {
        "abs_rel": totals.get("abs_rel_sum", 0.0) / n,
        "sq_rel": totals.get("sq_rel_sum", 0.0) / n,
        "rmse": rmse,
        "rmse_mm": rmse * 1000.0,
        "rmse_log": float(np.sqrt(totals.get("log_sq_error_sum", 0.0) / n)),
        "a1": totals.get("a1_count", 0.0) / n,
        "a2": totals.get("a2_count", 0.0) / n,
        "a3": totals.get("a3_count", 0.0) / n,
        "valid_pixels": n,
    }
