import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colon_datasets.c3vdv2_metrics import (
    build_c3vdv2_valid_mask_np,
    compute_depth_metrics_np,
    depth_metric_sums_torch,
    finalize_depth_metric_sums,
    merge_depth_metric_sums,
)


def test_c3vdv2_mask_excludes_depth_limits_and_occlusion():
    gt = np.array([[0.0, 0.02], [0.0995, 0.04]], dtype=np.float32)
    occlusion = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    mask = build_c3vdv2_valid_mask_np(gt, occlusion)

    assert mask.tolist() == [[False, True], [False, False]]


def test_metric_depth_defaults_to_no_median_scaling():
    gt = np.array([0.02, 0.04], dtype=np.float32)
    pred = np.array([0.01, 0.02], dtype=np.float32)
    mask = np.array([True, True])

    raw = compute_depth_metrics_np(gt, pred, mask)
    scaled = compute_depth_metrics_np(gt, pred, mask, median_scaling=True)

    assert raw["rmse"] > 0
    assert scaled["rmse"] == 0


def test_torch_metric_sums_aggregate_weighted_batches():
    totals = {}
    gt1 = torch.tensor([[[[0.02, 0.04]]]])
    pred1 = torch.tensor([[[[0.02, 0.02]]]])
    gt2 = torch.tensor([[[[0.03, 0.05]]]])
    pred2 = torch.tensor([[[[0.03, 0.05]]]])

    merge_depth_metric_sums(totals, depth_metric_sums_torch(gt1, pred1))
    merge_depth_metric_sums(totals, depth_metric_sums_torch(gt2, pred2))
    metrics = finalize_depth_metric_sums(totals)

    assert metrics["valid_pixels"] == 4
    assert metrics["rmse"] > 0
    assert 0 <= metrics["a1"] <= 1
