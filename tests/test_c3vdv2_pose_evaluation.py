import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the pure functions under test (not the full evaluate() entry point).
from tools.evaluate_c3vdv2_pose import _translation_metrics  # noqa: E402


class TestTranslationMetrics:
    """Unit tests for _translation_metrics."""

    def test_nonzero_gt_returns_valid_cosine_and_ratio(self):
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        gt = torch.tensor([[2.0, 0.0, 0.0]])
        result = _translation_metrics(pred, gt, min_gt_norm=1e-6)

        assert result["gt_valid"][0].item() is True
        assert math.isfinite(result["trans_cosine_valid"][0].item())
        assert math.isfinite(result["trans_mag_ratio_valid"][0].item())
        # Cosine should be 1.0 (same direction)
        assert abs(result["trans_cosine"][0].item() - 1.0) < 1e-6
        # Magnitude ratio should be 0.5
        assert abs(result["trans_mag_ratio"][0].item() - 0.5) < 1e-6

    def test_near_zero_gt_excludes_cosine_and_ratio_from_valid(self):
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        gt = torch.tensor([[1e-9, 0.0, 0.0]])
        result = _translation_metrics(pred, gt, min_gt_norm=1e-6)

        assert result["gt_valid"][0].item() is False
        assert math.isnan(result["trans_cosine_valid"][0].item())
        assert math.isnan(result["trans_mag_ratio_valid"][0].item())
        # trans_l2_mm still reported (raw L2, not a ratio)
        assert math.isfinite(result["trans_l2_mm"][0].item())

    def test_mixed_batch_splits_valid_and_invalid(self):
        pred = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        gt = torch.tensor([
            [2.0, 0.0, 0.0],   # valid (norm=2.0)
            [0.0, 0.0, 0.0],   # invalid (norm=0.0)
        ])
        result = _translation_metrics(pred, gt, min_gt_norm=1e-6)

        assert result["gt_valid"][0].item() is True
        assert result["gt_valid"][1].item() is False
        assert math.isfinite(result["trans_cosine_valid"][0].item())
        assert math.isnan(result["trans_cosine_valid"][1].item())

    def test_custom_threshold_respects_min_gt_norm(self):
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        gt = torch.tensor([[0.01, 0.0, 0.0]])  # norm = 0.01

        result_strict = _translation_metrics(pred, gt, min_gt_norm=0.1)
        assert result_strict["gt_valid"][0].item() is False

        result_lenient = _translation_metrics(pred, gt, min_gt_norm=0.001)
        assert result_lenient["gt_valid"][0].item() is True

    def test_trans_l2_mm_reported_regardless_of_validity(self):
        pred = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]])
        gt = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = _translation_metrics(pred, gt, min_gt_norm=1e-6)

        # Both rows should have finite trans_l2_mm values
        assert math.isfinite(result["trans_l2_mm"][0].item())
        assert math.isfinite(result["trans_l2_mm"][1].item())
