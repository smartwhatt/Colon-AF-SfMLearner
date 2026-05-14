"""Unit tests for B.1 — Uncertainty-Weighted Pose Loss (Kendall & Cipolla, CVPR 2017)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from trainers.c3vdv2_losses import pose_loss
from trainers.stage_two import PoseUncertaintyScales
from options import TrainingConfig


# ── pose_loss: fixed-weight path (backward compat) ───────────────────────────

class TestPoseLossFixedWeights:
    def test_zero_error_gives_zero_loss(self):
        x = torch.randn(4, 3)
        loss = pose_loss(x, x, x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_finite(self):
        pred_ax = torch.randn(4, 3)
        pred_tr = torch.randn(4, 3)
        gt_ax = torch.randn(4, 3)
        gt_tr = torch.randn(4, 3)
        loss = pose_loss(pred_ax, pred_tr, gt_ax, gt_tr)
        assert torch.isfinite(loss)

    def test_loss_is_scalar(self):
        a = torch.randn(2, 3)
        loss = pose_loss(a, a, torch.zeros_like(a), torch.zeros_like(a))
        assert loss.ndim == 0

    def test_rot_weight_scales_rotation_component(self):
        ax = torch.ones(1, 3)
        tr = torch.zeros(1, 3)
        gt_ax = torch.zeros(1, 3)
        gt_tr = torch.zeros(1, 3)
        l1 = pose_loss(ax, tr, gt_ax, gt_tr, rot_weight=1.0)
        l2 = pose_loss(ax, tr, gt_ax, gt_tr, rot_weight=2.0)
        assert l2.item() == pytest.approx(2.0 * l1.item(), rel=1e-5)

    def test_gradients_flow_to_predictions(self):
        pred_ax = torch.randn(4, 3, requires_grad=True)
        pred_tr = torch.randn(4, 3, requires_grad=True)
        gt_ax = torch.randn(4, 3)
        gt_tr = torch.randn(4, 3)
        loss = pose_loss(pred_ax, pred_tr, gt_ax, gt_tr)
        loss.backward()
        assert pred_ax.grad is not None and pred_ax.grad.abs().sum() > 0
        assert pred_tr.grad is not None and pred_tr.grad.abs().sum() > 0

    def test_log_s_none_uses_fixed_weights(self):
        a, b = torch.randn(2, 3), torch.randn(2, 3)
        l_fixed = pose_loss(a, b, torch.zeros_like(a), torch.zeros_like(b))
        l_none = pose_loss(a, b, torch.zeros_like(a), torch.zeros_like(b),
                           log_s_rot=None, log_s_trans=None)
        assert l_fixed.item() == pytest.approx(l_none.item(), rel=1e-6)


# ── pose_loss: uncertainty-weighted path ─────────────────────────────────────

class TestPoseLossUncertainty:
    def _make_scalars(self, val=0.0, requires_grad=True):
        return (
            torch.tensor([val], requires_grad=requires_grad),
            torch.tensor([val], requires_grad=requires_grad),
        )

    def test_loss_finite(self):
        log_s_r, log_s_t = self._make_scalars()
        loss = pose_loss(
            torch.randn(4, 3), torch.randn(4, 3),
            torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
        )
        assert torch.isfinite(loss)

    def test_loss_is_scalar(self):
        log_s_r, log_s_t = self._make_scalars()
        loss = pose_loss(
            torch.randn(2, 3), torch.randn(2, 3),
            torch.zeros(2, 3), torch.zeros(2, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
        )
        assert loss.ndim == 0

    def test_gradients_flow_to_scalars(self):
        log_s_r, log_s_t = self._make_scalars()
        loss = pose_loss(
            torch.randn(4, 3), torch.randn(4, 3),
            torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
        )
        loss.backward()
        assert log_s_r.grad is not None and log_s_r.grad.abs().sum() > 0
        assert log_s_t.grad is not None and log_s_t.grad.abs().sum() > 0

    def test_gradients_flow_to_predictions(self):
        pred_ax = torch.randn(4, 3, requires_grad=True)
        pred_tr = torch.randn(4, 3, requires_grad=True)
        log_s_r, log_s_t = self._make_scalars()
        loss = pose_loss(
            pred_ax, pred_tr, torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
        )
        loss.backward()
        assert pred_ax.grad is not None and pred_ax.grad.abs().sum() > 0
        assert pred_tr.grad is not None and pred_tr.grad.abs().sum() > 0

    def test_zero_error_loss_equals_regularisation_only(self):
        """When pred == gt, uncertainty loss == log_s_rot + log_s_trans."""
        x = torch.randn(4, 3)
        log_s_r, log_s_t = self._make_scalars(val=1.5, requires_grad=False)
        loss = pose_loss(x, x, x, x, log_s_rot=log_s_r, log_s_trans=log_s_t)
        expected = (log_s_r + log_s_t).item()
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_large_negative_scalars_do_not_produce_nan(self):
        """Very negative log_s values make exp(-log_s) large but loss stays finite."""
        log_s_r = torch.tensor([-10.0], requires_grad=True)
        log_s_t = torch.tensor([-10.0], requires_grad=True)
        loss = pose_loss(
            torch.randn(4, 3), torch.randn(4, 3),
            torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
        )
        assert torch.isfinite(loss)

    def test_uncertainty_loss_not_equal_to_fixed_loss(self):
        """With nonzero error, uncertainty path produces a different value than fixed weights."""
        pred_ax = torch.ones(4, 3)
        pred_tr = torch.ones(4, 3)
        gt = torch.zeros(4, 3)
        log_s_r, log_s_t = self._make_scalars(val=0.5, requires_grad=False)
        l_fixed = pose_loss(pred_ax, pred_tr, gt, gt)
        l_uncert = pose_loss(pred_ax, pred_tr, gt, gt,
                             log_s_rot=log_s_r, log_s_trans=log_s_t)
        assert l_fixed.item() != pytest.approx(l_uncert.item(), rel=1e-4)


# ── PoseUncertaintyScales module ──────────────────────────────────────────────

class TestPoseUncertaintyScales:
    def test_initialises_to_zero(self):
        pu = PoseUncertaintyScales()
        assert pu.log_s_rot.item() == pytest.approx(0.0)
        assert pu.log_s_trans.item() == pytest.approx(0.0)

    def test_is_nn_module(self):
        assert isinstance(PoseUncertaintyScales(), nn.Module)

    def test_has_exactly_two_parameters(self):
        pu = PoseUncertaintyScales()
        params = list(pu.parameters())
        assert len(params) == 2

    def test_parameters_require_grad(self):
        pu = PoseUncertaintyScales()
        for p in pu.parameters():
            assert p.requires_grad

    def test_state_dict_round_trip(self):
        pu = PoseUncertaintyScales()
        pu.log_s_rot.data.fill_(0.42)
        pu.log_s_trans.data.fill_(-1.7)
        sd = pu.state_dict()
        pu2 = PoseUncertaintyScales()
        pu2.load_state_dict(sd)
        assert pu2.log_s_rot.item() == pytest.approx(0.42, rel=1e-5)
        assert pu2.log_s_trans.item() == pytest.approx(-1.7, rel=1e-5)

    def test_gradients_update_via_optimizer(self):
        pu = PoseUncertaintyScales()
        opt = torch.optim.SGD(pu.parameters(), lr=0.1)
        pred_ax = torch.randn(4, 3)
        pred_tr = torch.randn(4, 3)
        loss = pose_loss(
            pred_ax, pred_tr, torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=pu.log_s_rot, log_s_trans=pu.log_s_trans,
        )
        loss.backward()
        before_r = pu.log_s_rot.item()
        before_t = pu.log_s_trans.item()
        opt.step()
        assert pu.log_s_rot.item() != pytest.approx(before_r)
        assert pu.log_s_trans.item() != pytest.approx(before_t)


# ── options: use_uncertainty_pose flag ───────────────────────────────────────

class TestUncertaintyPoseOption:
    def test_default_is_false(self):
        cfg = TrainingConfig(mode="c3vdv2_stage2")
        assert cfg.use_uncertainty_pose is False

    def test_can_be_set_true(self):
        cfg = TrainingConfig(mode="c3vdv2_stage2", use_uncertainty_pose=True)
        assert cfg.use_uncertainty_pose is True

    def test_cli_flag_sets_true(self):
        cfg = TrainingConfig.from_args(
            ["--mode", "c3vdv2_stage2", "--use_uncertainty_pose"]
        )
        assert cfg.use_uncertainty_pose is True

    def test_cli_omitting_flag_leaves_false(self):
        cfg = TrainingConfig.from_args(["--mode", "c3vdv2_stage2"])
        assert cfg.use_uncertainty_pose is False
