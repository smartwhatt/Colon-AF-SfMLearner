"""Unit tests for B.1 — EMA-Normalised Uncertainty-Weighted Pose Loss."""

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

    def test_ema_none_with_log_s_falls_back_to_raw(self):
        """When log_s scalars are provided but EMAs are None, uses raw MSE losses."""
        pred_ax = torch.ones(4, 3)
        pred_tr = torch.ones(4, 3)
        gt = torch.zeros(4, 3)
        log_s_r = torch.tensor([0.0], requires_grad=False)
        log_s_t = torch.tensor([0.0], requires_grad=False)
        l_raw = pose_loss(pred_ax, pred_tr, gt, gt,
                          log_s_rot=log_s_r, log_s_trans=log_s_t,
                          ema_rot=None, ema_trans=None)
        # F.mse_loss(ones, zeros) = 1.0 (mean over 12 elements)
        # With log_s=0: loss = 1.0*exp(0) + 0 + 1.0*exp(0) + 0 = 2.0
        assert l_raw.item() == pytest.approx(2.0, rel=1e-5)


# ── pose_loss: EMA-normalised uncertainty path ───────────────────────────────

class TestPoseLossEMANormalised:
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
            ema_rot=torch.tensor([0.01]), ema_trans=torch.tensor([0.0001]),
        )
        assert torch.isfinite(loss)

    def test_loss_is_scalar(self):
        log_s_r, log_s_t = self._make_scalars()
        loss = pose_loss(
            torch.randn(2, 3), torch.randn(2, 3),
            torch.zeros(2, 3), torch.zeros(2, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
            ema_rot=torch.tensor([0.01]), ema_trans=torch.tensor([0.0001]),
        )
        assert loss.ndim == 0

    def test_gradients_flow_to_scalars(self):
        log_s_r, log_s_t = self._make_scalars()
        loss = pose_loss(
            torch.randn(4, 3), torch.randn(4, 3),
            torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=log_s_r, log_s_trans=log_s_t,
            ema_rot=torch.tensor([0.01]), ema_trans=torch.tensor([0.0001]),
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
            ema_rot=torch.tensor([0.01]), ema_trans=torch.tensor([0.0001]),
        )
        loss.backward()
        assert pred_ax.grad is not None and pred_ax.grad.abs().sum() > 0
        assert pred_tr.grad is not None and pred_tr.grad.abs().sum() > 0

    def test_zero_error_loss_equals_regularisation_only(self):
        """When pred == gt the error terms vanish — loss = log_s_rot + log_s_trans."""
        x = torch.randn(4, 3)
        log_s_r, log_s_t = self._make_scalars(val=1.5, requires_grad=False)
        loss = pose_loss(
            x, x, x, x, log_s_rot=log_s_r, log_s_trans=log_s_t,
            ema_rot=torch.tensor([0.01]), ema_trans=torch.tensor([0.0001]),
        )
        expected = (log_s_r + log_s_t).item()
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_ema_division_scales_loss_down(self):
        """A larger EMA denominator makes rot_error/ema smaller → lower loss."""
        pred_ax = torch.ones(4, 3)
        pred_tr = torch.zeros(4, 3)
        gt = torch.zeros(4, 3)
        log_s_r = torch.tensor([0.0], requires_grad=False)
        log_s_t = torch.tensor([0.0], requires_grad=False)

        l1 = pose_loss(
            pred_ax, pred_tr, gt, gt, log_s_rot=log_s_r, log_s_trans=log_s_t,
            ema_rot=torch.tensor([0.01]), ema_trans=torch.tensor([0.0001]),
        )
        l2 = pose_loss(
            pred_ax, pred_tr, gt, gt, log_s_rot=log_s_r, log_s_trans=log_s_t,
            ema_rot=torch.tensor([10.0]), ema_trans=torch.tensor([10.0]),
        )
        assert l2.item() < l1.item()

    def test_equilibrium_near_zero_when_normalised(self):
        """With EMA ≈ error magnitude, the scalars should settle near 0."""
        err_scale = 0.05  # typical axis-angle error
        pred_ax = torch.full((32, 3), err_scale)
        pred_tr = torch.full((32, 3), 0.001)  # typical translation error in metres
        gt = torch.zeros(32, 3)
        ema_rot = torch.tensor([err_scale ** 2])   # ≈ 0.0025
        ema_trn = torch.tensor([1e-6])              # ≈ 0.001²

        log_s_r = torch.tensor([0.0], requires_grad=True)
        log_s_t = torch.tensor([0.0], requires_grad=True)

        loss = pose_loss(
            pred_ax, pred_tr, gt, gt,
            log_s_rot=log_s_r, log_s_trans=log_s_t,
            ema_rot=ema_rot, ema_trans=ema_trn,
        )
        loss.backward()

        # The gradient dL/d(log_s) = -loss_n * exp(-log_s) + 1
        # With loss_n ≈ 1 (normalised), exp(0)=1 → grad ≈ 0 at equilibrium
        # So a single step from zero with lr=0.1 should barely move the scalar
        with torch.no_grad():
            new_r = log_s_r - 0.1 * log_s_r.grad
        assert pytest.approx(new_r.item(), abs=0.5) == 0.0


# ── PoseUncertaintyScales module ──────────────────────────────────────────────

class TestPoseUncertaintyScales:
    def test_initialises_logs_to_zero(self):
        pu = PoseUncertaintyScales()
        assert pu.log_s_rot.item() == pytest.approx(0.0)
        assert pu.log_s_trans.item() == pytest.approx(0.0)

    def test_ema_buffers_initialised(self):
        pu = PoseUncertaintyScales()
        assert pu.ema_rot.item() == pytest.approx(0.01)
        assert pu.ema_trans.item() == pytest.approx(0.0001)

    def test_ema_decay_default(self):
        assert PoseUncertaintyScales().ema_decay == 0.99

    def test_ema_decay_custom(self):
        assert PoseUncertaintyScales(ema_decay=0.95).ema_decay == 0.95

    def test_is_nn_module(self):
        assert isinstance(PoseUncertaintyScales(), nn.Module)

    def test_has_two_params_plus_two_buffers(self):
        pu = PoseUncertaintyScales()
        params = list(pu.parameters())
        buffers = list(pu.buffers())
        assert len(params) == 2
        assert len(buffers) == 2

    def test_params_require_grad_buffers_do_not(self):
        pu = PoseUncertaintyScales()
        for p in pu.parameters():
            assert p.requires_grad
        for b in pu.buffers():
            assert not b.requires_grad

    def test_state_dict_round_trip(self):
        pu = PoseUncertaintyScales()
        pu.log_s_rot.data.fill_(0.42)
        pu.log_s_trans.data.fill_(-1.7)
        pu.ema_rot.fill_(0.003)
        pu.ema_trans.fill_(2e-6)
        sd = pu.state_dict()
        pu2 = PoseUncertaintyScales()
        pu2.load_state_dict(sd)
        assert pu2.log_s_rot.item() == pytest.approx(0.42, rel=1e-5)
        assert pu2.log_s_trans.item() == pytest.approx(-1.7, rel=1e-5)
        assert pu2.ema_rot.item() == pytest.approx(0.003, rel=1e-5)
        assert pu2.ema_trans.item() == pytest.approx(2e-6, rel=1e-5)

    def test_gradients_update_via_optimizer(self):
        pu = PoseUncertaintyScales()
        opt = torch.optim.SGD(pu.parameters(), lr=0.1)
        pred_ax = torch.randn(4, 3)
        pred_tr = torch.randn(4, 3)
        loss = pose_loss(
            pred_ax, pred_tr, torch.zeros(4, 3), torch.zeros(4, 3),
            log_s_rot=pu.log_s_rot, log_s_trans=pu.log_s_trans,
            ema_rot=pu.ema_rot, ema_trans=pu.ema_trans,
        )
        loss.backward()
        before_r = pu.log_s_rot.item()
        before_t = pu.log_s_trans.item()
        opt.step()
        assert pu.log_s_rot.item() != pytest.approx(before_r)
        assert pu.log_s_trans.item() != pytest.approx(before_t)

    def test_ema_manual_update(self):
        """Simulate the EMA update that the trainer performs each step."""
        pu = PoseUncertaintyScales(ema_decay=0.9)
        raw_rot = torch.tensor([0.05])
        raw_trn = torch.tensor([0.002])
        with torch.no_grad():
            d = pu.ema_decay
            pu.ema_rot.mul_(d).add_(raw_rot, alpha=1 - d)
            pu.ema_trans.mul_(d).add_(raw_trn, alpha=1 - d)
        # After one step: new = 0.9*old + 0.1*raw
        assert pu.ema_rot.item() == pytest.approx(0.9 * 0.01 + 0.1 * 0.05, rel=1e-5)
        assert pu.ema_trans.item() == pytest.approx(0.9 * 0.0001 + 0.1 * 0.002, rel=1e-5)

    def test_ema_converges_to_loss_magnitude(self):
        """After many updates with constant loss, EMA approaches that loss."""
        pu = PoseUncertaintyScales(ema_decay=0.99)
        const_rot = torch.tensor([0.04])
        for _ in range(2000):
            with torch.no_grad():
                pu.ema_rot.mul_(0.99).add_(const_rot, alpha=0.01)
        assert pu.ema_rot.item() == pytest.approx(0.04, rel=1e-2)


# ── options: use_uncertainty_pose + ema_decay flags ──────────────────────────

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

    def test_ema_decay_default(self):
        cfg = TrainingConfig(mode="c3vdv2_stage2")
        assert cfg.uncertainty_pose_ema_decay == 0.99

    def test_ema_decay_custom(self):
        cfg = TrainingConfig(mode="c3vdv2_stage2", uncertainty_pose_ema_decay=0.95)
        assert cfg.uncertainty_pose_ema_decay == 0.95
