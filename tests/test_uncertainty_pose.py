"""Unit tests for B.1 — Sigmoid-Bounded Pose Balance Scalar."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from trainers.c3vdv2_losses import pose_loss
from trainers.stage_two import PoseBalanceScalar
from options import TrainingConfig

# ── pose_loss: fixed-weight path (backward compat) ───────────────────────────


class TestPoseLossFixedWeights:
    def test_zero_error_gives_zero_loss(self):
        x = torch.randn(4, 3)
        loss = pose_loss(x, x, x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_finite(self):
        loss = pose_loss(
            torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
        )
        assert torch.isfinite(loss)

    def test_loss_is_scalar(self):
        loss = pose_loss(
            torch.randn(2, 3), torch.randn(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)
        )
        assert loss.ndim == 0

    def test_rot_weight_scales(self):
        ax, gt = torch.ones(1, 3), torch.zeros(1, 3)
        tr = torch.zeros(1, 3)
        l1 = pose_loss(ax, tr, gt, gt, rot_weight=1.0)
        l2 = pose_loss(ax, tr, gt, gt, rot_weight=2.0)
        assert l2.item() == pytest.approx(2.0 * l1.item(), rel=1e-5)

    def test_gradients_flow_to_predictions(self):
        pred_ax = torch.randn(4, 3, requires_grad=True)
        pred_tr = torch.randn(4, 3, requires_grad=True)
        loss = pose_loss(pred_ax, pred_tr, torch.randn(4, 3), torch.randn(4, 3))
        loss.backward()
        assert pred_ax.grad is not None and pred_ax.grad.abs().sum() > 0
        assert pred_tr.grad is not None and pred_tr.grad.abs().sum() > 0

    def test_balance_none_uses_fixed_weights(self):
        a, b = torch.randn(2, 3), torch.randn(2, 3)
        l_fixed = pose_loss(a, b, torch.zeros_like(a), torch.zeros_like(b))
        l_none = pose_loss(a, b, torch.zeros_like(a), torch.zeros_like(b), balance=None)
        assert l_fixed.item() == pytest.approx(l_none.item(), rel=1e-6)


# ── pose_loss: balance-scalar path ───────────────────────────────────────────


class TestPoseLossBalance:
    def test_init_alpha_half_gives_equal_weighting(self):
        """balance=0 → α=0.5 → rot and trans weighted equally."""
        ax = torch.ones(4, 3)
        tr = torch.full((4, 3), 2.0)
        gt = torch.zeros(4, 3)
        loss = pose_loss(
            ax, tr, gt, gt, balance=torch.tensor([0.0], requires_grad=True)
        )
        # rot_mse = 1.0, trans_mse = 4.0, loss = 0.5*1 + 0.5*4 = 2.5
        assert loss.item() == pytest.approx(2.5, rel=1e-5)

    def test_alpha_near_one_weights_rotation_more(self):
        """Large positive balance → α ≈ 1 → rotation dominates."""
        ax = torch.ones(4, 3)
        tr = torch.full((4, 3), 2.0)
        gt = torch.zeros(4, 3)
        loss = pose_loss(
            ax, tr, gt, gt, balance=torch.tensor([10.0], requires_grad=True)
        )
        # α ≈ 0.99995, loss ≈ 0.99995*1 + 0.00005*4 ≈ 1.0
        assert loss.item() < 1.1

    def test_alpha_near_zero_weights_translation_more(self):
        """Large negative balance → α ≈ 0 → translation dominates."""
        ax = torch.ones(4, 3)
        tr = torch.full((4, 3), 2.0)
        gt = torch.zeros(4, 3)
        loss = pose_loss(
            ax, tr, gt, gt, balance=torch.tensor([-10.0], requires_grad=True)
        )
        # α ≈ 0.00005, loss ≈ 0.00005*1 + 0.99995*4 ≈ 4.0
        assert loss.item() > 3.9

    def test_loss_finite(self):
        loss = pose_loss(
            torch.randn(4, 3),
            torch.randn(4, 3),
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            balance=torch.tensor([0.0], requires_grad=True),
        )
        assert torch.isfinite(loss)

    def test_loss_is_scalar(self):
        loss = pose_loss(
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            balance=torch.tensor([0.0], requires_grad=True),
        )
        assert loss.ndim == 0

    def test_gradient_flows_to_balance(self):
        balance = torch.tensor([0.0], requires_grad=True)
        loss = pose_loss(
            torch.randn(4, 3),
            torch.randn(4, 3),
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            balance=balance,
        )
        loss.backward()
        assert balance.grad is not None and balance.grad.abs().sum() > 0

    def test_gradients_flow_to_predictions(self):
        pred_ax = torch.randn(4, 3, requires_grad=True)
        pred_tr = torch.randn(4, 3, requires_grad=True)
        loss = pose_loss(
            pred_ax,
            pred_tr,
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            balance=torch.tensor([0.0], requires_grad=True),
        )
        loss.backward()
        assert pred_ax.grad is not None and pred_ax.grad.abs().sum() > 0
        assert pred_tr.grad is not None and pred_tr.grad.abs().sum() > 0

    def test_zero_error_zero_loss_regardless_of_balance(self):
        """When pred == gt, total loss is 0 regardless of balance value."""
        x = torch.randn(4, 3)
        for b in [-5.0, 0.0, 5.0]:
            loss = pose_loss(x, x, x, x, balance=torch.tensor([b]))
            assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_balance_converges_toward_larger_component(self):
        """Gradient pushes balance toward the component with larger MSE."""
        balance = torch.tensor([0.0], requires_grad=True)
        # rot: ones vs zeros → mse=1.0,  trans: 10*ones vs zeros → mse=100.0
        loss = pose_loss(
            torch.ones(4, 3),
            torch.full((4, 3), 10.0),
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            balance=balance,
        )
        loss.backward()
        # dL/d(balance) = α(1-α) * (rot_mse - trans_mse)
        # rot_mse - trans_mse = 1 - 100 = -99 < 0 → gradient negative
        # Negative gradient → balance decreases → α decreases → trans weighted more
        assert balance.grad.item() < 0


# ── PoseBalanceScalar module ─────────────────────────────────────────────────


class TestPoseBalanceScalar:
    def test_initialises_to_zero(self):
        pb = PoseBalanceScalar()
        assert pb.balance.item() == pytest.approx(0.0)

    def test_alpha_starts_at_half(self):
        pb = PoseBalanceScalar()
        assert torch.sigmoid(pb.balance).item() == pytest.approx(0.5)

    def test_is_nn_module(self):
        assert isinstance(PoseBalanceScalar(), nn.Module)

    def test_has_exactly_one_parameter(self):
        assert len(list(PoseBalanceScalar().parameters())) == 1

    def test_parameter_requires_grad(self):
        for p in PoseBalanceScalar().parameters():
            assert p.requires_grad

    def test_state_dict_round_trip(self):
        pb = PoseBalanceScalar()
        pb.balance.data.fill_(0.42)
        pb2 = PoseBalanceScalar()
        pb2.load_state_dict(pb.state_dict())
        assert pb2.balance.item() == pytest.approx(0.42, rel=1e-5)

    def test_gradient_updates_via_optimizer(self):
        pb = PoseBalanceScalar()
        opt = torch.optim.SGD(pb.parameters(), lr=0.1)
        loss = pose_loss(
            torch.randn(4, 3),
            torch.randn(4, 3),
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            balance=pb.balance,
        )
        loss.backward()
        before = pb.balance.item()
        opt.step()
        assert pb.balance.item() != pytest.approx(before)


# ── options ──────────────────────────────────────────────────────────────────


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
