"""Unit tests for PPS geometry and supervised PPS loss (Phase 1 PR 2)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layers import compute_pps_map, compute_surface_normals_from_depth
from trainers.c3vdv2_losses import pps_supervised_loss


def make_inv_k(batch=1, height=16, width=20, fx=80.0, fy=80.0):
    k = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    k[:, 0, 0] = fx
    k[:, 1, 1] = fy
    k[:, 0, 2] = (width - 1) / 2.0
    k[:, 1, 2] = (height - 1) / 2.0
    return torch.linalg.inv(k)


def sloped_depth(batch=1, height=16, width=20):
    x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    depth = 0.02 + 0.004 * x + 0.002 * y
    return depth.repeat(batch, 1, 1, 1)


def test_surface_normals_flat_plane_are_finite_and_camera_facing():
    height, width = 16, 20
    depth = torch.full((1, 1, height, width), 0.02)
    normals = compute_surface_normals_from_depth(depth, make_inv_k(1, height, width))

    assert normals.shape == (1, 3, height, width)
    assert torch.isfinite(normals).all()
    assert normals[:, 2, height // 2, width // 2] < -0.99


def test_surface_normals_tilted_plane_are_finite():
    height, width = 16, 20
    depth = sloped_depth(2, height, width)
    normals = compute_surface_normals_from_depth(depth, make_inv_k(2, height, width))

    assert normals.shape == (2, 3, height, width)
    assert torch.isfinite(normals).all()


def test_pps_map_accepts_single_and_batched_light_positions():
    height, width = 16, 20
    depth = sloped_depth(2, height, width)
    inv_k = make_inv_k(2, height, width)

    single_light = torch.tensor([0.0015, 0.0, 0.0])
    batched_light = torch.tensor([[0.0015, 0.0, 0.0], [-0.0015, 0.0, 0.0]])

    pps_single = compute_pps_map(depth, inv_k, single_light)
    pps_batch = compute_pps_map(depth, inv_k, batched_light)

    assert pps_single.shape == (2, 1, height, width)
    assert pps_batch.shape == (2, 1, height, width)
    assert torch.isfinite(pps_single).all()
    assert torch.isfinite(pps_batch).all()
    assert pps_single.sum() > 0
    assert pps_batch.sum() > 0


def test_pps_supervised_loss_backpropagates_to_pred_depth_only():
    height, width = 16, 20
    gt_depth = sloped_depth(1, height, width).requires_grad_(True)
    pred_depth = (gt_depth.detach() + 0.001 * torch.randn_like(gt_depth)).requires_grad_(True)
    valid = torch.ones_like(gt_depth)
    inv_k = make_inv_k(1, height, width)
    light = torch.tensor([0.0015, 0.0, 0.0])

    loss = pps_supervised_loss(pred_depth, gt_depth, inv_k, valid, light)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_depth.grad is not None
    assert pred_depth.grad.abs().sum() > 0
    assert gt_depth.grad is None


def test_pps_supervised_loss_handles_empty_mask():
    height, width = 8, 10
    gt_depth = sloped_depth(1, height, width)
    pred_depth = gt_depth.clone().requires_grad_(True)
    valid = torch.zeros_like(gt_depth)
    inv_k = make_inv_k(1, height, width)

    loss = pps_supervised_loss(pred_depth, gt_depth, inv_k, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert pred_depth.grad is not None
    assert pred_depth.grad.abs().sum() == 0.0
