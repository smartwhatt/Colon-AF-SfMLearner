"""Unit tests for SpecularDecoder and specular loss helpers (PR 1)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

from networks.resnet_encoder import ResnetEncoder
from networks.specular_decoder import SpecularDecoder
from trainers.c3vdv2_losses import compute_specular_pseudo_labels, specular_bce_loss


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def encoder():
    enc = ResnetEncoder(num_layers=18, pretrained=False)
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def decoder(encoder):
    dec = SpecularDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    dec.eval()
    return dec


# ── SpecularDecoder shape tests ───────────────────────────────────────────────

class TestSpecularDecoderShapes:
    def test_output_keys_exist(self, encoder, decoder):
        x = torch.zeros(2, 3, 256, 320)
        with torch.no_grad():
            feats = encoder(x)
            out = decoder(feats)
        for s in range(4):
            assert ("specular", s) in out, f"missing key ('specular', {s})"

    def test_scale_0_full_resolution(self, encoder, decoder):
        B, H, W = 2, 256, 320
        x = torch.zeros(B, 3, H, W)
        with torch.no_grad():
            out = decoder(encoder(x))
        s0 = out[("specular", 0)]
        assert s0.shape == (B, 1, H, W), f"unexpected shape {s0.shape}"

    def test_scale_half_resolution(self, encoder, decoder):
        B, H, W = 2, 256, 320
        x = torch.zeros(B, 3, H, W)
        with torch.no_grad():
            out = decoder(encoder(x))
        assert out[("specular", 1)].shape == (B, 1, H // 2, W // 2)

    def test_output_range_sigmoid(self, encoder, decoder):
        x = torch.randn(2, 3, 256, 320)
        with torch.no_grad():
            out = decoder(encoder(x))
        s0 = out[("specular", 0)]
        assert s0.min() >= 0.0 and s0.max() <= 1.0, "output outside [0,1]"

    def test_output_finite(self, encoder, decoder):
        x = torch.randn(2, 3, 256, 320)
        with torch.no_grad():
            out = decoder(encoder(x))
        for s in range(4):
            assert torch.isfinite(out[("specular", s)]).all(), \
                f"non-finite values at scale {s}"

    def test_single_batch(self, encoder, decoder):
        x = torch.zeros(1, 3, 256, 320)
        with torch.no_grad():
            out = decoder(encoder(x))
        assert out[("specular", 0)].shape == (1, 1, 256, 320)


# ── compute_specular_pseudo_labels ────────────────────────────────────────────

class TestPseudoLabels:
    def test_output_shape(self):
        img = torch.rand(2, 3, 64, 80)
        labels = compute_specular_pseudo_labels(img)
        assert labels.shape == (2, 1, 64, 80)

    def test_all_saturated_image(self):
        img = torch.ones(1, 3, 32, 32)  # all pixels = 1.0 (saturated)
        labels = compute_specular_pseudo_labels(img, threshold=0.98)
        assert labels.sum() == labels.numel(), "all saturated pixels should be labelled"

    def test_all_dark_excluded(self):
        img = torch.zeros(1, 3, 32, 32)  # all pixels = 0.0 (dark/invalid)
        labels = compute_specular_pseudo_labels(img, threshold=0.98)
        assert labels.sum() == 0, "dark pixels should not be labelled specular"

    def test_values_binary(self):
        img = torch.rand(2, 3, 64, 80)
        labels = compute_specular_pseudo_labels(img)
        unique = labels.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0}), \
            f"labels should be binary, got {unique}"

    def test_threshold_respected(self):
        img = torch.full((1, 3, 8, 8), 0.5)  # mean intensity = 0.5
        labels = compute_specular_pseudo_labels(img, threshold=0.98)
        assert labels.sum() == 0, "0.5 intensity should not exceed 0.98 threshold"


# ── specular_bce_loss ─────────────────────────────────────────────────────────

class TestSpecularBCELoss:
    def test_loss_finite(self):
        pred = torch.full((2, 1, 64, 80), 0.5)
        labels = torch.zeros(2, 1, 64, 80)
        loss = specular_bce_loss(pred, labels)
        assert torch.isfinite(loss)

    def test_perfect_prediction_low_loss(self):
        eps = 1e-6
        pred = torch.full((1, 1, 16, 16), eps)       # near 0
        labels = torch.zeros(1, 1, 16, 16)            # target 0
        loss = specular_bce_loss(pred, labels)
        assert loss < 0.01, f"perfect pred should yield near-zero loss, got {loss.item()}"

    def test_with_valid_mask(self):
        pred = torch.full((2, 1, 32, 32), 0.5)
        labels = torch.ones(2, 1, 32, 32)
        # Mask that excludes all pixels
        empty_mask = torch.zeros(2, 1, 32, 32)
        # Should not divide by zero
        loss = specular_bce_loss(pred, labels, valid_mask=empty_mask)
        assert torch.isfinite(loss)

    def test_scalar_output(self):
        pred = torch.rand(2, 1, 64, 80).clamp(1e-6, 1 - 1e-6)
        labels = (torch.rand(2, 1, 64, 80) > 0.5).float()
        loss = specular_bce_loss(pred, labels)
        assert loss.ndim == 0, "loss should be a scalar"


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_gradients_flow_to_decoder(self, encoder, decoder):
        """Specular decoder parameters should receive gradients via BCE loss."""
        enc = ResnetEncoder(num_layers=18, pretrained=False)
        dec = SpecularDecoder(num_ch_enc=enc.num_ch_enc, scales=[0])

        x = torch.randn(1, 3, 256, 320)
        feats = enc(x)
        out = dec(feats)

        pred = out[("specular", 0)]
        labels = compute_specular_pseudo_labels(
            torch.rand(1, 3, 256, 320), threshold=0.5
        )
        loss = specular_bce_loss(pred, labels)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dec.parameters()
        )
        assert has_grad, "no gradients reached decoder parameters"
