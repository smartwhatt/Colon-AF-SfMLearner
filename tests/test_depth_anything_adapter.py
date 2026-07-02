"""Tests for the Depth Anything V2 adapter.

These tests do NOT download HF weights and do NOT require the
``transformers`` package at import time: a tiny stub model implementing the
HF DA-v2 output interface (``.predicted_depth``) is injected via the
adapter's ``model=`` constructor argument. This matches the AGENTS.md
guidance to avoid tests that download HF data.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.depth_anything_adapter import (  # noqa: E402
    DepthAnythingAdapter,
    build_depth_anything_pair,
    depth_to_disp,
)
from layers import disp_to_depth  # noqa: E402
from trainers.stage_two import StageTwoTrainer  # noqa: E402


MIN_DEPTH = 0.001
MAX_DEPTH = 0.1


class _StubDAv2Output:
    """Mimics the HF transformers DA-v2 forward output (``.predicted_depth``)."""

    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth


class _StubMetricDAv2(nn.Module):
    """Stub for the official metric DepthAnythingV2 forward contract.

    ``forward(color)`` returns a ``(B, 1, H, W)`` metric-depth tensor
    directly (no wrapper object), matching the vendored
    :class:`networks.depth_anything_v2.DepthAnythingV2`. The constant is
    inside ``[min_depth, max_depth]`` so the depth->disp invert is
    well-conditioned.
    """

    def __init__(self, value=0.05):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self._value = float(value)

    def forward(self, color):
        b, _, h, w = color.shape
        return torch.full((b, 1, h, w), self._value, dtype=color.dtype)


class _StubRelativeHFDAv2(nn.Module):
    """Stub for the transformers ``-hf`` relative-depth forward contract.

    ``forward(color)`` returns a wrapper with ``.predicted_depth``, matching
    ``AutoModelForDepthEstimation`` output.
    """

    def __init__(self, value=0.05):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self._value = float(value)

    def forward(self, color):
        b, _, h, w = color.shape
        depth = torch.full((b, 1, h, w), self._value, dtype=color.dtype)
        return _StubDAv2Output(depth)


def _make_adapter(
    variant="metric_hypersim",
    frozen=True,
    value=0.05,
    scales=(0, 1, 2, 3),
):
    """Build an adapter with a stub model matching the variant's contract."""
    if variant == "metric_hypersim":
        model = _StubMetricDAv2(value=value)
    else:  # relative_hf
        model = _StubRelativeHFDAv2(value=value)
    return DepthAnythingAdapter(
        variant=variant,
        model=model,
        scales=scales,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        frozen=frozen,
    )


def _color(batch=2, h=64, w=64):
    return torch.rand(batch, 3, h, w)


# ---------------------------------------------------------------------------
# disp <-> depth invert round-trip
# ---------------------------------------------------------------------------


def test_depth_to_disp_is_exact_inverse_of_disp_to_depth():
    disp = torch.rand(4, 1, 8, 8)
    _, depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
    recovered = depth_to_disp(depth, MIN_DEPTH, MAX_DEPTH)
    assert torch.allclose(disp, recovered, atol=1e-6)


# ---------------------------------------------------------------------------
# forward / forward_depth
# ---------------------------------------------------------------------------


def test_forward_returns_disp_keys_for_all_scales():
    adapter = _make_adapter(scales=(0, 1, 2, 3))
    out = adapter(_color())
    for s in (0, 1, 2, 3):
        assert ("disp", s) in out
        assert out[("disp", s)].shape[0] == 2
        assert out[("disp", s)].shape[1] == 1


def test_forward_disp_round_trips_to_metric_depth():
    """Frozen adapter: disp -> depth should recover the DA-v2 metric depth."""
    value = 0.05
    adapter = _make_adapter(value=value)
    color = _color()
    out = adapter(color)
    disp = out[("disp", 0)]
    _, depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
    assert torch.allclose(depth, torch.full_like(depth, value), atol=1e-5)


def test_forward_depth_returns_metric_depth_in_range():
    adapter = _make_adapter(value=0.02)
    depth = adapter.forward_depth(_color())
    assert depth.shape == (2, 1, 64, 64)
    assert torch.all(depth >= MIN_DEPTH)
    assert torch.all(depth <= MAX_DEPTH)


def test_forward_accepts_list_input_like_depth_decoder():
    """The adapter must accept a list (encoder features) for API parity."""
    adapter = _make_adapter()
    color = _color()
    out = adapter([color])  # list-of-features contract
    assert ("disp", 0) in out


# ---------------------------------------------------------------------------
# freeze discipline
# ---------------------------------------------------------------------------


def test_frozen_adapter_has_no_grad_and_stays_eval():
    adapter = _make_adapter(frozen=True)
    assert all(not p.requires_grad for p in adapter.dav2.parameters())
    adapter.train()  # trainer set_train may call this
    assert not adapter.dav2.training


def test_trainable_adapter_allows_grad():
    adapter = _make_adapter(frozen=False)
    assert all(p.requires_grad for p in adapter.dav2.parameters())


def test_build_pair_shares_single_dav2_instance():
    encoder, depth = build_depth_anything_pair(
        variant="metric_hypersim",
        model=_StubMetricDAv2(),
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        frozen=True,
    )
    assert encoder.dav2 is depth.dav2
    assert encoder.num_ch_enc == depth.num_ch_enc


# ---------------------------------------------------------------------------
# variant selection
# ---------------------------------------------------------------------------


def test_metric_variant_forward_returns_disp_and_round_trips():
    """metric_hypersim: forward -> disp -> depth recovers the metric value."""
    value = 0.05
    adapter = _make_adapter(variant="metric_hypersim", value=value)
    out = adapter(_color())
    disp = out[("disp", 0)]
    _, depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
    assert torch.allclose(depth, torch.full_like(depth, value), atol=1e-5)


def test_relative_hf_variant_forward_uses_model_output_directly_as_disp():
    """relative_hf: the model outputs sigmoid disparity, used directly as disp.

    Unlike metric_hypersim (which inverts metric depth -> disp), the relative
    -hf model's ``predicted_depth`` is ALREADY a sigmoid disparity in (0,1).
    The adapter must pass it through unchanged — clamping/inverting it as
    though it were metric depth would destroy the relative shape.
    """
    value = 0.3  # a plausible sigmoid-disparity value
    adapter = _make_adapter(variant="relative_hf", value=value)
    out = adapter(_color())
    disp = out[("disp", 0)]
    # disp should equal the model's raw output, not a depth->disp invert.
    assert torch.allclose(disp, torch.full_like(disp, value), atol=1e-6)


def test_relative_hf_forward_depth_converts_disp_to_depth():
    """relative_hf: forward_depth converts the sigmoid disp to (relative) depth."""
    value = 0.3
    adapter = _make_adapter(variant="relative_hf", value=value)
    depth = adapter.forward_depth(_color())
    # depth = disp_to_depth(value); verify it matches the SSL pipeline's conversion.
    from layers import disp_to_depth
    expected = disp_to_depth(
        torch.full((1, 1, 1, 1), value), MIN_DEPTH, MAX_DEPTH
    )[1]
    assert torch.allclose(depth, expected.expand_as(depth), atol=1e-6)
    assert depth.shape == (2, 1, 64, 64)


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="variant must be one of"):
        DepthAnythingAdapter(variant="bogus", model=_StubMetricDAv2())


def test_metric_variant_requires_repo_id_when_no_model():
    with pytest.raises(ValueError, match="repo_id"):
        DepthAnythingAdapter(variant="metric_hypersim", repo_id=None)


def test_relative_hf_variant_requires_repo_id_when_no_model():
    with pytest.raises(ValueError, match="repo_id"):
        DepthAnythingAdapter(variant="relative_hf", repo_id=None)


def test_metric_variant_bad_encoder_size_raises():
    with pytest.raises(ValueError, match="encoder_size"):
        DepthAnythingAdapter(
            variant="metric_hypersim",
            model=_StubMetricDAv2(),
            encoder_size="vitxl",
        )


def test_metric_variant_pads_non_patch_multiple_input():
    """metric_hypersim pads H,W to multiples of 14 then crops back."""
    adapter = _make_adapter(variant="metric_hypersim", value=0.05)
    # 64 is not a multiple of 14 (64/14 = 4.57 -> pads to 70).
    depth = adapter.forward_depth(_color(h=64, w=64))
    assert depth.shape[-2:] == (64, 64)


# ---------------------------------------------------------------------------
# trainer guards
# ---------------------------------------------------------------------------


def _trainer_stub(depth_backend, depth_anything_frozen=True, **opt_kwargs):
    trainer = object.__new__(StageTwoTrainer)
    trainer.device = torch.device("cpu")
    trainer.models = {}
    trainer.parameters_to_train = []
    defaults = dict(
        depth_backend=depth_backend,
        depth_anything_frozen=depth_anything_frozen,
        depth_anything_repo="stub/repo",
        pose_model_type="separate_resnet",
        use_specular_net=False,
        predictive_mask=False,
    )
    defaults.update(opt_kwargs)
    trainer.opt = SimpleNamespace(**defaults)
    return trainer


def test_validate_depth_backend_allows_separate_resnet():
    trainer = _trainer_stub("depth_anything", pose_model_type="separate_resnet")
    # Should not raise.
    StageTwoTrainer._validate_depth_backend(trainer)


def test_validate_depth_backend_rejects_shared_pose():
    trainer = _trainer_stub("depth_anything", pose_model_type="shared")
    with pytest.raises(ValueError, match="shared"):
        StageTwoTrainer._validate_depth_backend(trainer)


def test_validate_depth_backend_rejects_specular_net():
    trainer = _trainer_stub("depth_anything", use_specular_net=True)
    with pytest.raises(ValueError, match="specular"):
        StageTwoTrainer._validate_depth_backend(trainer)


def test_validate_depth_backend_rejects_predictive_mask():
    trainer = _trainer_stub("depth_anything", predictive_mask=True)
    with pytest.raises(ValueError, match="predictive_mask"):
        StageTwoTrainer._validate_depth_backend(trainer)


def test_validate_depth_backend_rejects_missing_repo():
    trainer = _trainer_stub("depth_anything", depth_anything_repo=None)
    with pytest.raises(ValueError, match="depth_anything_repo"):
        StageTwoTrainer._validate_depth_backend(trainer)


def test_validate_depth_backend_noop_for_af_sfm():
    trainer = _trainer_stub("af_sfm", depth_anything_repo=None)
    # af_sfm has no DA-v2 constraints; should not raise even without repo.
    StageTwoTrainer._validate_depth_backend(trainer)


def test_should_skip_save_skips_frozen_dav2_keys():
    trainer = _trainer_stub("depth_anything", depth_anything_frozen=True)
    assert StageTwoTrainer._should_skip_save(trainer, "encoder") is True
    assert StageTwoTrainer._should_skip_save(trainer, "depth") is True
    assert StageTwoTrainer._should_skip_save(trainer, "pose") is False


def test_should_skip_save_does_not_skip_trainable_dav2():
    trainer = _trainer_stub("depth_anything", depth_anything_frozen=False)
    assert StageTwoTrainer._should_skip_save(trainer, "encoder") is False
    assert StageTwoTrainer._should_skip_save(trainer, "depth") is False


def test_should_skip_save_does_not_skip_af_sfm():
    trainer = _trainer_stub("af_sfm", depth_anything_frozen=True)
    assert StageTwoTrainer._should_skip_save(trainer, "encoder") is False


def test_freeze_depth_anything_sets_no_grad_and_eval():
    adapter = _make_adapter(frozen=True)
    encoder, depth = build_depth_anything_pair(
        model=adapter.dav2, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, frozen=True
    )
    trainer = _trainer_stub("depth_anything", depth_anything_frozen=True)
    trainer.models = {"encoder": encoder, "depth": depth}
    StageTwoTrainer._freeze_depth_anything(trainer)
    for name in ("encoder", "depth"):
        assert all(not p.requires_grad for p in trainer.models[name].parameters())
        assert not trainer.models[name].dav2.training


def test_freeze_depth_anything_noop_when_trainable():
    adapter = _make_adapter(frozen=False)
    encoder, depth = build_depth_anything_pair(
        model=adapter.dav2, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, frozen=False
    )
    trainer = _trainer_stub("depth_anything", depth_anything_frozen=False)
    trainer.models = {"encoder": encoder, "depth": depth}
    StageTwoTrainer._freeze_depth_anything(trainer)
    # Trainable: grad should remain enabled.
    assert all(p.requires_grad for p in trainer.models["depth"].dav2.parameters())


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------


def test_depth_backend_config_fields_parse():
    from options import TrainingConfig

    cfg = TrainingConfig.from_args(
        [
            "--depth_backend", "depth_anything",
            "--depth_anything_repo", "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
            "--depth_anything_frozen",
            "--depth_anything_variant", "metric_hypersim",
            "--depth_anything_encoder_size", "vitl",
        ]
    )
    assert cfg.depth_backend == "depth_anything"
    assert cfg.depth_anything_repo == "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
    assert cfg.depth_anything_frozen is True
    assert cfg.depth_anything_variant == "metric_hypersim"
    assert cfg.depth_anything_encoder_size == "vitl"


def test_depth_backend_defaults_preserve_af_sfm():
    from options import TrainingConfig

    cfg = TrainingConfig.from_args([])
    assert cfg.depth_backend == "af_sfm"
    assert cfg.depth_anything_repo is None
    assert cfg.depth_anything_frozen is True
    assert cfg.depth_anything_variant == "metric_hypersim"
    assert cfg.depth_anything_encoder_size == "vitl"


def test_relative_hf_variant_config_parses():
    from options import TrainingConfig

    cfg = TrainingConfig.from_args(
        [
            "--depth_backend", "depth_anything",
            "--depth_anything_repo", "depth-anything/Depth-Anything-V2-Large-hf",
            "--depth_anything_variant", "relative_hf",
        ]
    )
    assert cfg.depth_anything_variant == "relative_hf"

