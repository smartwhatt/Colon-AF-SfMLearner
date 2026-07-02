"""Depth Anything V2 adapter for the AF-SfMLearner SSL pipeline.

The entire SSL stack (trainer ``generate_images_pred``, smoothness loss,
teacher consistency, source replay, all eval tools) consumes sigmoid
**disparity** ``outputs[("disp", scale)]`` and converts it to depth via
``layers.disp_to_depth``. Depth Anything V2 outputs depth directly (metric
or relative depending on the variant). Rather than refactor every consumer,
this adapter wraps a DA-v2 model behind the existing ``("disp", scale)``
interface by inverting ``depth -> disp``:

    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)

where ``min_disp = 1 / max_depth`` and ``max_disp = 1 / min_depth``. This
invert is the exact inverse of ``disp_to_depth``, so when the adapter is
frozen the disp->depth->disp round-trip is identity and metrics are
faithful. When trainable, gradients flow back to DA-v2 through the invert.

Two variants are supported (selected by the ``variant`` constructor arg /
the ``depth_anything_variant`` config field):

- ``"metric_hypersim"`` (default for the metric-sizing experiment): loads
  the official DA-v2 metric checkpoint (raw ``.pth``, e.g.
  ``depth-anything/Depth-Anything-V2-Metric-Hypersim-Large``) via the
  vendored :class:`networks.depth_anything_v2.DepthAnythingV2`. Outputs
  metric depth in metres — directly comparable to C3VD/PolypSense GT.
- ``"relative_hf"``: loads the transformers-compatible relative-depth
  checkpoint (e.g. ``depth-anything/Depth-Anything-V2-Large-hf``) via
  ``transformers.AutoModelForDepthEstimation``. Outputs relative depth
  (no metric scale); for evaluation use ``--median_scaling``, and as a
  frozen SSL anchor it constrains only relative shape, not metric scale.

The adapter is registered under both the ``"encoder"`` and ``"depth"`` keys
by ``StageTwoTrainer.build_models`` (a thin :class:`_EncoderShim` holds the
DA-v2 model under ``"encoder"`` so checkpoint/save/load and
``trainable_modules`` keys still work). The shim's ``forward`` returns the
input image unchanged, so ``_forward_stage2``'s
``features = encoder(color); outputs = depth(features)`` keeps working.
"""

from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# Variant identifiers.
VARIANT_METRIC_HYPERSIM = "metric_hypersim"
VARIANT_RELATIVE_HF = "relative_hf"
_VARIANTS = (VARIANT_METRIC_HYPERSIM, VARIANT_RELATIVE_HF)

# Encoder config for each size. The official DepthAnythingV2 constructor
# takes (encoder, features, out_channels). Large = vitl.
_DA_V2_ENCODER_CONFIG = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}

# Placeholder channel list for the in-house num_ch_enc contract. The
# ResnetEncoder exposes [64,64,128,256,512]; downstream code (specular /
# predictive-mask heads) reads it. Those heads are disallowed for DA-v2, so
# this is only present to keep attribute access from crashing during early
# validation. It is never consumed to build heads when DA-v2 is active.
_DEPTH_ANYTHING_NUM_CH_ENC = [3, 48, 96, 192, 384]

# DA-v2 uses a 14x14 patch grid; inputs must be multiples of 14.
_DA_V2_PATCH_SIZE = 14

# ImageNet normalization (DINOv2 backbone expects this).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def depth_to_disp(depth, min_depth, max_depth):
    """Inverse of :func:`layers.disp_to_depth`.

    Converts a metric depth prediction to a sigmoid-range disparity in
    ``[0, 1]`` (assuming ``depth`` was clamped to ``[min_depth, max_depth]``).
    """
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = 1.0 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp


class _EncoderShim(nn.Module):
    """Holds the DA-v2 model under the ``"encoder"`` key.

    The shim's ``forward`` returns the input image unchanged so that
    ``_forward_stage2`` can call ``depth(features)`` with ``features`` being
    the raw color tensor, which :class:`DepthAnythingAdapter` accepts.
    """

    def __init__(self, dav2_model):
        super().__init__()
        self.dav2 = dav2_model
        self.num_ch_enc = list(_DEPTH_ANYTHING_NUM_CH_ENC)

    def forward(self, color):
        return color


class DepthAnythingAdapter(nn.Module):
    """Wrap a Depth Anything V2 model behind the disp interface.

    Parameters
    ----------
    variant:
        ``"metric_hypersim"`` (official .pth, metric depth in metres) or
        ``"relative_hf"`` (transformers -hf, relative depth). Selects the
        loader and the forward interpretation.
    repo_id:
        Hugging Face repo id. For ``metric_hypersim`` this is the repo
        hosting the raw ``.pth`` (e.g.
        ``depth-anything/Depth-Anything-V2-Metric-Hypersim-Large``); the
        ``.pth`` is downloaded via ``huggingface_hub.hf_hub_download`` and
        loaded into the vendored :class:`DepthAnythingV2`. For
        ``relative_hf`` this is the transformers repo (e.g.
        ``depth-anything/Depth-Anything-V2-Large-hf``), loaded via
        ``AutoModelForDepthEstimation.from_pretrained``. Ignored when
        ``model`` is given.
    model:
        Optional pre-built model (test path; avoids any download). When
        given, ``repo_id`` is ignored and no download/import happens. The
        caller is responsible for the model matching ``variant``'s forward
        contract (see ``_predict_metric_depth``).
    encoder_size:
        DINOv2 backbone size for ``metric_hypersim``: one of
        ``"vits"|"vitb"|"vitl"`` (default ``"vitl"`` for the Large variant).
        Ignored for ``relative_hf``.
    scales:
        Output disparity scales, mirroring ``DepthDecoder``.
    min_depth, max_depth:
        Metric depth range used for the disp<->depth invert. Must match the
        trainer/eval ``min_depth``/``max_depth``.
    frozen:
        Freeze DA-v2 weights (recommended for the known-good-depth
        experiment). Frozen models are kept in ``eval()`` and never flipped
        to ``train()``.
    """

    def __init__(
        self,
        variant=VARIANT_METRIC_HYPERSIM,
        repo_id=None,
        model=None,
        encoder_size="vitl",
        scales=(0, 1, 2, 3),
        min_depth=0.001,
        max_depth=0.1,
        frozen=True,
    ):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS}, got {variant!r}"
            )
        self.variant = variant
        self.encoder_size = encoder_size
        self.scales = tuple(scales)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.frozen = bool(frozen)
        self.num_ch_enc = list(_DEPTH_ANYTHING_NUM_CH_ENC)

        # Validate encoder_size early for the metric variant, even when a
        # pre-built model is injected (so callers can't silently mismatch).
        if variant == VARIANT_METRIC_HYPERSIM and encoder_size not in _DA_V2_ENCODER_CONFIG:
            raise ValueError(
                f"encoder_size must be one of {list(_DA_V2_ENCODER_CONFIG)}, "
                f"got {encoder_size!r}"
            )

        if model is not None:
            self.dav2 = model
        elif variant == VARIANT_METRIC_HYPERSIM:
            self.dav2 = self._load_metric_model(repo_id, encoder_size)
        else:  # relative_hf
            self.dav2 = self._load_relative_hf_model(repo_id)

        if self.frozen:
            self._freeze()

    # -- loaders ------------------------------------------------------------

    def _load_metric_model(self, repo_id, encoder_size):
        """Load the official metric .pth via the vendored DepthAnythingV2."""
        if not repo_id:
            raise ValueError(
                "variant='metric_hypersim' requires `repo_id` (the HF repo "
                "hosting the .pth checkpoint) or a pre-built `model`."
            )
        if encoder_size not in _DA_V2_ENCODER_CONFIG:
            raise ValueError(
                f"encoder_size must be one of {list(_DA_V2_ENCODER_CONFIG)}, "
                f"got {encoder_size!r}"
            )

        from huggingface_hub import hf_hub_download

        from .depth_anything_v2 import DepthAnythingV2

        cfg = _DA_V2_ENCODER_CONFIG[encoder_size]
        model = DepthAnythingV2(**cfg)

        # The metric Hypersim repos host a single .pth file.
        pth_filename = self._metric_pth_filename(repo_id, encoder_size)
        pth_path = hf_hub_download(repo_id=repo_id, filename=pth_filename)
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _metric_pth_filename(repo_id, encoder_size):
        """Map encoder size to the .pth filename in the HF repo.

        The official metric repos use names like
        ``depth_anything_v2_metric_hypersim_vitl.pth``. The size suffix
        matches the encoder (vits/vitb/vitl).
        """
        # Hypersim (indoor) and VirtualNorm (outdoor) both follow this pattern.
        return f"depth_anything_v2_metric_hypersim_{encoder_size}.pth"

    def _load_relative_hf_model(self, repo_id):
        """Load the transformers-compatible relative-depth checkpoint."""
        if not repo_id:
            raise ValueError(
                "variant='relative_hf' requires `repo_id` (the transformers "
                "-hf repo) or a pre-built `model`."
            )
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "variant='relative_hf' requires the `transformers` package. "
                "Install it with `uv add transformers` or "
                "`pip install transformers`."
            ) from exc
        return AutoModelForDepthEstimation.from_pretrained(repo_id)

    # -- freeze discipline --------------------------------------------------

    def _freeze(self):
        for p in self.dav2.parameters():
            p.requires_grad_(False)
        self.dav2.eval()

    def train(self, mode=True):
        """Keep a frozen DA-v2 in eval mode regardless of the trainer state.

        Mirrors the teacher-freeze discipline in ``StageTwoTrainer``: a
        frozen depth model must not be flipped to ``train()`` by
        ``set_train``/``set_eval`` toggles, otherwise BatchNorm/Dropout
        stats drift and gradients accidentally enable.
        """
        if self.frozen:
            self.dav2.eval()
            return self
        return super().train(mode)

    # -- preprocessing ------------------------------------------------------

    @staticmethod
    def _normalize(color):
        """Normalize an RGB image tensor to ImageNet stats for the DINOv2 backbone.

        ``color`` is expected in [0, 1] range, shape (B, 3, H, W). Returns a
        normalized tensor of the same shape.
        """
        mean = torch.tensor(_IMAGENET_MEAN, device=color.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=color.device).view(1, 3, 1, 1)
        return (color - mean) / std

    @staticmethod
    def _pad_to_patch_multiple(color, patch=_DA_V2_PATCH_SIZE):
        """Pad H, W up to the nearest multiple of 14 (DA-v2 patch size).

        DA-v2's DINOv2 backbone requires spatial dims divisible by 14. We
        right/bottom-pad with reflection to avoid border artifacts, run the
        model, then crop the output back to the original size.
        """
        _, _, h, w = color.shape
        new_h = (h + patch - 1) // patch * patch
        new_w = (w + patch - 1) // patch * patch
        if new_h == h and new_w == w:
            return color, (h, w)
        pad_h, pad_w = new_h - h, new_w - w
        padded = F.pad(color, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (h, w)

    # -- forward ------------------------------------------------------------

    @staticmethod
    def _coerce_color(features_or_color):
        """Accept either a raw color tensor or the encoder-shim output.

        The encoder shim returns the image tensor unchanged, so
        ``features`` arriving here is a tensor. For robustness (and to
        match ``DepthDecoder``'s list-of-features contract), also accept a
        list/tuple and take its first element.
        """
        if isinstance(features_or_color, (list, tuple)):
            return features_or_color[0]
        return features_or_color

    def _predict_metric_depth(self, color):
        """Run DA-v2 and return metric depth at the input resolution.

        Returns a ``(B, 1, H, W)`` tensor clamped to
        ``[min_depth, max_depth]``.

        For ``metric_hypersim`` the model outputs metric depth in metres.
        For ``relative_hf`` the model outputs relative depth (no units);
        callers relying on metric scale must use median scaling at eval
        time. Here we still clamp to [min_depth, max_depth] so the
        disp<->depth invert is well-conditioned.
        """
        original_h, original_w = color.shape[-2], color.shape[-1]
        normalized = self._normalize(color)

        if self.variant == VARIANT_METRIC_HYPERSIM:
            # Official DepthAnythingV2.forward needs spatial dims divisible
            # by 14 (patch size). Pad, run, crop.
            padded, (orig_h, orig_w) = self._pad_to_patch_multiple(normalized)
            depth = self.dav2(padded)  # (B, H, W) metric metres
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if depth.shape[-2:] != (orig_h, orig_w):
                depth = depth[..., :orig_h, :orig_w]
        else:  # relative_hf
            out = self.dav2(color)
            depth = out.predicted_depth  # (B, H, W)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)

        if depth.shape[-2:] != (original_h, original_w):
            depth = F.interpolate(
                depth,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )
        depth = depth.clamp(min=self.min_depth, max=self.max_depth)
        return depth

    def forward_depth(self, color):
        """Return raw metric depth ``(B, 1, H, W)`` (eval shortcut).

        Note: for ``relative_hf`` this returns relative depth clamped to the
        configured range, not true metric depth. Eval callers should apply
        ``--median_scaling`` for that variant.
        """
        return self._predict_metric_depth(color)

    def forward(self, features_or_color):
        """Return ``{("disp", s): ...}`` mirroring ``DepthDecoder``.

        Produces full-resolution disp at scale 0 and downsampled disp at
        higher scales (by ``2**s``), matching how ``generate_images_pred``
        interpolates each scale's disp up to full resolution anyway.
        """
        color = self._coerce_color(features_or_color)
        depth = self._predict_metric_depth(color)
        disp_full = depth_to_disp(depth, self.min_depth, self.max_depth)
        # Clamp guards tiny float drift outside [0, 1] from the invert.
        disp_full = disp_full.clamp(0.0, 1.0)

        outputs = {}
        for s in self.scales:
            if s == 0:
                outputs[("disp", s)] = disp_full
            else:
                outputs[("disp", s)] = F.interpolate(
                    disp_full,
                    scale_factor=1.0 / (2 ** s),
                    mode="bilinear",
                    align_corners=False,
                )
        return outputs


def build_depth_anything_pair(
    variant=VARIANT_METRIC_HYPERSIM,
    repo_id=None,
    model=None,
    encoder_size="vitl",
    scales=(0, 1, 2, 3),
    min_depth=0.001,
    max_depth=0.1,
    frozen=True,
):
    """Build the (encoder_shim, depth_adapter) pair for ``build_models``.

    Returns ``(encoder, depth)`` where ``encoder`` is an :class:`_EncoderShim`
    holding the DA-v2 model and ``depth`` is the
    :class:`DepthAnythingAdapter` sharing the same underlying model object
    (so there is exactly one copy of the weights).
    """
    adapter = DepthAnythingAdapter(
        variant=variant,
        repo_id=repo_id,
        model=model,
        encoder_size=encoder_size,
        scales=scales,
        min_depth=min_depth,
        max_depth=max_depth,
        frozen=frozen,
    )
    encoder = _EncoderShim(adapter.dav2)
    return encoder, adapter
