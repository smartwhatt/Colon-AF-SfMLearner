"""Depth Anything V2 adapter for the AF-SfMLearner SSL pipeline.

The entire SSL stack (trainer ``generate_images_pred``, smoothness loss,
teacher consistency, source replay, all eval tools) consumes sigmoid
**disparity** ``outputs[("disp", scale)]`` and converts it to depth via
``layers.disp_to_depth``. Depth Anything V2 outputs **metric depth**
directly. Rather than refactor every consumer, this adapter wraps a HF
DA-v2 model behind the existing ``("disp", scale)`` interface by inverting
``depth -> disp``:

    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)

where ``min_disp = 1 / max_depth`` and ``max_disp = 1 / min_depth``. This
invert is the exact inverse of ``disp_to_depth``, so when the adapter is
frozen the disp->depth->disp round-trip is identity and metrics are
faithful. When trainable, gradients flow back to DA-v2 through the invert.

The adapter is registered under both the ``"encoder"`` and ``"depth"`` keys
by ``StageTwoTrainer.build_models`` (a thin :class:`_EncoderShim` holds the
DA-v2 model under ``"encoder"`` so checkpoint/save/load and
``trainable_modules`` keys still work). The shim's ``forward`` returns the
input image unchanged, so ``_forward_stage2``'s
``features = encoder(color); outputs = depth(features)`` keeps working.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# Placeholder channel list. The in-house ResnetEncoder exposes
# ``num_ch_enc = [64, 64, 128, 256, 512]``; downstream code (specular /
# predictive-mask heads) reads it. Those heads are disallowed for DA-v2, so
# this value is only present to keep attribute access from crashing during
# early validation. It is never consumed to build heads when DA-v2 is active.
_DEPTH_ANYTHING_NUM_CH_ENC = [3, 48, 96, 192, 384]


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
    """Wrap a HF Depth Anything V2 model behind the disp interface.

    Parameters
    ----------
    hf_repo_id:
        Hugging Face repo id for DA-v2 (e.g.
        ``"depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"``).
        Ignored when ``model`` is given.
    model:
        Optional pre-built model (test path; avoids a HF download). When
        given, ``hf_repo_id`` is ignored and no ``transformers`` import is
        performed.
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
        hf_repo_id=None,
        model=None,
        scales=(0, 1, 2, 3),
        min_depth=0.001,
        max_depth=0.1,
        frozen=True,
    ):
        super().__init__()
        self.scales = tuple(scales)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.frozen = bool(frozen)
        self.num_ch_enc = list(_DEPTH_ANYTHING_NUM_CH_ENC)

        if model is not None:
            self.dav2 = model
        else:
            if not hf_repo_id:
                raise ValueError(
                    "DepthAnythingAdapter requires either `model` or `hf_repo_id`"
                )
            try:
                from transformers import AutoModelForDepthEstimation
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "depth_backend='depth_anything' requires the `transformers` "
                    "package. Install it with `uv add transformers` or "
                    "`pip install transformers`."
                ) from exc
            self.dav2 = AutoModelForDepthEstimation.from_pretrained(hf_repo_id)

        if self.frozen:
            self._freeze()

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
        """
        out = self.dav2(color)
        # HF DA-v2 exposes ``predicted_depth`` (B, H, W) -> (B, 1, H, W).
        depth = out.predicted_depth
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[-2:] != color.shape[-2:]:
            depth = F.interpolate(
                depth,
                size=color.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        depth = depth.clamp(min=self.min_depth, max=self.max_depth)
        return depth

    def forward_depth(self, color):
        """Return raw metric depth ``(B, 1, H, W)`` (eval shortcut)."""
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
    hf_repo_id=None,
    model=None,
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
        hf_repo_id=hf_repo_id,
        model=model,
        scales=scales,
        min_depth=min_depth,
        max_depth=max_depth,
        frozen=frozen,
    )
    encoder = _EncoderShim(adapter.dav2)
    return encoder, adapter
