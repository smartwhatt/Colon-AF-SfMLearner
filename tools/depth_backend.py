"""Shared depth-backend loader for evaluation tools.

Evaluation tools (evaluate_c3vdv2, evaluate_polysense3d_sparse_depth,
qualitative_realcolon_panel) historically hardcode the AF-SfMLearner
``ResnetEncoder + DepthDecoder`` (sigmoid-disparity) model. This module
generalizes that into a small backend abstraction so the same tools can
also evaluate Depth Anything V2 (metric depth) loaded via ``transformers``,
which is the path for the "DA-v2 vs paper" comparison.

A backend is identified by ``depth_backend`` ∈ {"af_sfm", "depth_anything"}.
``load_depth_model`` returns an opaque model handle; ``predict_depth_m``
returns metric depth in metres as a ``(B, 1, H, W)`` tensor regardless of
backend, so callers do not need to know whether the underlying model emits
disparity or metric depth.
"""

from __future__ import absolute_import, division, print_function

import os

import torch

from networks import DepthDecoder, ResnetEncoder


def load_depth_model(
    *,
    depth_backend: str,
    weights_folder,
    device: torch.device,
    min_depth: float = 0.001,
    max_depth: float = 0.1,
    depth_anything_repo: str = None,
    num_layers: int = 18,
):
    """Build and load a depth model for evaluation.

    Returns a backend-specific handle:

    - ``af_sfm``: ``(encoder, depth_decoder)`` tuple, as the eval tools
      already consume.
    - ``depth_anything``: a single :class:`networks.DepthAnythingAdapter`
      (frozen). Its ``forward_depth(color)`` returns metric depth directly.
    """
    if depth_backend == "depth_anything":
        from networks import DepthAnythingAdapter

        if not depth_anything_repo:
            raise ValueError(
                "depth_backend='depth_anything' requires --depth_anything_repo"
            )
        adapter = DepthAnythingAdapter(
            hf_repo_id=depth_anything_repo,
            scales=(0, 1, 2, 3),
            min_depth=min_depth,
            max_depth=max_depth,
            frozen=True,
        )
        adapter.eval().to(device)
        return adapter

    # af_sfm (default)
    weights_folder = os.path.expanduser(str(weights_folder))
    for fname in ("encoder.pth", "depth.pth"):
        if not os.path.isfile(os.path.join(weights_folder, fname)):
            raise FileNotFoundError(
                "Missing {} under {}".format(fname, weights_folder)
            )
    encoder = ResnetEncoder(num_layers, False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4))
    encoder.load_state_dict(
        torch.load(os.path.join(weights_folder, "encoder.pth"), map_location="cpu"),
        strict=False,
    )
    depth_decoder.load_state_dict(
        torch.load(os.path.join(weights_folder, "depth.pth"), map_location="cpu")
    )
    encoder.eval().to(device)
    depth_decoder.eval().to(device)
    return (encoder, depth_decoder)


def predict_depth_m(model, color, *, min_depth, max_depth, depth_backend: str):
    """Return metric depth ``(B, 1, H, W)`` on the model's device.

    For ``af_sfm`` the model is a ``(encoder, depth_decoder)`` tuple and we
    reconstruct depth from the sigmoid disparity via ``disp_to_depth`` (the
    canonical AF-SfMLearner formula). For ``depth_anything`` the model is a
    :class:`DepthAnythingAdapter` and we call ``forward_depth`` directly
    (no invert — cleaner metric depth).
    """
    if depth_backend == "depth_anything":
        return model.forward_depth(color)

    encoder, depth_decoder = model
    features = encoder(color)
    outputs = depth_decoder(features)
    disp = outputs[("disp", 0)]
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth


def requires_weights_folder(depth_backend: str) -> bool:
    """Whether the backend needs a local ``--weights_folder``."""
    return depth_backend != "depth_anything"
