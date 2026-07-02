"""Vendored Depth Anything V2 official source (metric-depth path).

Source: https://github.com/DepthAnything/Depth-Anything-V2
License: Apache License 2.0 (see networks/depth_anything_v2/LICENSE).

Vendored (rather than pip-installed) because the PyPI `depth-anything-v2`
package pulls unwanted deps (gradio-imageslider, matplotlib) and pins
torch/torchvision incompatibly with this repo. These source files have
zero extra deps beyond torch/torchvision/opencv/numpy, all already present.

Exposes :class:`DepthAnythingV2` for loading the official metric .pth
checkpoints (e.g. depth_anything_v2_metric_hypersim_vitl.pth) which output
metric depth in metres — the path for the "metric DA-v2" backend variant.
The transformers `-hf` variants (relative depth) are loaded separately via
the DepthAnythingAdapter's transformers code path.
"""

from .dpt import DepthAnythingV2

__all__ = ["DepthAnythingV2"]
