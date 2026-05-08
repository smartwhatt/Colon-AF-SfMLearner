"""
options.py
----------
Training configuration defined as a Pydantic model.

All fields are type-annotated; Pydantic validates values on construction.
An argparse parser is auto-generated from the model so the CLI is always
in sync with the type definitions — no separate argparse boilerplate needed.

Usage:
    # Parse from sys.argv (typical training script)
    cfg = TrainingConfig.from_args()

    # Construct programmatically (tests / notebooks)
    cfg = TrainingConfig(data_path="/data/endovis", batch_size=8)

    # Backward-compatible wrapper (used by old train_*.py scripts)
    opts = MonodepthOptions()
    cfg  = opts.parse()       # returns a TrainingConfig instance
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, Field, field_validator

_file_dir = os.path.dirname(os.path.abspath(__file__))

# ── Type aliases ──────────────────────────────────────────────────────────────

DatasetType     = Literal["endovis", "c3vdv2", "real_colon",
                           "kitti", "kitti_odom", "kitti_depth", "kitti_test"]
SplitType       = Literal["endovis", "c3vdv2", "real_colon",
                           "eigen_zhou", "eigen_full", "odom", "benchmark"]
NumLayersType   = Literal[18, 34, 50, 101, 152]
WeightsInitType = Literal["pretrained", "scratch"]
PoseInputType   = Literal["pairs", "all"]
PoseModelType   = Literal["posecnn", "separate_resnet", "shared"]
EvalSplitType   = Literal["eigen", "eigen_benchmark", "benchmark",
                           "odom_9", "odom_10", "endovis"]
TrainModeType   = Literal["stage1", "stage2", "end_to_end",
                           "c3vdv2_stage1", "c3vdv2_stage2"]
DataSourceType  = Literal["local", "hf"]


# ── Config model ──────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """All training hyperparameters and paths in one type-safe object."""

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode: TrainModeType = Field(
        default="stage2",
        description="training mode (stage1 | stage2 | end_to_end | c3vdv2_stage1 | c3vdv2_stage2)",
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_path: str = Field(
        default=os.path.join(_file_dir, "endovis_data"),
        description="path to the training data",
    )
    log_dir: str = Field(
        default=os.path.join(os.path.expanduser("~"), "tmp"),
        description="log directory",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    model_name: str = Field(default="mdp", description="folder name for the saved model")
    split: SplitType = Field(default="endovis", description="training split to use")
    num_layers: NumLayersType = Field(default=18, description="number of ResNet layers")
    dataset: DatasetType = Field(default="endovis", description="dataset to train on")
    png: bool = Field(default=False, description="train from raw KITTI png files")
    height: int = Field(default=256, description="input image height (must be multiple of 32)")
    width: int = Field(default=320, description="input image width (must be multiple of 32)")
    disparity_smoothness: float = Field(default=1e-4, description="disparity smoothness weight")
    position_smoothness: float = Field(default=1e-3, description="registration smoothness weight")
    consistency_constraint: float = Field(default=0.01, description="consistency constraint weight")
    epipolar_constraint: float = Field(default=0.01, description="epipolar constraint weight")
    geometry_constraint: float = Field(default=0.01, description="geometry constraint weight")
    transform_constraint: float = Field(default=0.01, description="transform constraint weight")
    transform_smoothness: float = Field(default=0.01, description="transform smoothness weight")
    scales: List[int] = Field(default=[0, 1, 2, 3], description="loss scales")
    min_depth: float = Field(default=0.1, description="minimum depth")
    max_depth: float = Field(default=150.0, description="maximum depth")
    use_stereo: bool = Field(default=False, description="use stereo pair for training")
    frame_ids: List[int] = Field(default=[0, -1, 1], description="frame IDs to load")

    # ── Data source ───────────────────────────────────────────────────────────
    data_source: DataSourceType = Field(
        default="local",
        description="data backend to use (local | hf)",
    )
    hf_repo_id: Optional[str] = Field(
        default=None,
        description="Hugging Face dataset repo id when data_source=hf",
    )
    hf_streaming: bool = Field(
        default=False,
        description="stream Hugging Face dataset instead of using the local HF cache",
    )
    hf_cache_dir: Optional[str] = Field(
        default=None,
        description="optional Hugging Face dataset cache directory",
    )
    hf_shuffle_buffer: int = Field(
        default=10000,
        description="shuffle buffer size for HF streaming datasets",
    )
    steps_per_epoch: Optional[int] = Field(
        default=None,
        description="optional batch cap per epoch, useful for streaming datasets",
    )

    # ── Optimisation ──────────────────────────────────────────────────────────
    batch_size: int = Field(default=12, description="batch size")
    learning_rate: float = Field(default=1e-4, description="learning rate")
    num_epochs: int = Field(default=20, description="number of epochs")
    scheduler_step_size: int = Field(default=10, description="LR scheduler step size")

    # ── Ablations ─────────────────────────────────────────────────────────────
    v1_multiscale: bool = Field(default=False, description="use monodepth v1 multiscale")
    avg_reprojection: bool = Field(default=False, description="use average reprojection loss")
    disable_automasking: bool = Field(default=False, description="disable auto-masking")
    predictive_mask: bool = Field(default=False, description="use Zhou et al predictive masking")
    no_ssim: bool = Field(default=False, description="disable SSIM in the loss")
    weights_init: WeightsInitType = Field(default="pretrained", description="weight initialisation")
    pose_model_input: PoseInputType = Field(default="pairs", description="pose network input frames")
    pose_model_type: PoseModelType = Field(default="separate_resnet", description="pose model architecture")

    # ── System ────────────────────────────────────────────────────────────────
    no_cuda: bool = Field(default=False, description="disable CUDA")
    num_workers: int = Field(default=12, description="dataloader workers")

    # ── Loading ───────────────────────────────────────────────────────────────
    load_weights_folder: Optional[str] = Field(default=None, description="folder of weights to load")
    models_to_load: List[str] = Field(
        default=["position_encoder", "position"],
        description="model names to load from weights folder",
    )
    load_trainable_weights_folder: Optional[str] = Field(
        default=None,
        description="optional folder of trainable Stage 2 weights to warm-start without freezing",
    )
    trainable_models_to_load: List[str] = Field(
        default=[
            "encoder",
            "depth",
            "transform_encoder",
            "transform",
            "pose_encoder",
            "pose",
            "specular",
        ],
        description="trainable model names to load from load_trainable_weights_folder",
    )

    # ── C3VDv2 / fine-tuning ──────────────────────────────────────────────────
    undistort_maps: str = Field(
        default="undistort_maps/",
        description="directory containing undistort_map1.npy / undistort_map2.npy (C3VDv2)",
    )
    w_depth_start: float = Field(default=1.0, description="supervised depth weight at epoch 0")
    w_depth_end: float = Field(default=0.3, description="supervised depth weight at final epoch")
    w_pose_start: float = Field(default=0.5, description="supervised pose weight at epoch 0")
    w_pose_end: float = Field(default=0.1, description="supervised pose weight at final epoch")
    ramp_epochs: int = Field(default=8, description="epochs before supervised weights start ramping")

    # ── SpecularNet ───────────────────────────────────────────────────────────
    use_specular_net: bool = Field(
        default=False,
        description="add a lightweight specular confidence head to Stage 2 training",
    )
    specular_loss_weight: float = Field(
        default=0.05,
        description="weight for the specular BCE weak-supervision loss",
    )
    specular_threshold: float = Field(
        default=0.98,
        description="intensity threshold (0–1) above which pixels are treated as specular",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_frequency: int = Field(default=200, description="batches between TensorBoard log writes")
    save_frequency: int = Field(default=1, description="epochs between model saves")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_stereo: bool = Field(default=False, description="evaluate in stereo mode")
    eval_mono: bool = Field(default=False, description="evaluate in mono mode")
    disable_median_scaling: bool = Field(default=False, description="disable median scaling in eval")
    pred_depth_scale_factor: float = Field(default=1.0, description="multiply predictions by this")
    ext_disp_to_eval: Optional[str] = Field(default=None, description=".npy disparities file to eval")
    eval_split: EvalSplitType = Field(default="endovis", description="split for evaluation")
    save_pred_disps: bool = Field(default=False, description="save predicted disparities")
    no_eval: bool = Field(default=False, description="disable evaluation")
    eval_eigen_to_benchmark: bool = Field(
        default=False,
        description="evaluate eigen predictions against the new benchmark",
    )
    eval_out_dir: Optional[str] = Field(default=None, description="output folder for disparities")
    post_process: bool = Field(
        default=False,
        description="apply flipping post-processing from the original MonoDepth paper",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("height", "width")
    @classmethod
    def must_be_multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(f"height and width must be multiples of 32, got {v}")
        return v

    @field_validator("frame_ids")
    @classmethod
    def frame_ids_must_start_with_zero(cls, v: List[int]) -> List[int]:
        if v[0] != 0:
            raise ValueError("frame_ids must start with 0")
        return v

    # ── Parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args: Optional[List[str]] = None) -> "TrainingConfig":
        """Parse from sys.argv (or a provided list) and validate."""
        parser = _build_argparse(cls)
        namespace = parser.parse_args(args)
        return cls(**vars(namespace))

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Expose the parser so callers can add extra arguments if needed."""
        return _build_argparse(cls)

    model_config = {"arbitrary_types_allowed": True}


# ── Argparse auto-generation ─────────────────────────────────────────────────

def _build_argparse(model_cls: type[TrainingConfig]) -> argparse.ArgumentParser:
    """
    Generate an ArgumentParser from a pydantic BaseModel.

    Rules:
      bool field         → --flag  (action="store_true")
      List[T] field      → --flag  (nargs="+", type=T)
      Literal[...] field → --flag  (choices=[...], type=type(first_choice))
      Optional[T] field  → --flag  (type=T, default=None)
      everything else    → --flag  (type=annotation)
    """
    parser = argparse.ArgumentParser(description="AF-SfMLearner training options")

    for name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        default    = field_info.default
        help_text  = field_info.description or ""
        flag       = f"--{name}"

        _add_argument(parser, flag, annotation, default, help_text)

    return parser


def _add_argument(
    parser: argparse.ArgumentParser,
    flag: str,
    annotation: type,
    default: object,
    help_text: str,
) -> None:
    origin = get_origin(annotation)
    type_args = get_args(annotation)

    # ── bool → store_true flag ────────────────────────────────────────────────
    if annotation is bool:
        parser.add_argument(flag, action="store_true", default=default, help=help_text)
        return

    # ── Optional[X] = Union[X, None] → unwrap inner type ─────────────────────
    if origin is Union:
        non_none = [a for a in type_args if a is not type(None)]
        inner = non_none[0] if non_none else str
        parser.add_argument(flag, type=inner, default=default, help=help_text)
        return

    # ── Literal[...] → choices ────────────────────────────────────────────────
    if origin is Literal:
        choices = list(type_args)
        elem_type = type(choices[0])
        parser.add_argument(
            flag, type=elem_type, choices=choices, default=default, help=help_text
        )
        return

    # ── List[T] → nargs="+" ───────────────────────────────────────────────────
    if origin is list:
        elem_type = type_args[0] if type_args else str
        parser.add_argument(flag, nargs="+", type=elem_type, default=default, help=help_text)
        return

    # ── Scalar ────────────────────────────────────────────────────────────────
    parser.add_argument(flag, type=annotation, default=default, help=help_text)


# ── Backward-compatible wrapper ───────────────────────────────────────────────

class MonodepthOptions:
    """
    Backward-compatible wrapper. Existing scripts that do:

        opts = MonodepthOptions()
        cfg  = opts.parse()

    still work. The returned object is now a TrainingConfig instance
    (a pydantic model) instead of a raw argparse Namespace.
    """

    def __init__(self) -> None:
        self.parser = TrainingConfig.build_parser()

    def parse(self) -> TrainingConfig:
        namespace = self.parser.parse_args()
        return TrainingConfig(**vars(namespace))
