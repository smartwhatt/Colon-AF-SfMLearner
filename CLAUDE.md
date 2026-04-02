# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AF-SfMLearner is a self-supervised monocular depth and ego-motion estimation system for endoscopy/colonoscopy video. It implements appearance flow — a brightness/color calibration mechanism — alongside optical flow and standard photometric losses to handle the challenging lighting in endoscopic scenes.

Key papers:
- "Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue" (Medical Image Analysis 2022)
- "Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes" (ICRA 2021)

## Setup

Uses `uv` as the package manager (not pip):

```bash
uv sync
```

Docker for GPU training:
```bash
docker build -t af-sfmlearner:c3vdv2 .
docker run --gpus all -it af-sfmlearner:c3vdv2 bash
```

Dependencies (pyproject.toml): torch, torchvision, opencv-python, scikit-image, tensorboardX.

## Training Commands

All training modes go through the single `train.py` entry point with `--mode`:

```bash
# Stage 1 – train registration net only
CUDA_VISIBLE_DEVICES=0 python train.py --mode stage1 \
  --data_path <endovis_data> --log_dir logs/

# Stage 2 – train depth + pose + transform (loads Stage 1 weights)
CUDA_VISIBLE_DEVICES=0 python train.py --mode stage2 \
  --data_path <endovis_data> --log_dir logs/ \
  --load_weights_folder logs/stage1/models/weights_19/

# End-to-end – train all networks simultaneously
CUDA_VISIBLE_DEVICES=0 python train.py --mode end_to_end \
  --data_path <endovis_data> --log_dir logs/

# C3VDv2 fine-tune Stage 1 (registration net)
CUDA_VISIBLE_DEVICES=0 python train.py --mode c3vdv2_stage1 \
  --dataset c3vdv2 --split c3vdv2 \
  --data_path /path/to/C3VDv2 --log_dir logs/ --undistort_maps undistort_maps/

# C3VDv2 fine-tune Stage 2 (with supervised depth + pose)
CUDA_VISIBLE_DEVICES=0 python train.py --mode c3vdv2_stage2 \
  --dataset c3vdv2 --split c3vdv2 \
  --data_path /path/to/C3VDv2 --log_dir logs/ --undistort_maps undistort_maps/ \
  --load_weights_folder logs/c3vdv2_stage1/models/weights_19/ \
  --w_depth_start 1.0 --w_depth_end 0.3 --w_pose_start 0.5 --w_pose_end 0.1
```

The old per-mode entry scripts (`train_stage_one.py`, `train_stage_two.py`, `train_end_to_end.py`) still work as backward-compat wrappers (they import directly from `trainers/`).

**C3VDv2 automated pipeline** (undistortion + both stages):
```bash
DATA_PATH=/path/to/C3VDv2 LOG_DIR=logs/c3vdv2 bash scripts/run_finetune_c3vdv2.sh
```

## Evaluation & Inference

```bash
# Single image / folder inference
python test_simple.py --model_path <weights_dir> --image_path <image_or_folder>

# SCARED/Endovis depth evaluation (requires exporting GT first)
python tools/export_gt_depth.py --data_path endovis_data --split endovis
python tools/evaluate_depth.py --data_path <endovis_data> --load_weights_folder <weights> --eval_mono

# C3VDv2 evaluation
uv run tools/evaluate_c3vdv2.py --data_path /path/to/C3VDv2 --load_weights_folder <weights> --split c3vdv2
```

## Trainer Architecture

All trainers live in `trainers/` and inherit from `BaseTrainer` (`trainers/base.py`):

```
BaseTrainer                  # shared infra: train loop, val, save/load, geometry layers
├── StageOneTrainer          # position net only; overrides process_batch + compute_losses
├── StageTwoTrainer          # full pipeline, position frozen; overrides load_model (freeze)
│   └── C3VDv2Stage2Trainer  # + C3VDv2 dataloaders + supervised depth/pose losses + weight ramp
└── EndToEndTrainer          # full pipeline, dual optimizer, alternating batch steps
    (C3VDv2Stage1Trainer uses _C3VDv2DataMixin + StageOneTrainer)
```

`TRAINERS` dict in `trainers/__init__.py` maps `--mode` → trainer class.

**What BaseTrainer provides (not repeated in subclasses):**
- `predict_poses()` — full position→registration→occlusion→transform→pose forward (used by Stage 2 and End-to-End)
- `generate_images_pred()` — depth→3D→reproject warped frames
- `compute_reprojection_loss()` — SSIM + L1
- `compute_losses_val()` — NCC-based val metric
- `val()`, `train()`, `log_time()`, `save_model()`, `load_model()`, `save_opts()`
- `_setup_geometry()` — SSIM, SpatialTransformer, BackprojectDepth, Project3D, occlusion masks
- `build_dataloaders()` — default SCARED splits/{split}/*.txt layout
- `build_optimizers()` — default single Adam

**What each subclass adds:**
- `StageOneTrainer`: `_predict_positions()` (flow only, no transform/pose)
- `StageTwoTrainer`: `load_model()` with param freezing; `set_train/eval` for stage-2 model groups
- `EndToEndTrainer`: second optimizer, `set_train_0()`, `run_epoch()` with alternating steps, `_predict_position_transform()`
- `C3VDv2*Trainer`: `build_dataloaders()` using `C3VDv2Dataset`, `_adapt_keys()` (maps dataset keys to trainer tuple keys), supervised losses + weight ramping

## Model Architecture

Five cooperating networks, all based on ResNet encoder + UNet-style decoder:

| Network | Input | Output | Trained in |
|---------|-------|--------|------------|
| `ResnetEncoder` + `DepthDecoder` | current frame | multi-scale disparity (4 scales) | Stage 2 / end-to-end |
| `ResnetEncoder` + `PoseDecoder` | frame pair | 6-DoF pose (axis-angle + translation) | Stage 2 / end-to-end |
| `ResnetEncoder` + `PositionDecoder` | frame pair | 2D optical flow (4 scales) | Stage 1 + frozen in Stage 2 |
| `ResnetEncoder` + `TransformDecoder` | frame pair | RGB brightness correction (4 scales) | Stage 2 / end-to-end |

**Inference pipeline** (per training step):
1. Encode current + adjacent frames
2. `DepthDecoder` → disparity → `disp_to_depth()` → metric depth
3. `PoseDecoder` → (R, t) → 4×4 SE(3) via `transformation_from_parameters()`
4. `PositionDecoder` → optical flow → `SpatialTransformer` warps previous frame
5. `TransformDecoder` → brightness correction → applied to warped frame
6. Photometric loss (SSIM + L1) between corrected-warped frame and target

Key geometry ops live in `layers.py`: `BackprojectDepth`, `Project3D`, `SpatialTransformer`, `disp_to_depth`, `SSIM`.

## Datasets

All datasets inherit from `datasets/mono_dataset.py` (`MonoDataset`), which handles multi-scale resizing, color augmentation, and frame loading.

| Class | File | Camera | Notes |
|-------|------|--------|-------|
| `SCAREDRAWDataset` | scared_dataset.py | Fisheye (Olympus CF-HQ190L) | No undistortion needed |
| `C3VDv2Dataset` | c3vdv2_dataset.py | Fisheye → perspective 512×512 | Requires pre-computed undistortion maps; has GT depth/pose/flow |
| `REALColonDataset` | real_colon_dataset.py | Perspective (Olympus/Fujifilm) | Self-supervised only, no GT |

SCARED data layout:
```
endovis_data/datasetN/keyframeN/image_02/data/0000000001.png
```

Undistortion maps for C3VDv2 are computed with `scripts/compute_undistort_map.py` and stored in `undistort_map/`.

## Key Configuration (options.py)

Notable non-obvious defaults:
- `--frame_ids [0, -1, 1]` — current, previous, next frame triplet
- `--height 256 --width 320` — training resolution
- `--num_layers 18` — ResNet-18 encoder
- `--pose_model_type separate_resnet` — separate encoder for pose (vs. shared)
- `--num_scales 4` — multi-scale losses at scales 0–3

Loss weights (all tunable): `--disparity_smoothness 1e-4`, `--position_smoothness 1e-3`, `--consistency_constraint 0.01`, `--epipolar_constraint 0.01`, `--geometry_constraint 0.01`, `--transform_constraint 0.01`, `--transform_smoothness 0.01`.

## Supervised Fine-tuning Losses (trainers/c3vdv2_losses.py)

For C3VDv2 fine-tuning, supervised losses are added on top of the self-supervised photometric loss:
- `scale_invariant_log_loss()` — depth supervision (Eigen et al. 2014)
- `smooth_l1_depth_loss()` — alternative robust depth loss
- `pose_loss()` — pose supervision (axis-angle + translation regression)

Weights for supervised terms are ramped up progressively during training (see `scripts/run_finetune_c3vdv2.sh`).
