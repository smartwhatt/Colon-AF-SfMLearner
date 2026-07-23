# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AF-SfMLearner is a self-supervised monocular depth and ego-motion estimation system for endoscopy/colonoscopy video. It implements appearance flow — a brightness/color calibration mechanism — alongside optical flow and standard photometric losses to handle the challenging lighting in endoscopic scenes.

## Active Project Scope

This repo is currently being used for colonoscopy polyp-size research, not generic KITTI/SCARED benchmarking. The immediate research goal is automated metric polyp size estimation from standard monocular colonoscopy video, with no hardware modification and with honest uncertainty/abstention when metric scale is not anchored.

Current research state:
- **Phase 1 C3VDv2 depth baseline**: Run D (`stage2_specular_pps/weights_best`) is the preferred phantom-depth baseline: SpecularNet + PPS is best on aggregate C3VDv2 HF test metrics.
- **Active blocker**: REALColon adaptation. Target-only REALColon fine-tuning damaged relative depth or absolute scale. Frozen `pose_transform` preserves the depth prior and is a no-forgetting control, not clinical-domain depth adaptation.
- **Immediate next work**: Select and validate source-replay checkpoints with a fixed REALColon qualitative panel and a target-domain scale-aware gate. The 3k final `weights_5` is clinically interesting but is not selected; do not use replay loss alone for clinical selection or launch another long replay campaign by default.
- **Phase 2 metric scale**: Patient video can estimate at most normalized near-light geometry (`beta = b/lambda`) unless an explicit metric anchor is supplied. Do not emit or claim metric polyp size without `ScaleAnchorResolver` or equivalent anchor-gating.
- **Paper claims**: C3VDv2 depth ablations and the near-light identifiability result are claim-ready. Clinical 10 mm threshold accuracy, Polyp-Size validation, and real clinical metric sizing are blocked.

Key papers:
- "Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue" (Medical Image Analysis 2022)
- "Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes" (ICRA 2021)

## Wiki Workflow

The project knowledge base is the topic wiki:

```text
/Users/smartwatt/wiki/topics/colonscopy-polyp-size-research
```

Read the wiki before making research-direction changes. Primary entry points:
- `inventory/views/active-actions.md` — blocker-first task list and next long-running job.
- `wiki/experiments/_index.md` — current empirical state and decisions.
- `wiki/experiments/c3vdv2-hf-depth-benchmark.md` — canonical C3VDv2 leaderboard.
- `output/projects/polyp-size-paper/claims-matrix.md` — what can and cannot be claimed in writing.
- `output/projects/polyp-size-paper/results-register.md` — paper-facing results inventory.

When adding experiment results:
1. Keep raw evidence immutable: add a raw note under `raw/notes/`.
2. Add or update a concise experiment card under `wiki/experiments/`.
3. Update `wiki/experiments/_index.md`; update `c3vdv2-hf-depth-benchmark.md` only for comparable C3VDv2 HF protocol runs.
4. If the result changes paper claims, update `output/projects/polyp-size-paper/claims-matrix.md` and `results-register.md`.
5. Append a short entry to the topic `log.md`.

Do not treat repo reports, terminal output, or chat summaries as sufficient provenance once a result matters. Promote them into the wiki.

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

**REALColon active selection workflow**:

The `teacher_pose_transform` Run B command is a completed frozen-depth control,
not the next experiment. First compare the source-replay candidates with a fixed
REALColon panel, then define a target-domain scale-aware selector before any
longer campaign. Use `teacher_source_replay` only as a guarded trainable-depth
branch; it does not establish clinical metric depth by itself.

Gate every depth-adapting REALColon checkpoint with:
```bash
# absolute-scale / no-median check
uv run python tools/evaluate_c3vdv2.py \
  --data_source hf --hf_repo_id SmartWhatt/c3vdv2-SfM --hf_split test \
  --weights_folder <weights> --load_specular_weights

# relative-depth / median-scaled check
uv run python tools/evaluate_c3vdv2.py \
  --data_source hf --hf_repo_id SmartWhatt/c3vdv2-SfM --hf_split test \
  --weights_folder <weights> --median_scaling --load_specular_weights

# pose guardrail
uv run python tools/evaluate_c3vdv2_pose.py \
  --data_source hf --hf_repo_id SmartWhatt/c3vdv2-SfM --hf_split test \
  --weights_folder <weights>
```

Pass gate for REALColon recovery remains conservative: median-scaled C3VDv2 AbsRel <= 0.18, delta < 1.25 >= 0.75, no severe no-median scale drift, no material pose regression, and no qualitative REALColon collapse.

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
- `StageTwoTrainer`: `load_model()` with param freezing; `set_train/eval` for stage-2 model groups; optional frozen-teacher relative-depth consistency for REALColon recovery
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

All local datasets inherit from `colon_datasets/mono_dataset.py` (`MonoDataset`), which handles multi-scale resizing, color augmentation, and frame loading.

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
- `--trainable_modules ...` — optional Stage 2 allowlist; used by `pose_transform`, `teacher_pose_transform`, and `teacher_source_replay`
- `--teacher_weights_folder`, `--teacher_consistency_weight`, `--teacher_consistency_mode relative_log_median` — opt-in REALColon recovery teacher path; defaults are off and preserve prior behavior

Loss weights (all tunable): `--disparity_smoothness 1e-4`, `--position_smoothness 1e-3`, `--consistency_constraint 0.01`, `--epipolar_constraint 0.01`, `--geometry_constraint 0.01`, `--transform_constraint 0.01`, `--transform_smoothness 0.01`.

## Validation Before Committing

Preferred quick checks after code changes:

```bash
bash -n scripts/run_finetune_real_colon.sh scripts/submit_real_colon_hf.sh scripts/submit_real_colon_hf_lanta.sh scripts/submit_realcolon_panel_lanta.sh
.venv/bin/python -m py_compile options.py trainers/stage_two.py trainers/base.py scripts/utils/download_real_colon_to_hf.py
uv run pytest tests
git diff --check
```

Full `uv run pytest` currently collects top-level `test_simple.py`, which imports `matplotlib`; if that dependency is absent, use `uv run pytest tests` for the maintained suite and report the collection limitation.

## Supervised Fine-tuning Losses (trainers/c3vdv2_losses.py)

For C3VDv2 fine-tuning, supervised losses are added on top of the self-supervised photometric loss:
- `scale_invariant_log_loss()` — depth supervision (Eigen et al. 2014)
- `smooth_l1_depth_loss()` — alternative robust depth loss
- `pose_loss()` — pose supervision (axis-angle + translation regression)

Weights for supervised terms are ramped up progressively during training (see `scripts/run_finetune_c3vdv2.sh`).
