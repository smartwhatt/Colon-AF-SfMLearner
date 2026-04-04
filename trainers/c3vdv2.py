"""
trainers/c3vdv2.py
------------------
C3VDv2 fine-tuning variants of Stage 1 and Stage 2 trainers.

Both trainers:
  - Replace the SCARED dataloaders with C3VDv2Dataset
  - Fix Adam β2 to paper value (0.99)
  - Adapt C3VDv2Dataset key format to the tuple keys expected by trainers

C3VDv2Stage2Trainer additionally:
  - Adds supervised depth loss (scale-invariant log)
  - Adds supervised pose loss (axis-angle + translation regression)
  - Ramps supervised loss weights from high → low over training epochs
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.c3vdv2_dataset import C3VDv2Dataset
from .c3vdv2_losses import scale_invariant_log_loss, pose_loss
from .base import BaseTrainer
from .stage_one import StageOneTrainer
from .stage_two import StageTwoTrainer

# Paper constants (Section 4.2)
PAPER_BETA1 = 0.9
PAPER_BETA2 = 0.99

C3VD_MIN_DEPTH = 0.001   # 1 mm
C3VD_MAX_DEPTH = 0.100   # 100 mm


# ── Shared mixin ─────────────────────────────────────────────────────────────

class _C3VDv2DataMixin(BaseTrainer):
    """
    Abstract mixin that replaces build_dataloaders() and provides _adapt_keys().

    Inherits from BaseTrainer so that:
      - type checkers resolve self.opt, super().process_batch(), etc.
      - Python's cooperative MRO chains correctly to the concrete trainer
        (e.g. C3VDv2Stage1Trainer → _C3VDv2DataMixin → StageOneTrainer → BaseTrainer)

    Never instantiate directly — always use via multiple inheritance with a
    concrete trainer: class C3VDv2Stage1Trainer(_C3VDv2DataMixin, StageOneTrainer).
    """

    # Satisfy ABC — these delegate to the concrete trainer via cooperative MRO.
    def build_models(self) -> None: super().build_models()  # type: ignore[abstract]
    def set_train(self) -> None: super().set_train()        # type: ignore[abstract]
    def set_eval(self) -> None: super().set_eval()          # type: ignore[abstract]
    def compute_losses(self, inputs: dict, outputs: dict) -> dict:
        return super().compute_losses(inputs, outputs)      # type: ignore[abstract]

    def build_dataloaders(self):
        undistort_maps = getattr(self.opt, "undistort_maps", "undistort_maps/")

        train_dataset = C3VDv2Dataset(
            data_root=self.opt.data_path,
            split_file=os.path.join("splits", self.opt.split, "train.txt"),
            undistort_maps=undistort_maps,
            target_size=(self.opt.height, self.opt.width),
            augment=True,
        )
        val_dataset = C3VDv2Dataset(
            data_root=self.opt.data_path,
            split_file=os.path.join("splits", self.opt.split, "val.txt"),
            undistort_maps=undistort_maps,
            target_size=(self.opt.height, self.opt.width),
            augment=False,
        )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            shuffle=True, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size,
            shuffle=False, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True,
        )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = (
            len(train_dataset) // self.opt.batch_size * self.opt.num_epochs
        )
        print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    def process_batch(self, inputs: dict) -> tuple:  # type: ignore[override]
        # super() resolves to the concrete trainer (StageOneTrainer / StageTwoTrainer)
        # via cooperative MRO — not to BaseTrainer's abstract method.
        return super().process_batch(self._adapt_keys(inputs))  # type: ignore[abstract]

    def process_batch_val(self, inputs: dict) -> tuple:
        return super().process_batch_val(self._adapt_keys(inputs))  # type: ignore[abstract]

    def _adapt_keys(self, batch):
        """
        Maps C3VDv2Dataset keys → (key, frame_id, scale) tuple keys.

        C3VDv2 key    →  Trainer tuple key
        ─────────────────────────────────────────────────────────
        color_curr    →  ("color",  0, 0)  ("color_aug",  0, 0)
        color_prev    →  ("color", -1, 0)  ("color_aug", -1, 0)
        color_next    →  ("color",  1, 0)  ("color_aug",  1, 0)
        K (B,3,3)     →  ("K", s) and ("inv_K", s) for s in scales
        everything else → pass-through (gt_depth, occlusion, poses…)
        """
        device = next(
            (v.device for v in batch.values() if hasattr(v, "device")), None
        )
        out = {}

        for f_id, key in {0: "color_curr", -1: "color_prev", 1: "color_next"}.items():
            t = batch[key].to(device)
            out[("color", f_id, 0)] = t
            out[("color_aug", f_id, 0)] = t  # augmentation applied in dataset

        # Intrinsics: normalised (B,3,3) → pixel-unit (B,4,4) per scale.
        # K_norm[0,0]=fx/512, K_norm[0,2]=cx/512, etc.
        # BackprojectDepth / Project3D expect K in pixel units at each scale,
        # so multiply the normalised row-0 by W//2^s and row-1 by H//2^s.
        if "K" in batch:
            K_norm = batch["K"].to(device)          # (B,3,3) normalised
            B = K_norm.shape[0]
            for s in self.opt.scales:
                h_s = self.opt.height // (2 ** s)
                w_s = self.opt.width  // (2 ** s)
                K_44 = torch.zeros(B, 4, 4, device=device)
                K_44[:, :3, :3] = K_norm
                K_44[:, 0, :] *= w_s
                K_44[:, 1, :] *= h_s
                K_44[:, 3, 3] = 1.0
                out[("K", s)] = K_44
                out[("inv_K", s)] = torch.linalg.inv(K_44)

        # Multi-scale color for smoothness losses
        for scale in self.opt.scales[1:]:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            for f_id in self.opt.frame_ids:
                if ("color", f_id, 0) in out:
                    out[("color", f_id, scale)] = F.interpolate(
                        out[("color", f_id, 0)], [h, w],
                        mode="bilinear", align_corners=False,
                    )

        # Pass-through remaining keys (gt_depth, occlusion, pose vectors…)
        skip = {"color_curr", "color_prev", "color_next", "K", "seq", "frame_idx"}
        for key, val in batch.items():
            if key not in skip:
                out[key] = val.to(device) if hasattr(val, "to") else val

        return out


# ── Stage 1 fine-tune ────────────────────────────────────────────────────────

class C3VDv2Stage1Trainer(_C3VDv2DataMixin, StageOneTrainer):
    """Fine-tune the registration net (Stage 1) on C3VDv2 colonoscopy data.
    Self-supervised only — no GT depth or pose used."""

    def __init__(self, options):
        super().__init__(options)

        # Fix Adam β2 to paper value (0.99 vs PyTorch default 0.999)
        for pg in self.model_optimizer.param_groups:
            pg["betas"] = (PAPER_BETA1, PAPER_BETA2)

        print(f"\n{'='*60}")
        print("  Stage 1 Fine-Tuning  |  C3VDv2 Colonoscopy")
        print(f"{'='*60}")
        print(f"  Training:  position_encoder + position  (only)")
        print(f"  Loss:      photometric + smoothness  (no GT used)")
        print(f"  Resolution: {self.opt.width}x{self.opt.height}  Batch: {self.opt.batch_size}")
        print(f"  LR: {self.opt.learning_rate}  Epochs: {self.opt.num_epochs}")
        print(f"{'='*60}\n")


# ── Stage 2 fine-tune ────────────────────────────────────────────────────────

class C3VDv2Stage2Trainer(_C3VDv2DataMixin, StageTwoTrainer):
    """
    Fine-tune Stage 2 on C3VDv2 with supervised depth + pose losses.

    Loss weights ramp from high GT supervision → lower as the self-supervised
    signal becomes reliable over training epochs.
    """

    def __init__(self, options):
        self.w_depth_start = getattr(options, "w_depth_start", 1.0)
        self.w_depth_end = getattr(options, "w_depth_end", 0.3)
        self.w_pose_start = getattr(options, "w_pose_start", 0.5)
        self.w_pose_end = getattr(options, "w_pose_end", 0.1)
        self.ramp_epochs = getattr(options, "ramp_epochs", 8)
        self.w_depth = self.w_depth_start
        self.w_pose = self.w_pose_start

        # Force correct depth range for C3VDv2 (1–100 mm).
        # The default options use outdoor values (0.1–100 m) which would make
        # disp_to_depth produce depths 1000× too large for this dataset.
        options.min_depth = C3VD_MIN_DEPTH
        options.max_depth = C3VD_MAX_DEPTH

        super().__init__(options)

        for pg in self.model_optimizer.param_groups:
            pg["betas"] = (PAPER_BETA1, PAPER_BETA2)

        print(f"\n{'='*60}")
        print("  Stage 2 Fine-Tuning  |  C3VDv2 Colonoscopy")
        print(f"{'='*60}")
        print(f"  Training:  encoder+depth  transform  pose_encoder+pose")
        print(f"  Frozen:    position_encoder + position  (Stage 1 weights)")
        print(f"  Resolution: {self.opt.width}x{self.opt.height}  Batch: {self.opt.batch_size}")
        print(f"  LR: {self.opt.learning_rate}  Epochs: {self.opt.num_epochs}")
        print(
            f"  w_depth: {self.w_depth_start} → {self.w_depth_end}  "
            f"w_pose: {self.w_pose_start} → {self.w_pose_end}  "
            f"over {self.ramp_epochs} epochs"
        )
        print(f"{'='*60}\n")

    def _update_loss_weights(self):
        if self.epoch < self.ramp_epochs:
            self.w_depth = self.w_depth_start
            self.w_pose = self.w_pose_start
        else:
            total_ramp = max(self.opt.num_epochs - self.ramp_epochs, 1)
            progress = min((self.epoch - self.ramp_epochs) / total_ramp, 1.0)
            self.w_depth = self.w_depth_start + progress * (
                self.w_depth_end - self.w_depth_start
            )
            self.w_pose = self.w_pose_start + progress * (
                self.w_pose_end - self.w_pose_start
            )
        if hasattr(self, "writers"):
            self.writers["train"].add_scalar("schedule/w_depth", self.w_depth, self.epoch)
            self.writers["train"].add_scalar("schedule/w_pose", self.w_pose, self.epoch)

    def run_epoch(self):
        self._update_loss_weights()
        print(f"  Epoch {self.epoch}  |  w_depth={self.w_depth:.3f}  w_pose={self.w_pose:.3f}")
        super().run_epoch()

    def compute_losses(self, inputs, outputs):
        losses = super().compute_losses(inputs, outputs)

        # Supervised depth
        if "gt_depth" in inputs:
            gt_depth = inputs["gt_depth"]
            pred_depth = outputs[("depth", 0, 0)]
            depth_mask = (
                (gt_depth > C3VD_MIN_DEPTH) & (gt_depth < C3VD_MAX_DEPTH - 0.001)
            ).float()
            if "occlusion" in inputs:
                depth_mask = depth_mask * (1.0 - inputs["occlusion"])
            l_depth = scale_invariant_log_loss(pred_depth, gt_depth, depth_mask)
            losses["loss/depth_supervised"] = l_depth
            losses["loss"] = losses["loss"] + self.w_depth * l_depth

        # Supervised pose
        gt_key_map = {
            -1: ("axang_curr2prev", "trans_curr2prev"),
             1: ("axang_curr2next", "trans_curr2next"),
        }
        if "axang_curr2prev" in inputs:
            total_pose = 0.0
            n = 0
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s" or f_i not in gt_key_map:
                    continue
                axk, trk = gt_key_map[f_i]
                if axk not in inputs:
                    continue
                pred_ax = outputs[("axisangle", 0, f_i)].squeeze(1)
                pred_tr = outputs[("translation", 0, f_i)].squeeze(1)
                total_pose += pose_loss(pred_ax, pred_tr, inputs[axk], inputs[trk])
                n += 1
            if n > 0:
                l_pose = total_pose / n
                losses["loss/pose_supervised"] = l_pose
                losses["loss"] = losses["loss"] + self.w_pose * l_pose

        return losses
