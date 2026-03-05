"""
finetune_c3vdv2_stage2.py
--------------------------
Fine-tune the AF-SfMLearner Stage 2 trainer on C3VDv2 colonoscopy data,
with supervised depth and pose losses and a ramping weight schedule.

WHAT STAGE 2 DOES (from trainer_stage_two.py):
  Models trained:  encoder + depth + transform_encoder + transform
                   + pose_encoder + pose
  Models frozen:   position_encoder + position  (loaded from Stage 1)
                   load_model() sets requires_grad=False on everything it loads

  Per-batch forward pass:
    1. encoder(I_t)                    → features → DepthDecoder → disp at 4 scales
    2. position_encoder([I_s, I_t])    → flow field (frozen)
       SpatialTransformer(I_s, flow)   → registration  (I_s warped to I_t viewpoint)
       reverse flow + range-map check  → occu_mask_backward
    3. transform_encoder([registration, I_t])
       → transform (per-pixel brightness correction)
       refined = transform * occu_mask + I_t      ← brightness-calibrated I_t
    4. pose_encoder([I_s, I_t])        → axisangle, translation → cam_T_cam
    5. depth → backproject → cam_T_cam → reproject → sample I_s
       → ("color", f_i, scale)        ← depth-based warped frame

  Parent loss:
    reprojection = Φ(depth_warped_color, refined)  * occu_mask
    transform    = |refined - registration|         * occu_mask
    smooth_bright = get_smooth_bright(transform, I_t, registration, occu_mask)
    smooth_disp  = get_smooth_loss(norm_disp, I_t)

    total = reprojection/2 + tc*transform/2 + ts*smooth_bright/2 + ds*smooth/2^s

THE WEIGHT RAMPING STRATEGY:
  Early training: registration net (just adapted to colonoscopy) still has
  imperfect flow → registration is noisy → refined is a corrupted target →
  reprojection loss gives unreliable depth gradients.

  Solution: ramp w_depth HIGH → LOW and w_reprojection LOW → HIGH over epochs.

  Phase 1 (epochs 0 → ramp_epochs-1):
    w_depth = w_depth_start (e.g. 1.0) — GT depth dominates, anchors metric scale
    w_pose  = w_pose_start  (e.g. 0.5) — GT pose anchors motion estimation
    self-supervised reprojection loss still runs but has less influence

  Phase 2 (epochs ramp_epochs → num_epochs-1):
    w_depth linearly decays toward w_depth_end (e.g. 0.3)
    w_pose  linearly decays toward w_pose_end  (e.g. 0.1)
    self-supervised signal grows more reliable as transform net adapts

  The self-supervised losses (reprojection, transform, smooth) always run
  at their original paper weights — we never disable them.

Usage:
    CUDA_VISIBLE_DEVICES=0 python finetune_c3vdv2_stage2.py \\
        --data_path           /path/to/C3VDv2 \\
        --split               c3vdv2 \\
        --undistort_maps      undistort_maps/ \\
        --log_dir             logs/c3vdv2 \\
        --model_name          stage2 \\
        --load_weights_folder logs/c3vdv2/stage1/models/weights_19/ \\
        --num_epochs          20 \\
        --w_depth_start       1.0 \\
        --w_depth_end         0.3 \\
        --w_pose_start        0.5 \\
        --w_pose_end          0.1 \\
        --ramp_epochs         8

    --load_weights_folder must contain:
        position_encoder.pth  position.pth     (frozen, from Stage 1)
        encoder.pth           depth.pth        (optional warm-start for Stage 2)
        pose_encoder.pth      pose.pth         (optional warm-start)
        transform_encoder.pth transform.pth    (optional warm-start)
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trainer_stage_two import Trainer as TrainerStageTwo
from options import MonodepthOptions
from datasets.c3vdv2_dataset import C3VDv2Dataset
from c3vdv2_losses import scale_invariant_log_loss, pose_loss

# Paper constants (Section 4.2)
PAPER_BETA1 = 0.9
PAPER_BETA2 = 0.99
PAPER_BATCH_SIZE = 12
PAPER_LR = 1e-4
PAPER_LR_STEP = 10
PAPER_NUM_EPOCHS = 20
PAPER_WIDTH = 320
PAPER_HEIGHT = 256
PAPER_NUM_SCALES = 4
PAPER_DISP_SMOOTH = 0.0001
PAPER_TRANSFORM_CONS = 1.0
PAPER_TRANSFORM_SMTH = 0.01

C3VD_MIN_DEPTH = 0.001  # 1 mm
C3VD_MAX_DEPTH = 0.100  # 100 mm


class C3VDv2Stage2Trainer(TrainerStageTwo):
    """
    Subclasses Stage 2 Trainer with:
      1. C3VDv2Dataset replacing SCAREDRAWDataset
      2. Supervised depth loss  (scale-invariant log)
      3. Supervised pose loss   (axis-angle + translation L2)
      4. Loss weight ramping schedule (high GT weight → lower as self-supervised
         signal becomes reliable)

    Preserved unchanged:
      ✓ set_train() / set_eval()
      ✓ run_epoch()  — single optimizer, single forward pass per batch
      ✓ predict_poses()  — frozen position → registration → transform → pose
      ✓ generate_images_pred()  — depth → 3D → reproject
      ✓ compute_reprojection_loss()
      ✓ val() / process_batch_val() / compute_losses_val()
      ✓ log(), save_model(), load_model()  (load_model freezes position net)
    """

    def __init__(self, options):
        # C3VDv2-specific params — must be stored before super().__init__()
        self.undistort_maps = getattr(options, "undistort_maps", "undistort_maps/")
        self.w_depth_start = getattr(options, "w_depth_start", 1.0)
        self.w_depth_end = getattr(options, "w_depth_end", 0.3)
        self.w_pose_start = getattr(options, "w_pose_start", 0.5)
        self.w_pose_end = getattr(options, "w_pose_end", 0.1)
        self.ramp_epochs = getattr(options, "ramp_epochs", 8)

        # Active weights — updated each epoch by _update_loss_weights()
        self.w_depth = self.w_depth_start
        self.w_pose = self.w_pose_start

        # Parent builds all models, single optimizer, SCARED dataloaders.
        # load_model() inside here freezes position_encoder + position.
        super().__init__(options)

        # Fix Adam β2 to paper value
        for pg in self.model_optimizer.param_groups:
            pg["betas"] = (PAPER_BETA1, PAPER_BETA2)

        # Replace SCARED dataloaders with C3VDv2
        train_dataset = C3VDv2Dataset(
            data_root=self.opt.data_path,
            split_file=os.path.join("splits", self.opt.split, "train.txt"),
            undistort_maps=self.undistort_maps,
            target_size=(self.opt.height, self.opt.width),
            augment=True,
        )
        val_dataset = C3VDv2Dataset(
            data_root=self.opt.data_path,
            split_file=os.path.join("splits", self.opt.split, "val.txt"),
            undistort_maps=self.undistort_maps,
            target_size=(self.opt.height, self.opt.width),
            augment=False,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = (
            len(train_dataset) // self.opt.batch_size * self.opt.num_epochs
        )

        print(f"\n{'='*60}")
        print("  Stage 2 Fine-Tuning  |  C3VDv2 Colonoscopy")
        print(f"{'='*60}")
        print(f"  Training:  encoder+depth  transform  pose_encoder+pose")
        print(f"  Frozen:    position_encoder + position  (Stage 1 weights)")
        print(
            f"  Resolution: {self.opt.width}x{self.opt.height}  Batch: {self.opt.batch_size}"
        )
        print(f"  LR: {self.opt.learning_rate}  Epochs: {self.opt.num_epochs}")
        print(
            f"  w_depth: {self.w_depth_start} → {self.w_depth_end}  "
            f"w_pose: {self.w_pose_start} → {self.w_pose_end}  "
            f"over {self.ramp_epochs} epochs"
        )
        print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")
        print(f"{'='*60}\n")

    # ── Loss weight schedule ──────────────────────────────────────────────────

    def _update_loss_weights(self):
        """
        Called at the start of each epoch.

        Phase 1 (epoch < ramp_epochs):  fixed at start values
        Phase 2 (epoch >= ramp_epochs): linear decay to end values

        Rationale: early epochs the registration/transform net hasn't adapted
        to colonoscopy yet, so the self-supervised reprojection loss is noisy.
        High GT weight keeps depth and pose anchored to correct values.
        As training progresses and the transform net adapts, we trust the
        self-supervised signal more and reduce GT supervision weight.
        """
        if self.epoch < self.ramp_epochs:
            self.w_depth = self.w_depth_start
            self.w_pose = self.w_pose_start
        else:
            # Linear decay from start to end over remaining epochs
            total_ramp = max(self.opt.num_epochs - self.ramp_epochs, 1)
            progress = min((self.epoch - self.ramp_epochs) / total_ramp, 1.0)
            self.w_depth = self.w_depth_start + progress * (
                self.w_depth_end - self.w_depth_start
            )
            self.w_pose = self.w_pose_start + progress * (
                self.w_pose_end - self.w_pose_start
            )

        # Log current weights to TensorBoard
        if hasattr(self, "writers"):
            self.writers["train"].add_scalar(
                "schedule/w_depth", self.w_depth, self.epoch
            )
            self.writers["train"].add_scalar("schedule/w_pose", self.w_pose, self.epoch)

    # ── Override run_epoch to call weight update at top of each epoch ─────────

    def run_epoch(self):
        self._update_loss_weights()
        print(
            f"  Epoch {self.epoch}  |  w_depth={self.w_depth:.3f}  w_pose={self.w_pose:.3f}"
        )
        super().run_epoch()

    # ── Override process_batch ────────────────────────────────────────────────

    def process_batch(self, inputs):
        inputs = self._adapt_keys(inputs)
        return super().process_batch(inputs)

    def process_batch_val(self, inputs):
        inputs = self._adapt_keys(inputs)
        return super().process_batch_val(inputs)

    # ── Override compute_losses ───────────────────────────────────────────────

    def compute_losses(self, inputs, outputs):
        """
        Calls parent's compute_losses (reprojection + transform + smooth),
        then injects supervised depth and pose losses on top.

        Parent loss uses:
          outputs[("color",    f_i, scale)]      depth-warped source frame
          outputs[("refined",  scale, f_i)]      brightness-calibrated target
          outputs[("occu_mask_backward", 0, f_i)] visibility mask
          outputs[("disp", scale)]               disparity

        Supervised losses use:
          outputs[("depth",       0, 0)]         finest-scale metric depth
          outputs[("axisangle",   0, f_i)]       shape (B,1,3) → squeeze to (B,3)
          outputs[("translation", 0, f_i)]       shape (B,1,3) → squeeze to (B,3)

        GT from inputs:
          inputs["gt_depth"]                     (B,1,H,W) metres
          inputs["axang_curr2prev/next"]         (B,3)
          inputs["trans_curr2prev/next"]         (B,3)
        """
        # ── Parent: reprojection + transform + smooth ─────────────────────────
        losses = super().compute_losses(inputs, outputs)

        # ── Supervised depth ──────────────────────────────────────────────────
        if "gt_depth" in inputs:
            gt_depth = inputs["gt_depth"]
            pred_depth = outputs[("depth", 0, 0)]  # finest scale, target frame

            # Validity mask: exclude no-data (0) and clamped-max pixels
            depth_mask = (
                (gt_depth > C3VD_MIN_DEPTH) & (gt_depth < C3VD_MAX_DEPTH - 0.001)
            ).float()
            if "occlusion" in inputs:
                depth_mask = depth_mask * (1.0 - inputs["occlusion"])

            l_depth = scale_invariant_log_loss(pred_depth, gt_depth, depth_mask)
            losses["loss/depth_supervised"] = l_depth
            losses["loss"] = losses["loss"] + self.w_depth * l_depth

        # ── Supervised pose ───────────────────────────────────────────────────
        # frame_ids[1:] = [-1, 1] → prev and next frames
        # outputs[("axisangle",   0, f_i)] shape (B,1,3)
        # outputs[("translation", 0, f_i)] shape (B,1,3)
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
                pred_ax = outputs[("axisangle", 0, f_i)].squeeze(1)  # (B,3)
                pred_tr = outputs[("translation", 0, f_i)].squeeze(1)  # (B,3)
                total_pose += pose_loss(pred_ax, pred_tr, inputs[axk], inputs[trk])
                n += 1
            if n > 0:
                l_pose = total_pose / n
                losses["loss/pose_supervised"] = l_pose
                losses["loss"] = losses["loss"] + self.w_pose * l_pose

        return losses

    # ── Key format adapter ────────────────────────────────────────────────────

    def _adapt_keys(self, batch):
        """
        Maps C3VDv2Dataset keys → (key, frame_id, scale) tuple keys that
        the Stage 2 trainer reads throughout process_batch, predict_poses,
        generate_images_pred, and compute_losses.

        C3VDv2Dataset key         Trainer key
        ─────────────────────────────────────────────────────────────
        "color_curr"          →   ("color",     0,  0) + ("color_aug", 0,  0)
        "color_prev"          →   ("color",    -1,  0) + ("color_aug",-1,  0)
        "color_next"          →   ("color",     1,  0) + ("color_aug", 1,  0)
        "K"  (B,3,3 norm.)    →   ("K", s) and ("inv_K", s) for s in 0..3
        gt/occ/pose keys      →   same keys  (pass through)

        Frame ID convention:  0=target  -1=prev  +1=next
        color_aug aliases color: augmentation already applied in C3VDv2Dataset.
        K is padded to (B,4,4) for BackprojectDepth / Project3D in layers.py.
        Coarser-scale K is the same as scale-0 K (fine-tuning from pretrained
        weights; the scale pyramid is already calibrated).
        """
        device = next(v.device for v in batch.values() if hasattr(v, "device"))
        out = {}

        # RGB
        for f_id, key in {0: "color_curr", -1: "color_prev", 1: "color_next"}.items():
            t = batch[key].to(device)
            out[("color", f_id, 0)] = t
            out[("color_aug", f_id, 0)] = t

        # Intrinsics: (B,3,3) → (B,4,4)
        K_33 = batch["K"].to(device)
        B = K_33.shape[0]
        K_44 = torch.zeros(B, 4, 4, device=device)
        K_44[:, :3, :3] = K_33
        K_44[:, 3, 3] = 1.0
        inv_K = torch.linalg.inv(K_44)
        for s in self.opt.scales:
            out[("K", s)] = K_44
            out[("inv_K", s)] = inv_K

        # Pass-through: gt_depth, occlusion, pose keys
        skip = {"color_curr", "color_prev", "color_next", "K", "seq", "frame_idx"}
        for key, val in batch.items():
            if key not in skip:
                out[key] = val.to(device) if hasattr(val, "to") else val

        return out


# =============================================================================
# CLI
# =============================================================================


def add_args(parser):
    parser.add_argument("--undistort_maps", type=str, default="undistort_maps/")
    parser.add_argument(
        "--w_depth_start",
        type=float,
        default=1.0,
        help="Supervised depth weight at start of training",
    )
    parser.add_argument(
        "--w_depth_end",
        type=float,
        default=0.3,
        help="Supervised depth weight at end of training",
    )
    parser.add_argument(
        "--w_pose_start",
        type=float,
        default=0.5,
        help="Supervised pose weight at start of training",
    )
    parser.add_argument(
        "--w_pose_end",
        type=float,
        default=0.1,
        help="Supervised pose weight at end of training",
    )
    parser.add_argument(
        "--ramp_epochs",
        type=int,
        default=8,
        help="Epochs before loss weights start ramping down",
    )
    return parser


if __name__ == "__main__":
    opts = MonodepthOptions()
    add_args(opts.parser)
    opt = opts.parse()

    if opt.dataset in ("endovis", "kitti"):
        opt.dataset = "c3vdv2"
    if opt.split in ("endovis", "eigen_zhou"):
        opt.split = "c3vdv2"
    opt.height = opt.height if opt.height != 192 else PAPER_HEIGHT
    opt.width = opt.width if opt.width != 640 else PAPER_WIDTH
    opt.num_scales = PAPER_NUM_SCALES
    opt.scales = list(range(PAPER_NUM_SCALES))
    opt.min_depth = C3VD_MIN_DEPTH
    opt.max_depth = C3VD_MAX_DEPTH
    if hasattr(opt, "disparity_smoothness"):
        opt.disparity_smoothness = PAPER_DISP_SMOOTH
    if hasattr(opt, "transform_constraint"):
        opt.transform_constraint = PAPER_TRANSFORM_CONS
    if hasattr(opt, "transform_smoothness"):
        opt.transform_smoothness = PAPER_TRANSFORM_SMTH

    trainer = C3VDv2Stage2Trainer(opt)
    trainer.train()
