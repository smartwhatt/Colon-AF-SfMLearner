"""
finetune_c3vdv2_stage1.py
--------------------------
Fine-tune the AF-SfMLearner Stage 1 trainer (registration/correspondence net)
on C3VDv2 colonoscopy data.

WHAT STAGE 1 DOES:
  Trains position_encoder + position (PositionDecoder) only.
  These form the correspondence/registration module — they take a concatenated
  pair [I_source, I_target] and predict a 2D flow field that geometrically
  aligns I_source to I_target's viewpoint via a SpatialTransformer warp.

  Loss = min_over_frames(Φ(registration, target)) + λ * smooth(position)

  This is purely self-supervised — no GT depth or pose is used.
  The network learns the appearance of colonoscopy tissue and the
  typical motion patterns of the colonoscope from raw video.

WHY THIS MUST RUN BEFORE STAGE 2:
  Stage 2 freezes this net and uses its output (registration) as:
    1. Input to the transform/brightness-calibration network
    2. The geometric anchor for the occlusion mask
    3. The `refined` supervision target for depth/pose training
  If the registration net has never seen colonoscopy tissue, all three
  of these will be wrong, corrupting Stage 2's self-supervised signal.

TRAINING SCHEDULE (paper Section 4.2, Table 8):
  20 epochs, lr=1e-4, ×0.1 after epoch 10
  Adam β1=0.9, β2=0.99, batch=12
  Resolution 320×256, 4 scales

Usage:
    CUDA_VISIBLE_DEVICES=0 python finetune_c3vdv2_stage1.py \\
        --data_path           /path/to/C3VDv2 \\
        --split               c3vdv2 \\
        --undistort_maps      undistort_maps/ \\
        --log_dir             logs/c3vdv2 \\
        --model_name          stage1 \\
        --load_weights_folder pretrained/scared_stage1/   # optional SCARED init
        --num_epochs          20

    Checkpoints saved to: logs/c3vdv2/stage1/models/weights_N/
    Pass logs/c3vdv2/stage1/models/weights_19/ as --load_weights_folder to stage2.
"""

import os
import torch
from torch.utils.data import DataLoader

from trainer_stage_one import Trainer as TrainerStageOne
from options import MonodepthOptions
from datasets.c3vdv2_dataset import C3VDv2Dataset

# Paper constants (Section 4.2)
PAPER_BETA1 = 0.9
PAPER_BETA2 = 0.99  # NOT PyTorch default 0.999
PAPER_BATCH_SIZE = 12
PAPER_LR = 1e-4
PAPER_LR_STEP = 10
PAPER_NUM_EPOCHS = 20
PAPER_WIDTH = 320
PAPER_HEIGHT = 256
PAPER_NUM_SCALES = 4
PAPER_DISP_SMTH = 0.0001  # position_smoothness λ


class C3VDv2Stage1Trainer(TrainerStageOne):
    """
    Subclasses Stage 1 Trainer.

    The only change from the parent is:
      - Replace SCAREDRAWDataset with C3VDv2Dataset
      - Fix Adam β2 to paper value

    Everything else — predict_poses(), compute_losses(), run_epoch(),
    val(), save/load — runs unchanged from trainer_stage_one.py.

    Stage 1 has no GT supervision — photometric + smoothness only.
    The self-supervised signal is sufficient here because we just
    need the registration net to learn colonoscopy appearance.
    """

    def __init__(self, options):
        self.undistort_maps = getattr(options, "undistort_maps", "undistort_maps/")
        super().__init__(options)

        # Fix β2
        for pg in self.model_optimizer.param_groups:
            pg["betas"] = (PAPER_BETA1, PAPER_BETA2)

        # Replace SCARED dataloaders
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
        print("  Stage 1 Fine-Tuning  |  C3VDv2 Colonoscopy")
        print(f"{'='*60}")
        print(f"  Training:  position_encoder + position  (only)")
        print(f"  Loss:      photometric + smoothness  (no GT used)")
        print(f"  Purpose:   learn colonoscopy tissue appearance + scope motion")
        print(
            f"  Resolution: {self.opt.width}x{self.opt.height}  Batch: {self.opt.batch_size}"
        )
        print(f"  LR: {self.opt.learning_rate}  Epochs: {self.opt.num_epochs}")
        print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")
        print(f"{'='*60}\n")

    def process_batch(self, inputs):
        """Adapt C3VDv2Dataset keys → trainer tuple keys, then run parent."""
        inputs = self._adapt_keys(inputs)
        return super().process_batch(inputs)

    def process_batch_val(self, inputs):
        inputs = self._adapt_keys(inputs)
        return super().process_batch_val(inputs)

    def _adapt_keys(self, batch):
        """
        C3VDv2Dataset key     →  Trainer key
        ──────────────────────────────────────
        "color_curr"          →  ("color",     0, 0) and ("color_aug", 0, 0)
        "color_prev"          →  ("color",    -1, 0) and ("color_aug",-1, 0)
        "color_next"          →  ("color",     1, 0) and ("color_aug", 1, 0)

        Stage 1 only uses ("color_aug", f_i, 0) inside predict_poses().
        K / gt_depth / pose keys are not needed by Stage 1 at all,
        but we pass them through in case val logging tries to access them.
        """
        device = next(v.device for v in batch.values() if hasattr(v, "device"))
        out = {}

        frame_map = {0: "color_curr", -1: "color_prev", 1: "color_next"}
        for f_id, key in frame_map.items():
            t = batch[key].to(device)
            out[("color", f_id, 0)] = t
            out[("color_aug", f_id, 0)] = t  # augmentation already applied in dataset

        # Pass through anything else (gt_depth, occlusion, poses) for safety
        skip = {"color_curr", "color_prev", "color_next"}
        for key, val in batch.items():
            if key not in skip:
                out[key] = val.to(device) if hasattr(val, "to") else val

        return out


# =============================================================================
# CLI
# =============================================================================


def add_args(parser):
    parser.add_argument(
        "--undistort_maps",
        type=str,
        default="undistort_maps/",
        help="Directory with undistort_map1.npy / undistort_map2.npy",
    )
    return parser


if __name__ == "__main__":
    opts = MonodepthOptions()
    add_args(opts.parser)
    opt = opts.parse()

    # Apply paper defaults
    if opt.dataset in ("endovis", "kitti"):
        opt.dataset = "c3vdv2"
    if opt.split in ("endovis", "eigen_zhou"):
        opt.split = "c3vdv2"
    opt.height = opt.height if opt.height != 192 else PAPER_HEIGHT
    opt.width = opt.width if opt.width != 640 else PAPER_WIDTH
    opt.num_scales = PAPER_NUM_SCALES
    opt.scales = list(range(PAPER_NUM_SCALES))
    if hasattr(opt, "position_smoothness"):
        opt.position_smoothness = PAPER_DISP_SMTH

    trainer = C3VDv2Stage1Trainer(opt)
    trainer.train()
