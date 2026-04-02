"""
train.py
--------
Single entry point for all training modes.

Usage:
    # Stage 1 – train registration net on SCARED
    python train.py --mode stage1 --data_path /data/endovis --log_dir logs/

    # Stage 2 – train depth/pose/transform (load Stage 1 weights)
    python train.py --mode stage2 --data_path /data/endovis --log_dir logs/ \\
        --load_weights_folder logs/stage1/models/weights_19/

    # End-to-end – train everything simultaneously
    python train.py --mode end_to_end --data_path /data/endovis --log_dir logs/

    # C3VDv2 fine-tune Stage 1
    python train.py --mode c3vdv2_stage1 --dataset c3vdv2 --split c3vdv2 \\
        --data_path /data/C3VDv2 --log_dir logs/ --undistort_maps undistort_maps/

    # C3VDv2 fine-tune Stage 2 (with supervised depth + pose)
    python train.py --mode c3vdv2_stage2 --dataset c3vdv2 --split c3vdv2 \\
        --data_path /data/C3VDv2 --log_dir logs/ --undistort_maps undistort_maps/ \\
        --load_weights_folder logs/c3vdv2_stage1/models/weights_19/ \\
        --w_depth_start 1.0 --w_depth_end 0.3 --w_pose_start 0.5 --w_pose_end 0.1
"""

from options import MonodepthOptions
from trainers import TRAINERS

if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.add_argument(
        "--mode",
        type=str,
        choices=list(TRAINERS.keys()),
        default="stage2",
        help=(
            "Training mode: stage1 | stage2 | end_to_end | "
            "c3vdv2_stage1 | c3vdv2_stage2"
        ),
    )
    opt = options.parse()

    trainer_cls = TRAINERS[opt.mode]
    trainer = trainer_cls(opt)
    trainer.train()
