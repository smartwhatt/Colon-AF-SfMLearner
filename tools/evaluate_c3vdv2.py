"""
evaluate_c3vdv2.py
------------------
Evaluates a fine-tuned AF-SfMLearner on C3VDv2 test sequences.

Reports standard depth metrics (AbsRel, SqRel, RMSE, RMSElog, δ<1.25)
plus polyp-region metrics when a segmentation mask directory is provided.

Run from AF-SfMLearner repo root:

    python tools/evaluate_c3vdv2.py \
        --data_root       /path/to/C3VDv2 \
        --split_file      splits/c3vdv2/test.txt \
        --undistort_maps  /path/to/undistort_maps \
        --weights_folder  logs/c3vdv2_finetune/weights_29 \
        --seg_root        /path/to/segmentation_masks   # optional

Or evaluate the Hugging Face C3VDv2 package directly:

    python tools/evaluate_c3vdv2.py \
        --data_source     hf \
        --hf_repo_id      username/c3vdv2 \
        --hf_split        test \
        --weights_folder  logs/c3vdv2/stage2/models/weights_19

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from networks import ResnetEncoder, DepthDecoder
from colon_datasets.c3vdv2_dataset import C3VDv2Dataset
from colon_datasets.c3vdv2_metrics import (
    C3VD_MAX_DEPTH,
    C3VD_MIN_DEPTH,
    build_c3vdv2_valid_mask_np,
    compute_depth_metrics_np,
)


def _build_loader(args):
    if args.data_source == "hf":
        from colon_datasets.hf_triplet_dataset import build_hf_triplet_dataset

        dataset = build_hf_triplet_dataset(
            repo_id=args.hf_repo_id,
            split=args.hf_split,
            streaming=args.hf_streaming,
            cache_dir=args.hf_cache_dir,
            target_size=(args.height, args.width),
            is_train=False,
            output_format="c3vdv2",
            scales=(0,),
            shuffle_buffer=0,
        )
        num_workers = 0 if args.hf_streaming else args.num_workers
    else:
        dataset = C3VDv2Dataset(
            data_root=args.data_root,
            split_file=args.split_file,
            undistort_maps=args.undistort_maps,
            target_size=(args.height, args.width),
            augment=False,
        )
        num_workers = args.num_workers

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return dataset, loader


def _safe_len(dataset):
    try:
        return len(dataset)
    except TypeError:
        return None


def _batch_metadata(batch):
    seq = ""
    frame_idx = None

    if "seq" in batch:
        seq_value = batch["seq"]
        seq = seq_value[0] if isinstance(seq_value, (list, tuple)) else str(seq_value)
    if "frame_idx" in batch:
        idx_value = batch["frame_idx"]
        if hasattr(idx_value, "item"):
            frame_idx = int(idx_value.item())
        elif isinstance(idx_value, (list, tuple)) and idx_value:
            frame_idx = int(idx_value[0])
        else:
            frame_idx = int(idx_value)

    return seq, frame_idx


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ------------------------------------------------------
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4))

    encoder.load_state_dict(
        torch.load(os.path.join(args.weights_folder, "encoder.pth"), map_location="cpu"),
        strict=False,
    )
    depth_decoder.load_state_dict(
        torch.load(os.path.join(args.weights_folder, "depth.pth"), map_location="cpu")
    )
    encoder.eval().to(device)
    depth_decoder.eval().to(device)

    # ---- Dataset (used for colour frames and metadata only) --------------
    dataset, loader = _build_loader(args)
    sample_count = _safe_len(dataset)

    # ---- Evaluation loop -------------------------------------------------
    all_metrics = []
    polyp_metrics = []

    with torch.no_grad():
        for batch in loader:
            if "gt_depth" not in batch:
                raise ValueError(
                    "Evaluation requires ground-truth depth. "
                    "Use a C3VDv2 HF split with depth labels."
                )
            color_curr = batch["color_curr"].to(device)
            gt_depth = batch["gt_depth"].numpy().squeeze()  # H,W
            seq, frame_idx = _batch_metadata(batch)

            features = encoder(color_curr)
            depth_out = depth_decoder(features)
            disp = depth_out[("disp", 0)].cpu().numpy().squeeze()

            # Convert disparity to metric depth
            pred_depth = 1.0 / (
                disp * (1 / C3VD_MIN_DEPTH - 1 / C3VD_MAX_DEPTH) + 1 / C3VD_MAX_DEPTH
            )

            # Build validity mask
            valid_mask = build_c3vdv2_valid_mask_np(gt_depth)
            if valid_mask.sum() < 10:
                continue

            metrics = compute_depth_metrics_np(
                gt_depth,
                pred_depth,
                valid_mask,
                median_scaling=args.median_scaling,
            )
            all_metrics.append([metrics[k] for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")])

            # ---- Polyp-region metrics (if segmentation available) --------
            if args.seg_root and seq and frame_idx is not None:
                seg_path = os.path.join(args.seg_root, seq, f"{frame_idx:04d}_seg.png")
                if os.path.exists(seg_path):
                    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    if seg is None:
                        continue
                    seg = cv2.resize(
                        seg, (args.width, args.height), interpolation=cv2.INTER_NEAREST
                    )
                    polyp_mask = (seg > 127) & valid_mask
                    if polyp_mask.sum() > 10:
                        pm = compute_depth_metrics_np(
                            gt_depth,
                            pred_depth,
                            polyp_mask,
                            median_scaling=args.median_scaling,
                        )
                        polyp_metrics.append(
                            [pm[k] for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")]
                        )

    # ---- Report ----------------------------------------------------------
    print("\n========== C3VDv2 Depth Evaluation ==========")
    if sample_count is None:
        print("  Samples in split:  unknown (streaming)")
    else:
        print(f"  Samples in split:  {sample_count}")
    print(f"  Samples evaluated: {len(all_metrics)}")
    print(f"  Median scaling:    {'on' if args.median_scaling else 'off'}")
    if len(all_metrics) == 0:
        print("  [WARNING] No samples passed the depth validity mask.")
        print("==============================================\n")
        return
    metrics_arr = np.array(all_metrics)
    mean = metrics_arr.mean(axis=0)
    print(f"  AbsRel:   {mean[0]:.4f}")
    print(f"  SqRel:    {mean[1]:.4f}")
    print(f"  RMSE(m):  {mean[2]:.4f}   ({mean[2]*1000:.2f} mm)")
    print(f"  RMSElog:  {mean[3]:.4f}")
    print(f"  δ<1.25:   {mean[4]:.4f}")
    print(f"  δ<1.25²:  {mean[5]:.4f}")
    print(f"  δ<1.25³:  {mean[6]:.4f}")

    if polyp_metrics:
        pm_arr = np.array(polyp_metrics)
        pm_mean = pm_arr.mean(axis=0)
        print("\n--- Polyp-region metrics ---")
        print(f"  Samples: {len(polyp_metrics)}")
        print(f"  AbsRel:  {pm_mean[0]:.4f}")
        print(f"  RMSE(m): {pm_mean[2]:.4f}   ({pm_mean[2]*1000:.2f} mm)")
        print(f"  δ<1.25:  {pm_mean[4]:.4f}")
    print("==============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source",
        choices=["local", "hf"],
        default="local",
        help="data backend to use, matching training",
    )
    parser.add_argument("--data_root", "--data_path", dest="data_root", default=None)
    parser.add_argument("--split_file", default=None,
                        help="Path to split txt file. Mutually exclusive with --split.")
    parser.add_argument("--split", default=None,
                        help="Split name (e.g. c3vdv2); resolves to splits/<split>/test.txt")
    parser.add_argument("--undistort_maps", default=None)
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_split", default="test")
    parser.add_argument("--hf_streaming", action="store_true")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--weights_folder", "--load_weights_folder", dest="weights_folder", required=True)
    parser.add_argument(
        "--seg_root",
        default=None,
        help="Directory of segmentation masks for polyp-region metrics",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--median_scaling",
        action="store_true",
        help="Apply per-frame median scale alignment before metrics",
    )
    args = parser.parse_args()

    if args.data_source == "hf":
        if args.hf_repo_id is None:
            parser.error("--hf_repo_id is required when --data_source hf")
    else:
        if args.data_root is None:
            parser.error("--data_root/--data_path is required when --data_source local")
        if args.undistort_maps is None:
            parser.error("--undistort_maps is required when --data_source local")
        if args.split_file is None and args.split is None:
            parser.error("one of --split_file or --split is required when --data_source local")
        if args.split_file is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            args.split_file = os.path.join(repo_root, "splits", args.split, "test.txt")

    evaluate(args)
