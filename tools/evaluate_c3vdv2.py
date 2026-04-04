"""
evaluate_c3vdv2.py
------------------
Evaluates a fine-tuned AF-SfMLearner on C3VDv2 test sequences.

Reports standard depth metrics (AbsRel, SqRel, RMSE, RMSElog, δ<1.25)
plus polyp-region metrics when a segmentation mask directory is provided.

Run from AF-SfMLearner repo root:

    python evaluate_c3vdv2.py \
        --data_root       /path/to/C3VDv2 \
        --split_file      splits/c3vdv2/test.txt \
        --undistort_maps  /path/to/undistort_maps \
        --weights_folder  logs/c3vdv2_finetune/weights_29 \
        --seg_root        /path/to/segmentation_masks   # optional
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Optional
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from networks import ResnetEncoder, DepthDecoder
from datasets.c3vdv2_dataset import C3VDv2Dataset


MIN_DEPTH = 0.001
MAX_DEPTH = 0.100


def compute_depth_metrics(gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None):
    """
    Standard depth estimation metrics.
    gt, pred: 1D arrays of valid depth values in metres.
    """
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]

    # Median scale alignment (standard for self-supervised methods)
    scale = np.median(gt) / np.median(pred)
    pred = pred * scale

    pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


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

    # ---- Dataset ---------------------------------------------------------
    dataset = C3VDv2Dataset(
        data_root=args.data_root,
        split_file=args.split_file,
        undistort_maps=args.undistort_maps,
        target_size=(args.height, args.width),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # ---- Evaluation loop -------------------------------------------------
    all_metrics = []
    polyp_metrics = []  # filled if seg_root provided

    _depth_debug_done = False
    with torch.no_grad():
        for batch in loader:
            color_curr = batch["color_curr"].to(device)
            gt_depth = batch["gt_depth"].numpy().squeeze()  # H,W
            seq = batch["seq"][0]
            frame_idx = batch["frame_idx"].item()

            if not _depth_debug_done:
                import cv2 as _cv2
                from datasets.c3vdv2_dataset import decode_depth_tiff as _decode, load_undistort_maps as _load_maps
                tiff_path = os.path.join(args.data_root, seq, "depth", f"{frame_idx:04d}_depth.tiff")
                print(f"\n[DEBUG] First batch: seq={seq}, frame={frame_idx}")
                raw = _cv2.imread(tiff_path, _cv2.IMREAD_ANYDEPTH)
                print(f"[DEBUG] raw TIFF   dtype={raw.dtype}  shape={raw.shape}  min={raw.min()}  max={raw.max()}")
                depth_decoded = _decode(tiff_path)
                print(f"[DEBUG] after decode (m): nonzero={np.count_nonzero(depth_decoded)}  max={depth_decoded.max():.4f}")
                map1, map2 = _load_maps(args.undistort_maps)
                h_src, w_src = depth_decoded.shape
                # Inspect centre pixel of the 512x512 output map
                cy_m, cx_m = map1.shape[0] // 2, map1.shape[1] // 2
                mx, my = map1[cy_m, cx_m], map2[cy_m, cx_m]
                print(f"[DEBUG] centre map pixel → src x={mx:.1f}  y={my:.1f}  (src WxH={w_src}x{h_src})")
                # Fraction of map coords that are within source bounds
                valid_x = (map1 >= 0) & (map1 < w_src)
                valid_y = (map2 >= 0) & (map2 < h_src)
                both_valid = valid_x & valid_y
                n = map1.size
                print(f"[DEBUG] map coords in-bounds: x={valid_x.sum()}/{n}  y={valid_y.sum()}/{n}  both={both_valid.sum()}/{n}")
                # Check depth at those valid locations
                if both_valid.sum() > 0:
                    xs = map1[both_valid].astype(int)
                    ys = map2[both_valid].astype(int)
                    vals = depth_decoded[ys, xs]
                    print(f"[DEBUG] depth at valid map locs: min={vals.min():.4f}  max={vals.max():.4f}  nonzero={np.count_nonzero(vals)}")
                else:
                    print("[DEBUG] NO map coordinates fall within source image bounds — maps are miscalibrated")
                # Also try plain resize (no remap) for comparison
                depth_resized = _cv2.resize(depth_decoded, (args.width, args.height), interpolation=_cv2.INTER_NEAREST)
                print(f"[DEBUG] plain resize (no remap): max={depth_resized.max():.4f}  nonzero={np.count_nonzero(depth_resized)}")
                _depth_debug_done = True

            features = encoder(color_curr)
            depth_out = depth_decoder(features)
            disp = depth_out[("disp", 0)].cpu().numpy().squeeze()

            # Convert disparity to metric depth
            pred_depth = 1.0 / (disp * (1 / MIN_DEPTH - 1 / MAX_DEPTH) + 1 / MAX_DEPTH)

            # Build validity mask
            valid_mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            if valid_mask.sum() < 10:
                continue

            metrics = compute_depth_metrics(gt_depth, pred_depth, valid_mask)
            all_metrics.append(metrics)

            # ---- Polyp-region metrics (if segmentation available) --------
            if args.seg_root:
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
                        pm = compute_depth_metrics(gt_depth, pred_depth, polyp_mask)
                        polyp_metrics.append(pm)

    # ---- Report ----------------------------------------------------------
    print("\n========== C3VDv2 Depth Evaluation ==========")
    print(f"  Samples in split:  {len(dataset)}")
    print(f"  Samples evaluated: {len(all_metrics)}")
    if len(all_metrics) == 0:
        print("  [WARNING] No samples were evaluated.")
        print("  Possible causes:")
        print("    - split file is empty (re-run generate_splits.py with --test_segs)")
        print("    - all frames failed the depth validity mask (depth out of 1–100 mm range)")
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
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_file", default=None,
                        help="Path to split txt file. Mutually exclusive with --split.")
    parser.add_argument("--split", default=None,
                        help="Split name (e.g. c3vdv2); resolves to splits/<split>/test.txt")
    parser.add_argument("--undistort_maps", required=True)
    parser.add_argument("--weights_folder", required=True)
    parser.add_argument(
        "--seg_root",
        default=None,
        help="Directory of segmentation masks for polyp-region metrics",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    if args.split_file is None and args.split is None:
        parser.error("one of --split_file or --split is required")
    if args.split_file is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.split_file = os.path.join(repo_root, "splits", args.split, "test.txt")

    evaluate(args)
