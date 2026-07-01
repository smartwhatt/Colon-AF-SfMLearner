"""
evaluate_c3vdv2.py
------------------
Evaluates a fine-tuned AF-SfMLearner on C3VDv2 test sequences.

Reports standard depth metrics (AbsRel, SqRel, RMSE, RMSElog, δ<1.25)
plus polyp-region metrics when a segmentation mask directory is provided.

Run from AF-SfMLearner repo root:

    python tools/evaluate_c3vdv2.py \\
        --data_root       /path/to/C3VDv2 \\
        --split_file      splits/c3vdv2/test.txt \\
        --undistort_maps  /path/to/undistort_maps \\
        --weights_folder  logs/c3vdv2_finetune/weights_29 \\
        --seg_root        /path/to/segmentation_masks   # optional

Or evaluate the Hugging Face C3VDv2 package directly:

    python tools/evaluate_c3vdv2.py \\
        --data_source     hf \\
        --hf_repo_id      username/c3vdv2 \\
        --hf_split        test \\
        --weights_folder  logs/c3vdv2/stage2/models/weights_19

Stratified mode (--stratify) adds per-frame JSONL, depth-band breakdowns,
specular-region metrics, and optional top-K artifact output:

    python tools/evaluate_c3vdv2.py \\
        --data_source hf \\
        --hf_repo_id SmartWhatt/c3vdv2-SfM \\
        --hf_split test \\
        --weights_folder logs/c3vdv2_hf/stage2_specular_pps/models/weights_best \\
        --run_id run_d \\
        --stratify \\
        --metrics_out reports/c3vdv2/run_d_metrics.jsonl \\
        --summary_out reports/c3vdv2/run_d_strata.md \\
        --topk_out_dir reports/c3vdv2/run_d_topk \\
        --load_specular_weights

"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from networks import ResnetEncoder, DepthDecoder, SpecularDecoder
from colon_datasets.c3vdv2_dataset import C3VDv2Dataset
from colon_datasets.c3vdv2_metrics import (
    C3VD_MAX_DEPTH,
    C3VD_MIN_DEPTH,
    DEPTH_METRIC_KEYS,
    build_c3vdv2_valid_mask_np,
    compute_depth_metrics_np,
)
from tools.depth_backend import (
    load_depth_model,
    predict_depth_m,
    requires_weights_folder,
)


# ---------------------------------------------------------------------------
# Depth-band binning helpers (metric: metres → mm for bin names only)
# ---------------------------------------------------------------------------

DEPTH_BAND_EDGES_MM = [0, 5, 10, 20, 50, 100]
DEPTH_BAND_NAMES = ["0_5mm", "5_10mm", "10_20mm", "20_50mm", "50_100mm"]


def _depth_bands_np(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    median_scaling: bool,
    min_pixels: int = 10,
) -> dict[str, dict | None]:
    """Return a dict ``{band_name: metrics_dict | None}`` for each depth band.

    Bands with fewer than *min_pixels* valid pixels are recorded as ``null``.
    """
    gt_mm = gt_depth * 1000.0  # metres → mm for thresholding only
    results: dict[str, dict | None] = {}
    for i, name in enumerate(DEPTH_BAND_NAMES):
        lo, hi = DEPTH_BAND_EDGES_MM[i], DEPTH_BAND_EDGES_MM[i + 1]
        band_mask = valid_mask & (gt_mm >= lo) & (gt_mm < hi)
        if band_mask.sum() < min_pixels:
            results[name] = None
        else:
            results[name] = compute_depth_metrics_np(
                gt_depth, pred_depth, band_mask, median_scaling=median_scaling
            )
    return results


# ---------------------------------------------------------------------------
# Specular pseudo-mask (intensity-based, no model required)
# ---------------------------------------------------------------------------

def _specular_intensity_mask_np(
    color_np: np.ndarray,
    max_channel_thresh: float = 0.98,
    intensity_thresh: float = 0.95,
    dilate_radius: int = 2,
) -> np.ndarray:
    """Build a specular pseudo-mask from an RGB image (C,H,W or H,W,C in [0,1]).

    Returns a boolean array of shape ``(H, W)`` where saturated / high-intensity
    pixels are ``True``.
    """
    if color_np.shape[0] == 3:
        rgb = np.transpose(color_np, (1, 2, 0))  # C,H,W → H,W,C
    else:
        rgb = color_np

    max_ch = rgb.max(axis=2)
    intensity = rgb.mean(axis=2)

    mask = (max_ch > max_channel_thresh) | (intensity > intensity_thresh)

    if dilate_radius > 0 and mask.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    return mask


# ---------------------------------------------------------------------------
# Specular model mask (requires SpecularDecoder + encoder features)
# ---------------------------------------------------------------------------

def _specular_model_mask_np(
    specular_out: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert SpecularDecoder sigmoid output to a boolean mask.

    Args:
        specular_out: (H, W) float array in [0, 1].
        threshold: decision boundary.

    Returns:
        Boolean array of shape ``(H, W)``.
    """
    return specular_out > threshold


# ---------------------------------------------------------------------------
# Error-tail metrics
# ---------------------------------------------------------------------------

def _error_tail_np(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Compute p90/p95 absolute relative error and per-pixel metrics."""
    gt = gt_depth[mask].astype(np.float64)
    pred = pred_depth[mask].astype(np.float64)
    if gt.size == 0:
        return {"abs_rel_p90": float("nan"), "abs_rel_p95": float("nan"),
                "p95_rmse_contrib": float("nan")}

    gt = np.maximum(gt, eps)
    pred = np.maximum(pred, eps)
    abs_rel = np.abs(gt - pred) / gt
    sq_error = (gt - pred) ** 2

    return {
        "abs_rel_p90": float(np.percentile(abs_rel, 90)),
        "abs_rel_p95": float(np.percentile(abs_rel, 95)),
        "p95_rmse_contrib": float(sq_error[abs_rel >= np.percentile(abs_rel, 95)].sum())
        if sq_error.size > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Specular metrics (inside + outside mask)
# ---------------------------------------------------------------------------

def _specular_metrics_np(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    spec_mask: np.ndarray,
    median_scaling: bool,
    min_pixels: int = 10,
) -> dict[str, dict | None]:
    """Return ``{"inside": metrics | None, "outside": metrics | None}``."""
    inside = valid_mask & spec_mask
    outside = valid_mask & ~spec_mask

    result: dict[str, dict | None] = {}
    for label, mask in [("inside", inside), ("outside", outside)]:
        n = int(mask.sum())
        if n < min_pixels:
            result[label] = None
        else:
            m = compute_depth_metrics_np(
                gt_depth, pred_depth, mask, median_scaling=median_scaling
            )
            m["pixel_count"] = float(n)
            result[label] = m
    return result


# ---------------------------------------------------------------------------
# Visualisation helpers for top-K artifacts
# ---------------------------------------------------------------------------

_COLORMAP = cv2.COLORMAP_TURBO


def _depth_vis(depth_m: np.ndarray, vmin: float = 0.0, vmax: float = 0.1) -> np.ndarray:
    """Render a metric depth array as a BGR turbo colormap image."""
    clipped = np.clip(depth_m, vmin, vmax)
    norm = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, _COLORMAP)


def _error_heatmap(gt: np.ndarray, pred: np.ndarray, vmax: float = 0.05) -> np.ndarray:
    """Render |gt - pred| as a BGR turbo heatmap."""
    err = np.abs(gt - pred)
    err = np.clip(err, 0, vmax)
    norm = (err / vmax * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, _COLORMAP)


def _color_to_bgr(color_np: np.ndarray) -> np.ndarray:
    """Convert a C,H,W float RGB tensor (numpy) to a BGR uint8 image."""
    if color_np.shape[0] == 3:
        rgb = np.transpose(color_np, (1, 2, 0))
    else:
        rgb = color_np
    rgb = np.clip(rgb, 0.0, 1.0)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _save_topk_artifacts(
    out_dir: str,
    topk_frames: list[dict],
    include_specular: bool = False,
) -> None:
    """Write PNG artifacts and a ``topk_index.json`` for the worst-K frames."""
    os.makedirs(out_dir, exist_ok=True)

    for rank, entry in enumerate(topk_frames, start=1):
        prefix = f"rank{rank:03d}_{entry['seq']}_{entry.get('frame_idx','n/a')}"

        if "color_bgr" in entry:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_rgb.png"), entry["color_bgr"])

        if "pred_vis" in entry:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_pred_depth.png"), entry["pred_vis"])

        if "gt_vis" in entry:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_gt_depth.png"), entry["gt_vis"])

        if "error_vis" in entry:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_error.png"), entry["error_vis"])

        if include_specular and "specular_mask" in entry:
            mask_vis = (entry["specular_mask"].astype(np.uint8) * 255) if entry["specular_mask"].dtype == bool else entry["specular_mask"]
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_specular.png"), mask_vis)

    # Write index with artifact paths (strip numpy arrays)
    index_entries = []
    for rank, entry in enumerate(topk_frames, start=1):
        prefix = f"rank{rank:03d}_{entry['seq']}_{entry.get('frame_idx','n/a')}"
        clean = {
            "rank": rank,
            "seq": entry["seq"],
            "frame_idx": entry.get("frame_idx"),
            "abs_rel": entry.get("abs_rel"),
            "rmse": entry.get("rmse"),
            "artifacts": {
                k: f"{prefix}_{k}.png"
                for k in ["rgb", "pred_depth", "gt_depth", "error"]
            },
        }
        if include_specular and "specular_mask" in entry:
            clean["artifacts"]["specular"] = f"{prefix}_specular.png"
        index_entries.append(clean)

    with open(os.path.join(out_dir, "topk_index.json"), "w") as f:
        json.dump(index_entries, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Summary markdown writer
# ---------------------------------------------------------------------------

def _write_summary_md(
    path: str,
    run_id: str,
    all_metrics: list[dict],
) -> None:
    """Write a per-run stratified summary in markdown."""
    if not all_metrics:
        with open(path, "w") as f:
            f.write(f"# {run_id} Stratified Summary\n\n*No frames evaluated.*\n")
        return

    lines = [f"# {run_id} Stratified Summary", ""]

    # --- Aggregate ---
    lines.append("## Aggregate")
    lines.append("")
    glob = [m["global"] for m in all_metrics if m.get("global")]
    if glob:
        lines.append(_metrics_table_md(glob))
        lines.append("")

    # --- Depth bands ---
    lines.append("## Depth Bands")
    lines.append("")
    for band_name in DEPTH_BAND_NAMES:
        band_metrics = [
            m["depth_bands"][band_name]
            for m in all_metrics
            if m.get("depth_bands", {}).get(band_name) is not None
        ]
        if band_metrics:
            lines.append(f"### {band_name}  (n={len(band_metrics)})")
            lines.append("")
            lines.append(_metrics_table_md(band_metrics))
            lines.append("")
        else:
            lines.append(f"### {band_name}  (no samples)")
            lines.append("")

    # --- Specular intensity ---
    lines.append("## Specular (intensity pseudo-mask)")
    lines.append("")
    for region in ("inside", "outside"):
        region_metrics = [
            m["specular_intensity"][region]
            for m in all_metrics
            if m.get("specular_intensity", {}).get(region) is not None
        ]
        if region_metrics:
            lines.append(f"### {region}  (n={len(region_metrics)})")
            lines.append("")
            lines.append(_metrics_table_md(region_metrics))
            lines.append("")

    # --- Specular model ---
    has_model = any(
        m.get("specular_model", {}).get("inside") is not None
        or m.get("specular_model", {}).get("outside") is not None
        for m in all_metrics
    )
    if has_model:
        lines.append("## Specular (model mask)")
        lines.append("")
        for region in ("inside", "outside"):
            region_metrics = [
                m["specular_model"][region]
                for m in all_metrics
                if m.get("specular_model", {}).get(region) is not None
            ]
            if region_metrics:
                lines.append(f"### {region}  (n={len(region_metrics)})")
                lines.append("")
                lines.append(_metrics_table_md(region_metrics))
                lines.append("")

    # --- Error tail ---
    lines.append("## Error Tail")
    lines.append("")
    tails = [m["error_tail"] for m in all_metrics if m.get("error_tail")]
    if tails:
        lines.append("| Metric | Mean | Median |")
        lines.append("|---|---|---|")
        for key in ["abs_rel_p90", "abs_rel_p95"]:
            vals = [t[key] for t in tails if not math.isnan(t.get(key, float("nan")))]
            if vals:
                lines.append(f"| {key} | {np.mean(vals):.4f} | {np.median(vals):.4f} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _metrics_table_md(metrics_list: list[dict]) -> str:
    """Return a markdown table string for a list of metric dicts."""
    if not metrics_list:
        return "*(no data)*"
    keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    arr = np.array([[m.get(k, float("nan")) for k in keys] for m in metrics_list])
    mean = np.nanmean(arr, axis=0)

    header = "| " + " | ".join(keys) + " |"
    sep = "|" + "|".join(["---"] * len(keys)) + "|"
    vals = "| " + " | ".join(f"{v:.4f}" for v in mean) + " |"
    return "\n".join([header, sep, vals])


# ---------------------------------------------------------------------------
# Dataset / loader helpers (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize depth-backend attributes so legacy callers / test stubs that
    # predate the depth_backend option still default to af_sfm.
    depth_backend = getattr(args, "depth_backend", "af_sfm") or "af_sfm"
    depth_anything_repo = getattr(args, "depth_anything_repo", None)

    if getattr(args, "load_specular_weights", False) and depth_backend == "depth_anything":
        raise ValueError(
            "depth_backend='depth_anything' is incompatible with "
            "--load_specular_weights (DA-v2 does not expose a specular head)."
        )

    # ---- Load model ------------------------------------------------------
    model = load_depth_model(
        depth_backend=depth_backend,
        weights_folder=args.weights_folder,
        device=device,
        min_depth=C3VD_MIN_DEPTH,
        max_depth=C3VD_MAX_DEPTH,
        depth_anything_repo=depth_anything_repo,
    )
    # AF-SfM backend returns an (encoder, depth_decoder) tuple; DA-v2 returns
    # a single adapter. Unify so the specular + prediction paths below read
    # cleanly.
    if depth_backend == "af_sfm":
        encoder, depth_decoder = model
    else:
        encoder, depth_decoder = None, None

    # ---- Optional specular model (AF-SfM only) ---------------------------
    specular_decoder = None
    if args.load_specular_weights:
        specular_path = os.path.join(args.weights_folder, "specular.pth")
        if os.path.isfile(specular_path):
            specular_decoder = SpecularDecoder(encoder.num_ch_enc, scales=range(4))
            specular_decoder.load_state_dict(
                torch.load(specular_path, map_location="cpu")
            )
            specular_decoder.eval().to(device)

    # ---- Dataset (used for colour frames and metadata only) --------------
    dataset, loader = _build_loader(args)
    sample_count = _safe_len(dataset)

    # ---- Evaluation loop -------------------------------------------------
    all_metrics = []
    polyp_metrics = []
    stratified_frames: list[dict] = []

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

            pred_depth_t = predict_depth_m(
                model,
                color_curr,
                min_depth=C3VD_MIN_DEPTH,
                max_depth=C3VD_MAX_DEPTH,
                depth_backend=depth_backend,
            )
            pred_depth = pred_depth_t.cpu().numpy().squeeze()

            # Build validity mask
            valid_mask = build_c3vdv2_valid_mask_np(gt_depth)
            n_valid = int(valid_mask.sum())
            if n_valid < 10:
                continue

            metrics = compute_depth_metrics_np(
                gt_depth,
                pred_depth,
                valid_mask,
                median_scaling=args.median_scaling,
            )
            all_metrics.append([metrics[k] for k in DEPTH_METRIC_KEYS])

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
                            [pm[k] for k in DEPTH_METRIC_KEYS]
                        )

            # ---- Stratified per-frame metrics -----------------------------
            if args.stratify:
                color_np = color_curr.cpu().numpy().squeeze()

                frame_entry: dict = {
                    "run_id": args.run_id or "",
                    "seq": seq,
                    "frame_idx": frame_idx,
                    "n_valid": n_valid,
                    "global": metrics,
                    "depth_bands": _depth_bands_np(
                        gt_depth, pred_depth, valid_mask, args.median_scaling
                    ),
                    "error_tail": _error_tail_np(gt_depth, pred_depth, valid_mask),
                }

                # Specular intensity pseudo-mask
                spec_intensity_mask = _specular_intensity_mask_np(color_np)
                frame_entry["specular_intensity"] = _specular_metrics_np(
                    gt_depth, pred_depth, valid_mask, spec_intensity_mask,
                    args.median_scaling,
                )

                # Specular model mask
                if specular_decoder is not None:
                    spec_out = specular_decoder(features)
                    spec_np = spec_out[("specular", 0)].cpu().numpy().squeeze()
                    spec_model_mask = _specular_model_mask_np(spec_np)
                    frame_entry["specular_model"] = _specular_metrics_np(
                        gt_depth, pred_depth, valid_mask, spec_model_mask,
                        args.median_scaling,
                    )
                else:
                    frame_entry["specular_model"] = {"inside": None, "outside": None}

                stratified_frames.append(frame_entry)

                # Top-K tracking
                if args.topk_out_dir:
                    frame_entry["_color_bgr"] = _color_to_bgr(color_np)
                    frame_entry["_pred_vis"] = _depth_vis(pred_depth)
                    frame_entry["_gt_vis"] = _depth_vis(gt_depth)
                    frame_entry["_error_vis"] = _error_heatmap(gt_depth, pred_depth)
                    if specular_decoder is not None or True:  # always save intensity mask
                        frame_entry["_specular_mask"] = spec_intensity_mask

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

    # ---- Stratified outputs ----------------------------------------------
    if args.stratify:
        _write_stratified_outputs(args, stratified_frames)


def _write_stratified_outputs(args, stratified_frames: list[dict]) -> None:
    """Write JSONL, summary MD, and top-K artifacts when --stratify is active."""

    # JSONL
    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as f:
            for frame in stratified_frames:
                # Strip internal buffer keys before serialising
                clean = {
                    k: v for k, v in frame.items()
                    if not k.startswith("_")
                }
                json.dump(clean, f)
                f.write("\n")
        print(f"Metrics JSONL written to {args.metrics_out}")

    # Summary markdown
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        _write_summary_md(args.summary_out, args.run_id or "unknown", stratified_frames)
        print(f"Summary written to {args.summary_out}")

    # Top-K artifacts
    if args.topk_out_dir:
        topk = args.topk or 25
        # Sort by AbsRel descending (worst first)
        sorted_frames = sorted(
            stratified_frames,
            key=lambda f: f.get("global", {}).get("abs_rel", 0.0),
            reverse=True,
        )[:topk]

        topk_entries = []
        for frame in sorted_frames:
            topk_entries.append({
                "seq": frame["seq"],
                "frame_idx": frame.get("frame_idx"),
                "abs_rel": frame.get("global", {}).get("abs_rel"),
                "rmse": frame.get("global", {}).get("rmse"),
                "color_bgr": frame.get("_color_bgr"),
                "pred_vis": frame.get("_pred_vis"),
                "gt_vis": frame.get("_gt_vis"),
                "error_vis": frame.get("_error_vis"),
                "specular_mask": frame.get("_specular_mask"),
            })

        _save_topk_artifacts(
            args.topk_out_dir,
            topk_entries,
            include_specular=True,
        )
        print(f"Top-{topk} artifacts written to {args.topk_out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--weights_folder",
        "--load_weights_folder",
        dest="weights_folder",
        default=None,
        help="Folder of AF-SfM encoder.pth/depth.pth. Required for depth_backend=af_sfm; "
        "ignored for depth_backend=depth_anything (loads from HF).",
    )
    parser.add_argument(
        "--depth_backend",
        choices=["af_sfm", "depth_anything"],
        default="af_sfm",
        help="Depth model family to evaluate. depth_anything loads DA-v2 from HF.",
    )
    parser.add_argument(
        "--depth_anything_repo",
        default=None,
        help="HF repo id for DA-v2 when depth_backend=depth_anything.",
    )
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

    # --- Stratified evaluation flags ---------------------------------------
    parser.add_argument("--stratify", action="store_true",
                        help="Enable per-frame stratified metrics output.")
    parser.add_argument("--run_id", default=None,
                        help="Human-readable run label written into each JSONL row.")
    parser.add_argument("--metrics_out", default=None,
                        help="Path for per-frame JSONL metrics file.")
    parser.add_argument("--summary_out", default=None,
                        help="Path for markdown summary of stratified metrics.")
    parser.add_argument("--topk_out_dir", default=None,
                        help="Directory to write top-K artifact images.")
    parser.add_argument("--topk", type=int, default=25,
                        help="Number of worst frames to save as artifacts (default: 25).")
    parser.add_argument("--specular_mode", choices=["intensity"], default="intensity",
                        help="Specular pseudo-mask mode (default: intensity).")
    parser.add_argument("--load_specular_weights", action="store_true",
                        help="Load specular.pth from weights_folder for model-based specular metrics.")

    args = parser.parse_args()

    if requires_weights_folder(args.depth_backend) and args.weights_folder is None:
        parser.error("--weights_folder is required when depth_backend=af_sfm")
    if args.depth_backend == "depth_anything" and not args.depth_anything_repo:
        parser.error("--depth_anything_repo is required when depth_backend=depth_anything")

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
