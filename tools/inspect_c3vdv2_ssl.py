#!/usr/bin/env python3
"""Render fixed-frame C3VDv2 depth/pose panels across SSL checkpoints.

This is intended for SSL failure analysis: compare the same C3VDv2 frames at
step-0/best/final checkpoints and inspect whether the model loses absolute
scale, relative shape, or pose quality.

Example:

    uv run python tools/inspect_c3vdv2_ssl.py \\
        --data_source hf \\
        --hf_repo_id SmartWhatt/c3vdv2-SfM \\
        --hf_split val \\
        --weights run_a_best=logs/c3vdv2_hf/stage2/models/weights_best \\
        --weights ssl_w0=logs/c3vdv2_hf_ssl/c3vdv2_run_a_ssl_full_500/models/weights_best \\
        --weights ssl_w19=logs/c3vdv2_hf_ssl/c3vdv2_run_a_ssl_full_full/models/weights_19 \\
        --indices 0 50 100 150 200 \\
        --out_dir reports/c3vdv2_ssl/run_a_panels
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networks  # noqa: E402
from colon_datasets.c3vdv2_metrics import (  # noqa: E402
    C3VD_MAX_DEPTH,
    C3VD_MIN_DEPTH,
    build_c3vdv2_valid_mask_np,
    compute_depth_metrics_np,
)
from colon_datasets.hf_triplet_dataset import build_hf_triplet_dataset  # noqa: E402
from layers import transformation_from_parameters  # noqa: E402


def _parse_weight_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).name
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in --weights {spec!r}")
        parsed.append((label, Path(path).expanduser()))
    return parsed


def _load_depth_model(weights: Path, device: torch.device, num_layers: int):
    encoder_path = weights / "encoder.pth"
    depth_path = weights / "depth.pth"
    if not encoder_path.is_file():
        raise FileNotFoundError(f"missing {encoder_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"missing {depth_path}")

    encoder = networks.ResnetEncoder(num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"), strict=False)
    depth_decoder.load_state_dict(torch.load(depth_path, map_location="cpu"))
    encoder.eval().to(device)
    depth_decoder.eval().to(device)
    return encoder, depth_decoder


def _load_pose_model(weights: Path, device: torch.device, num_layers: int):
    pose_encoder_path = weights / "pose_encoder.pth"
    pose_path = weights / "pose.pth"
    if not pose_encoder_path.is_file() or not pose_path.is_file():
        return None

    pose_encoder = networks.ResnetEncoder(num_layers, False, 2)
    pose = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location="cpu"))
    pose.load_state_dict(torch.load(pose_path, map_location="cpu"))
    pose_encoder.eval().to(device)
    pose.eval().to(device)
    return pose_encoder, pose


def _predict_depth(encoder, depth_decoder, color: torch.Tensor) -> np.ndarray:
    features = encoder(color)
    outputs = depth_decoder(features)
    disp = outputs[("disp", 0)].detach().cpu().numpy().squeeze()
    return 1.0 / (
        disp * (1.0 / C3VD_MIN_DEPTH - 1.0 / C3VD_MAX_DEPTH)
        + 1.0 / C3VD_MAX_DEPTH
    )


def _to_vec3(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)[:, :3]


def _predict_pose(pose_model, source: torch.Tensor, curr: torch.Tensor):
    pose_encoder, pose = pose_model
    features = [pose_encoder(torch.cat([source, curr], dim=1))]
    axisangle, translation = pose(features)
    return _to_vec3(axisangle), _to_vec3(translation)


def _rotation_error_deg(pred_axang: torch.Tensor, gt_axang: torch.Tensor) -> float:
    zeros = torch.zeros_like(pred_axang)
    pred_T = transformation_from_parameters(pred_axang.unsqueeze(1), zeros.unsqueeze(1))
    gt_T = transformation_from_parameters(gt_axang.unsqueeze(1), zeros.unsqueeze(1))
    rel_R = torch.matmul(pred_T[:, :3, :3], gt_T[:, :3, :3].transpose(1, 2))
    trace = rel_R[:, 0, 0] + rel_R[:, 1, 1] + rel_R[:, 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos_theta))[0].detach().cpu())


def _pose_row(
    label: str,
    sample_meta: dict,
    pair_label: str,
    pred_ax: torch.Tensor,
    pred_tr: torch.Tensor,
    gt_ax: torch.Tensor,
    gt_tr: torch.Tensor,
) -> dict:
    pred_tr_norm = torch.linalg.norm(pred_tr, dim=1)
    gt_tr_norm = torch.linalg.norm(gt_tr, dim=1)
    trans_l2 = torch.linalg.norm(pred_tr - gt_tr, dim=1)
    return {
        **sample_meta,
        "label": label,
        "pair": pair_label,
        "rot_error_deg": _rotation_error_deg(pred_ax, gt_ax),
        "trans_l2_mm": float((trans_l2 * 1000.0)[0].detach().cpu()),
        "pred_trans_norm_mm": float((pred_tr_norm * 1000.0)[0].detach().cpu()),
        "gt_trans_norm_mm": float((gt_tr_norm * 1000.0)[0].detach().cpu()),
        "pred_axang_x": float(pred_ax[0, 0].detach().cpu()),
        "pred_axang_y": float(pred_ax[0, 1].detach().cpu()),
        "pred_axang_z": float(pred_ax[0, 2].detach().cpu()),
        "pred_trans_x_m": float(pred_tr[0, 0].detach().cpu()),
        "pred_trans_y_m": float(pred_tr[0, 1].detach().cpu()),
        "pred_trans_z_m": float(pred_tr[0, 2].detach().cpu()),
    }


def _median_scaled_pred(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-7,
) -> tuple[np.ndarray, float]:
    gt = gt_depth[valid_mask]
    pred = pred_depth[valid_mask]
    if gt.size == 0:
        return pred_depth, 1.0
    median_pred = float(np.median(pred))
    if median_pred <= eps:
        return pred_depth, 1.0
    scale = float(np.median(gt)) / median_pred
    return pred_depth * scale, scale


def _depth_to_bgr(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    clipped = np.clip(depth, vmin, vmax)
    norm = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _error_to_bgr(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    vmax: float,
) -> np.ndarray:
    error = np.abs(gt_depth - pred_depth)
    error = np.where(valid_mask, error, 0.0)
    norm = (np.clip(error, 0.0, vmax) / vmax * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def _rgb_to_bgr(rgb_chw: torch.Tensor) -> np.ndarray:
    rgb = rgb_chw.detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _label_tile(tile: np.ndarray, text: str, width: int) -> np.ndarray:
    tile = cv2.resize(tile, (width, width), interpolation=cv2.INTER_AREA)
    header = np.zeros((42, width, 3), dtype=np.uint8)
    cv2.putText(
        header,
        text[:42],
        (6, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, tile], axis=0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_indices(args, dataset) -> list[int]:
    if args.indices:
        return args.indices
    try:
        n = len(dataset)
    except TypeError:
        n = args.num_samples * args.stride
    return [idx for idx in range(0, min(n, args.num_samples * args.stride), args.stride)][
        : args.num_samples
    ]


def inspect(args) -> None:
    if args.data_source != "hf":
        raise ValueError("Only --data_source hf is supported by this inspection tool")
    if not args.hf_repo_id:
        raise ValueError("--hf_repo_id is required")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_specs = _parse_weight_specs(args.weights)
    models = []
    for label, weights in weight_specs:
        depth_model = _load_depth_model(weights, device, args.num_layers)
        pose_model = _load_pose_model(weights, device, args.num_layers)
        models.append((label, weights, depth_model, pose_model))

    dataset = build_hf_triplet_dataset(
        repo_id=args.hf_repo_id,
        split=args.hf_split,
        streaming=False,
        cache_dir=args.hf_cache_dir,
        target_size=(args.height, args.width),
        is_train=False,
        output_format="c3vdv2",
        scales=(0,),
        shuffle_buffer=0,
        seed=args.seed,
    )
    indices = _selected_indices(args, dataset)

    depth_rows: list[dict] = []
    pose_rows: list[dict] = []
    panel_index = []

    with torch.no_grad():
        for panel_rank, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            curr = sample["color_curr"].unsqueeze(0).to(device)
            prev = sample["color_prev"].unsqueeze(0).to(device)
            nxt = sample["color_next"].unsqueeze(0).to(device)
            gt_depth = sample["gt_depth"].squeeze(0).numpy()
            occlusion = sample.get("occlusion")
            occ_np = None if occlusion is None else occlusion.squeeze(0).numpy()
            valid_mask = build_c3vdv2_valid_mask_np(gt_depth, occ_np)
            seq = str(sample.get("seq", ""))
            frame_idx = int(sample.get("frame_idx", -1))
            sample_meta = {
                "sample_index": int(sample_idx),
                "seq": seq,
                "frame_idx": frame_idx,
            }

            tiles = [
                _label_tile(_rgb_to_bgr(sample["color_curr"]), "RGB", args.tile_size),
                _label_tile(
                    _depth_to_bgr(gt_depth, args.depth_vis_min, args.depth_vis_max),
                    "GT depth",
                    args.tile_size,
                ),
            ]

            for label, weights, depth_model, pose_model in models:
                encoder, depth_decoder = depth_model
                pred_depth = _predict_depth(encoder, depth_decoder, curr)
                scaled_pred, median_scale = _median_scaled_pred(
                    gt_depth, pred_depth, valid_mask
                )
                raw_metrics = compute_depth_metrics_np(
                    gt_depth, pred_depth, valid_mask, median_scaling=False
                )
                median_metrics = compute_depth_metrics_np(
                    gt_depth, pred_depth, valid_mask, median_scaling=True
                )
                pred_valid = pred_depth[valid_mask]
                row = {
                    **sample_meta,
                    "label": label,
                    "weights_folder": str(weights),
                    "raw_abs_rel": raw_metrics["abs_rel"],
                    "raw_rmse_mm": raw_metrics["rmse"] * 1000.0,
                    "raw_a1": raw_metrics["a1"],
                    "median_abs_rel": median_metrics["abs_rel"],
                    "median_rmse_mm": median_metrics["rmse"] * 1000.0,
                    "median_a1": median_metrics["a1"],
                    "median_scale": median_scale,
                    "pred_min_m": float(np.min(pred_valid)) if pred_valid.size else np.nan,
                    "pred_median_m": float(np.median(pred_valid)) if pred_valid.size else np.nan,
                    "pred_max_m": float(np.max(pred_valid)) if pred_valid.size else np.nan,
                }
                depth_rows.append(row)

                tiles.extend(
                    [
                        _label_tile(
                            _depth_to_bgr(
                                pred_depth,
                                args.depth_vis_min,
                                args.depth_vis_max,
                            ),
                            f"{label} raw {row['raw_rmse_mm']:.1f}mm",
                            args.tile_size,
                        ),
                        _label_tile(
                            _error_to_bgr(
                                gt_depth,
                                pred_depth,
                                valid_mask,
                                args.error_vis_max,
                            ),
                            f"{label} raw err",
                            args.tile_size,
                        ),
                        _label_tile(
                            _depth_to_bgr(
                                scaled_pred,
                                args.depth_vis_min,
                                args.depth_vis_max,
                            ),
                            f"{label} med {row['median_rmse_mm']:.1f}mm",
                            args.tile_size,
                        ),
                        _label_tile(
                            _error_to_bgr(
                                gt_depth,
                                scaled_pred,
                                valid_mask,
                                args.error_vis_max,
                            ),
                            f"{label} med err",
                            args.tile_size,
                        ),
                    ]
                )

                if pose_model is not None and "axang_curr2prev" in sample:
                    gt_prev_ax = sample["axang_curr2prev"].float().unsqueeze(0).to(device)
                    gt_prev_tr = sample["trans_curr2prev"].float().unsqueeze(0).to(device)
                    gt_next_ax = sample["axang_curr2next"].float().unsqueeze(0).to(device)
                    gt_next_tr = sample["trans_curr2next"].float().unsqueeze(0).to(device)
                    pred_prev_ax, pred_prev_tr = _predict_pose(pose_model, prev, curr)
                    pred_next_ax, pred_next_tr = _predict_pose(pose_model, nxt, curr)
                    pose_rows.append(
                        _pose_row(
                            label,
                            sample_meta,
                            "prev",
                            pred_prev_ax,
                            pred_prev_tr,
                            gt_prev_ax,
                            gt_prev_tr,
                        )
                    )
                    pose_rows.append(
                        _pose_row(
                            label,
                            sample_meta,
                            "next",
                            pred_next_ax,
                            pred_next_tr,
                            gt_next_ax,
                            gt_next_tr,
                        )
                    )

            panel = np.concatenate(tiles, axis=1)
            stem = f"{panel_rank:03d}_idx{sample_idx}_{seq}_{frame_idx}".replace(
                "/", "_"
            )
            panel_path = out_dir / f"{stem}_panel.png"
            cv2.imwrite(str(panel_path), panel)
            panel_index.append({**sample_meta, "panel": panel_path.name})

    _write_csv(out_dir / "depth_metrics.csv", depth_rows)
    _write_csv(out_dir / "pose_metrics.csv", pose_rows)
    with (out_dir / "index.json").open("w") as f:
        json.dump(
            {
                "hf_repo_id": args.hf_repo_id,
                "hf_split": args.hf_split,
                "indices": indices,
                "weights": [
                    {"label": label, "weights_folder": str(weights)}
                    for label, weights in weight_specs
                ],
                "panels": panel_index,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"Wrote {len(panel_index)} panels to {out_dir}")
    print(f"Depth CSV: {out_dir / 'depth_metrics.csv'}")
    if pose_rows:
        print(f"Pose CSV:  {out_dir / 'pose_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render fixed-frame C3VDv2 SSL depth/pose panels."
    )
    parser.add_argument("--data_source", choices=["hf"], default="hf")
    parser.add_argument("--hf_repo_id", required=True)
    parser.add_argument("--hf_split", default="val")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--weights", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--depth_vis_min", type=float, default=0.001)
    parser.add_argument("--depth_vis_max", type=float, default=0.100)
    parser.add_argument("--error_vis_max", type=float, default=0.050)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    inspect(parser.parse_args())


if __name__ == "__main__":
    main()
