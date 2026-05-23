#!/usr/bin/env python3
"""Evaluate AF-SfMLearner pose predictions on C3VDv2 HF triplets.

This evaluates the trained ``pose_encoder`` and ``pose`` modules against the
relative pose labels packed in the C3VDv2 Hugging Face triplet dataset. It uses
the same pose input ordering as the trainer:

    f_i = -1: concat(prev, curr) -> gt curr2prev
    f_i =  1: concat(next, curr) -> gt curr2next

Example:

    uv run python tools/evaluate_c3vdv2_pose.py \
        --data_source hf \
        --hf_repo_id SmartWhatt/c3vdv2-SfM \
        --hf_split test \
        --weights_folder logs/c3vdv2_hf/stage2_specular_pps/models/weights_best
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networks  # noqa: E402
from colon_datasets.hf_triplet_dataset import build_hf_triplet_dataset  # noqa: E402
from layers import transformation_from_parameters  # noqa: E402


def _safe_len(dataset) -> int | str:
    try:
        return len(dataset)
    except TypeError:
        return "unknown"


def _to_vec3(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)[:, :3]


def _rotation_error_deg(pred_axang: torch.Tensor, gt_axang: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(pred_axang)
    pred_T = transformation_from_parameters(pred_axang, zeros)
    gt_T = transformation_from_parameters(gt_axang, zeros)
    rel_R = torch.matmul(pred_T[:, :3, :3], gt_T[:, :3, :3].transpose(1, 2))
    trace = rel_R[:, 0, 0] + rel_R[:, 1, 1] + rel_R[:, 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _translation_metrics(
    pred_trans: torch.Tensor,
    gt_trans: torch.Tensor,
    eps: float = 1e-9,
) -> dict[str, torch.Tensor]:
    diff = pred_trans - gt_trans
    pred_norm = torch.linalg.norm(pred_trans, dim=1)
    gt_norm = torch.linalg.norm(gt_trans, dim=1)
    l2 = torch.linalg.norm(diff, dim=1)
    cosine = torch.sum(pred_trans * gt_trans, dim=1) / (
        pred_norm.clamp(min=eps) * gt_norm.clamp(min=eps)
    )
    mag_ratio = pred_norm / gt_norm.clamp(min=eps)
    return {
        "trans_l2_m": l2,
        "trans_l2_mm": l2 * 1000.0,
        "trans_cosine": cosine.clamp(-1.0, 1.0),
        "trans_mag_ratio": mag_ratio,
        "pred_trans_norm_m": pred_norm,
        "gt_trans_norm_m": gt_norm,
    }


def _batch_metadata(batch: dict, batch_size: int, item: int) -> tuple[str, int | None]:
    seq = ""
    frame_idx = None
    if "seq" in batch:
        seq_value = batch["seq"]
        if isinstance(seq_value, (list, tuple)):
            seq = str(seq_value[item])
        else:
            seq = str(seq_value)
    if "frame_idx" in batch:
        idx_value = batch["frame_idx"]
        if hasattr(idx_value, "detach"):
            frame_idx = int(idx_value.reshape(batch_size)[item].item())
        elif isinstance(idx_value, (list, tuple)):
            frame_idx = int(idx_value[item])
        else:
            frame_idx = int(idx_value)
    return seq, frame_idx


def _build_loader(args):
    if args.data_source != "hf":
        raise ValueError("evaluate_c3vdv2_pose.py currently supports --data_source hf")
    if not args.hf_repo_id:
        raise ValueError("--hf_repo_id is required when --data_source hf")

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
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return dataset, loader


def _load_pose_model(args, device: torch.device):
    weights = Path(args.weights_folder)
    pose_encoder_path = weights / "pose_encoder.pth"
    pose_path = weights / "pose.pth"
    if not pose_encoder_path.is_file():
        raise FileNotFoundError(f"missing pose_encoder.pth under {weights}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"missing pose.pth under {weights}")

    pose_encoder = networks.ResnetEncoder(args.num_layers, False, 2)
    pose = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)

    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location="cpu"))
    pose.load_state_dict(torch.load(pose_path, map_location="cpu"))

    pose_encoder.eval().to(device)
    pose.eval().to(device)
    return pose_encoder, pose


def _predict_pair(pose_encoder, pose, source: torch.Tensor, curr: torch.Tensor):
    features = [pose_encoder(torch.cat([source, curr], dim=1))]
    axisangle, translation = pose(features)
    return _to_vec3(axisangle), _to_vec3(translation)


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _write_outputs(args, rows: list[dict], summary: dict) -> None:
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for row in rows:
                json.dump(row, f, sort_keys=True)
                f.write("\n")
        print(f"Metrics JSONL written to {out}")

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Summary JSON written to {out}")


def evaluate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pose_encoder, pose = _load_pose_model(args, device)
    dataset, loader = _build_loader(args)
    sample_count = _safe_len(dataset)

    rows: list[dict] = []
    aggregates: dict[str, list[float]] = defaultdict(list)
    pair_counts: dict[str, int] = defaultdict(int)
    sample_seen = 0

    pair_specs = [
        ("prev", "color_prev", "axang_curr2prev", "trans_curr2prev"),
        ("next", "color_next", "axang_curr2next", "trans_curr2next"),
    ]

    with torch.no_grad():
        for batch in loader:
            if args.max_batches is not None and sample_seen >= args.max_batches * args.batch_size:
                break
            curr = batch["color_curr"].to(device)
            batch_size = curr.shape[0]
            sample_seen += batch_size

            for label, color_key, gt_ax_key, gt_tr_key in pair_specs:
                if gt_ax_key not in batch or gt_tr_key not in batch:
                    continue
                source = batch[color_key].to(device)
                gt_ax = _to_vec3(batch[gt_ax_key].to(device).float())
                gt_tr = _to_vec3(batch[gt_tr_key].to(device).float())
                pred_ax, pred_tr = _predict_pair(pose_encoder, pose, source, curr)

                rot_deg = _rotation_error_deg(pred_ax, gt_ax)
                trans = _translation_metrics(pred_tr, gt_tr)
                ax_l2 = torch.linalg.norm(pred_ax - gt_ax, dim=1)

                for i in range(batch_size):
                    seq, frame_idx = _batch_metadata(batch, batch_size, i)
                    row = {
                        "pair": label,
                        "seq": seq,
                        "frame_idx": frame_idx,
                        "rot_error_deg": float(rot_deg[i].cpu()),
                        "axisangle_l2": float(ax_l2[i].cpu()),
                        "trans_l2_m": float(trans["trans_l2_m"][i].cpu()),
                        "trans_l2_mm": float(trans["trans_l2_mm"][i].cpu()),
                        "trans_cosine": float(trans["trans_cosine"][i].cpu()),
                        "trans_mag_ratio": float(trans["trans_mag_ratio"][i].cpu()),
                        "pred_trans_norm_m": float(trans["pred_trans_norm_m"][i].cpu()),
                        "gt_trans_norm_m": float(trans["gt_trans_norm_m"][i].cpu()),
                    }
                    rows.append(row)
                    pair_counts[label] += 1
                    for key in (
                        "rot_error_deg",
                        "axisangle_l2",
                        "trans_l2_m",
                        "trans_l2_mm",
                        "trans_cosine",
                        "trans_mag_ratio",
                    ):
                        aggregates[f"{label}/{key}"].append(row[key])
                        aggregates[f"all/{key}"].append(row[key])

    summary = {
        "hf_repo_id": args.hf_repo_id,
        "hf_split": args.hf_split,
        "weights_folder": args.weights_folder,
        "samples_in_split": sample_count,
        "pose_pairs_evaluated": len(rows),
        "pair_counts": dict(pair_counts),
        "metrics": {key: _summarize(vals) for key, vals in aggregates.items()},
    }

    print("========== C3VDv2 Pose Evaluation ==========")
    print(f"  Samples in split:    {sample_count}")
    print(f"  Pose pairs evaluated:{len(rows)}")
    for prefix in ("all", "prev", "next"):
        if not aggregates.get(f"{prefix}/rot_error_deg"):
            continue
        rot = _summarize(aggregates[f"{prefix}/rot_error_deg"])
        tr = _summarize(aggregates[f"{prefix}/trans_l2_mm"])
        cos = _summarize(aggregates[f"{prefix}/trans_cosine"])
        ratio = _summarize(aggregates[f"{prefix}/trans_mag_ratio"])
        print(f"--- {prefix} ---")
        print(
            f"  Rot error:       mean {rot['mean']:.4f} deg | "
            f"median {rot['median']:.4f} | p90 {rot['p90']:.4f}"
        )
        print(
            f"  Trans L2:        mean {tr['mean']:.4f} mm | "
            f"median {tr['median']:.4f} | p90 {tr['p90']:.4f}"
        )
        print(
            f"  Trans cosine:    mean {cos['mean']:.4f} | "
            f"median {cos['median']:.4f}"
        )
        print(
            f"  Trans mag ratio: median {ratio['median']:.4f} | "
            f"p90 {ratio['p90']:.4f}"
        )
    print("============================================\n")

    _write_outputs(args, rows, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C3VDv2 relative pose.")
    parser.add_argument("--data_source", choices=["hf"], default="hf")
    parser.add_argument("--hf_repo_id", required=True)
    parser.add_argument("--hf_split", default="test")
    parser.add_argument("--hf_streaming", action="store_true")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--weights_folder", "--load_weights_folder", dest="weights_folder", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument("--summary_out", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
