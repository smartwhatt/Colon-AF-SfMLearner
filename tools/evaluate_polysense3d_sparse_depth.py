#!/usr/bin/env python3
"""Evaluate AF-SfMLearner checkpoints on PolypSense3D clinical sparse depth.

The clinical subset provides sparse forceps-derived depth labels rather than
dense depth maps. This evaluator follows the public PolypSense3D convention:
sample predictions at LabelMe points labelled ``value`` and compare them to the
per-image ``depth_gt.xlsx`` value in millimetres.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colon_datasets.polysense3d_sparse_dataset import (  # noqa: E402
    PolypSense3DSparseDepthDataset,
    sample_depth_at_points,
)
from tools.depth_backend import (  # noqa: E402
    load_depth_model,
    predict_depth_m,
    requires_weights_folder,
)


POINT_METRIC_KEYS = ("mae_mm", "rmse_mm", "abs_rel", "log10", "a1", "a2")


def _load_model(
    weights_folder,
    device: torch.device,
    *,
    depth_backend: str = "af_sfm",
    depth_anything_repo: str = None,
    min_depth: float = 0.001,
    max_depth: float = 0.1,
):
    return load_depth_model(
        depth_backend=depth_backend,
        weights_folder=weights_folder,
        device=device,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_anything_repo=depth_anything_repo,
    )


def _predict_depth_m(
    model,
    color: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    depth_backend: str = "af_sfm",
) -> np.ndarray:
    depth = predict_depth_m(
        model,
        color,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_backend=depth_backend,
    )
    return depth.squeeze(0).squeeze(0).detach().cpu().numpy()


def _metrics_from_arrays(pred_mm: Iterable[float], truth_mm: Iterable[float]) -> dict[str, float]:
    pred = np.asarray(list(pred_mm), dtype=np.float64)
    truth = np.asarray(list(truth_mm), dtype=np.float64)
    if pred.size == 0:
        return {
            "count": 0.0,
            "pred_mean_mm": float("nan"),
            "truth_mean_mm": float("nan"),
            **{key: float("nan") for key in POINT_METRIC_KEYS},
        }

    eps = 1e-8
    pred = np.maximum(pred, eps)
    truth = np.maximum(truth, eps)
    err = pred - truth
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "count": float(pred.size),
        "pred_mean_mm": float(np.mean(pred)),
        "truth_mean_mm": float(np.mean(truth)),
        "mae_mm": float(np.mean(np.abs(err))),
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "abs_rel": float(np.mean(np.abs(err) / truth)),
        "log10": float(np.mean(np.abs(np.log10(pred) - np.log10(truth)))),
        "a1": float(np.mean(ratio < 1.25)),
        "a2": float(np.mean(ratio < 1.25**2)),
    }


def _median_scaled_metrics_from_arrays(
    pred_mm: Iterable[float],
    truth_mm: Iterable[float],
) -> tuple[float, dict[str, float]]:
    pred = np.asarray(list(pred_mm), dtype=np.float64)
    truth = np.asarray(list(truth_mm), dtype=np.float64)
    if pred.size == 0:
        return float("nan"), _metrics_from_arrays([], [])
    eps = 1e-8
    scale = float(np.median(np.maximum(truth, eps) / np.maximum(pred, eps)))
    return scale, _metrics_from_arrays(pred * scale, truth)


def _floatify(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, float) and math.isnan(value):
            out[key] = value
        else:
            out[key] = value
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            json.dump(_floatify(row), f, sort_keys=True)
            f.write("\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_floatify(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def _rgb_tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _depth_to_bgr(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-8)
    norm = np.clip((depth_m - vmin) / denom, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _draw_points(
    image_bgr: np.ndarray,
    value_points: np.ndarray,
    top_points: np.ndarray,
) -> np.ndarray:
    out = image_bgr.copy()
    for x_f, y_f in top_points:
        cv2.circle(out, (int(round(x_f)), int(round(y_f))), 4, (255, 255, 255), -1)
        cv2.circle(out, (int(round(x_f)), int(round(y_f))), 5, (0, 0, 0), 1)
    for idx, (x_f, y_f) in enumerate(value_points, start=1):
        point = (int(round(x_f)), int(round(y_f)))
        cv2.circle(out, point, 4, (0, 255, 255), -1)
        cv2.putText(
            out,
            str(idx),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(idx),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _labeled_tile(image_bgr: np.ndarray, label: str) -> np.ndarray:
    tile = image_bgr.copy()
    bar_h = 28
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], bar_h), (0, 0, 0), -1)
    tile = cv2.addWeighted(overlay, 0.58, tile, 0.42, 0)
    cv2.putText(
        tile,
        label[:96],
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _save_topk_artifacts(
    out_dir: Path,
    frame_rows: list[dict],
    frame_buffers: dict[str, dict],
    *,
    topk: int,
    depth_vis_min: float,
    depth_vis_max: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(frame_rows, key=lambda row: row["mae_mm"], reverse=True)[:topk]
    index = []
    for rank, row in enumerate(worst, start=1):
        sample_id = row["sample_id"]
        buffers = frame_buffers[sample_id]
        rgb_points = _draw_points(
            buffers["rgb_bgr"],
            buffers["value_points"],
            buffers["top_points"],
        )
        depth_points = _draw_points(
            _depth_to_bgr(buffers["pred_depth_m"], depth_vis_min, depth_vis_max),
            buffers["value_points"],
            buffers["top_points"],
        )
        label = (
            f"{sample_id} gt={row['gt_depth_mm']:.2f}mm "
            f"pred={row['pred_mean_mm']:.2f}mm mae={row['mae_mm']:.2f}mm"
        )
        panel = np.concatenate(
            [
                _labeled_tile(rgb_points, "rgb sparse points"),
                _labeled_tile(depth_points, label),
            ],
            axis=1,
        )
        filename = f"rank{rank:03d}_{sample_id}_sparse_depth.png"
        cv2.imwrite(str(out_dir / filename), panel)
        index.append(
            {
                "rank": rank,
                "sample_id": sample_id,
                "group_id": row["group_id"],
                "gt_depth_mm": row["gt_depth_mm"],
                "pred_mean_mm": row["pred_mean_mm"],
                "mae_mm": row["mae_mm"],
                "artifact": filename,
            }
        )

    with (out_dir / "topk_index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def _paper_depth_all_rows(frame_rows: list[dict], *, paper_num_points: int) -> list[dict]:
    out = []
    for row in frame_rows:
        paper_values = [row[f"pred{i}_mm"] for i in range(1, paper_num_points + 1)]
        out.append(
            {
                "id": row["sample_id"],
                **{f"pred{i}": value for i, value in enumerate(paper_values, start=1)},
                "gt": row["gt_depth_mm"],
                "MAE": row["paper_mae_mm"],
                "abs.REL": row["paper_abs_rel"],
                "RMSE": row["paper_rmse_mm"],
            }
        )
    return out


def _official_group_rows(frame_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in frame_rows:
        groups[row["group_id"]].append(row)

    out = []
    for group_id in sorted(groups):
        rows = sorted(groups[group_id], key=lambda row: row["sample_id"])
        selected = (
            sorted(rows, key=lambda row: row["paper_mae_mm"])[:2]
            if len(rows) > 2
            else rows
        )
        pred = float(np.mean([row["paper_pred_mean_mm"] for row in selected]))
        truth = float(np.mean([row["gt_depth_mm"] for row in selected]))
        metrics = _metrics_from_arrays([pred], [truth])
        out.append(
            {
                "group_id": group_id,
                "group_size": len(rows),
                "selected_count": len(selected),
                "selected_sample_ids": ",".join(row["sample_id"] for row in selected),
                "pred_mm": pred,
                "truth_mm": truth,
                "mae_mm": metrics["mae_mm"],
                "rmse_mm": metrics["rmse_mm"],
                "abs_rel": metrics["abs_rel"],
                "log10": metrics["log10"],
                "a1": metrics["a1"],
                "a2": metrics["a2"],
            }
        )
    return out


def _paper_eval_metrics(group_rows: list[dict]) -> dict[str, float]:
    metrics = _metrics_from_arrays(
        [row["pred_mm"] for row in group_rows],
        [row["truth_mm"] for row in group_rows],
    )
    return {
        "count": metrics["count"],
        "rmse": metrics["rmse_mm"],
        "mae": metrics["mae_mm"],
        "absrel": metrics["abs_rel"],
        "log10": metrics["log10"],
        "a1_percent": metrics["a1"] * 100.0,
        "a2_percent": metrics["a2"] * 100.0,
        "pred_mean_mm": metrics["pred_mean_mm"],
        "truth_mean_mm": metrics["truth_mean_mm"],
    }


def _median_scaled_group_rows(group_rows: list[dict]) -> tuple[float, list[dict]]:
    pred = np.asarray([row["pred_mm"] for row in group_rows], dtype=np.float64)
    truth = np.asarray([row["truth_mm"] for row in group_rows], dtype=np.float64)
    scale = float(np.median(np.maximum(truth, 1e-8) / np.maximum(pred, 1e-8)))
    rows = []
    for row in group_rows:
        scaled_pred = float(row["pred_mm"] * scale)
        metrics = _metrics_from_arrays([scaled_pred], [row["truth_mm"]])
        scaled_row = dict(row)
        scaled_row["raw_pred_mm"] = row["pred_mm"]
        scaled_row["pred_mm"] = scaled_pred
        scaled_row["median_scale"] = scale
        scaled_row["mae_mm"] = metrics["mae_mm"]
        scaled_row["rmse_mm"] = metrics["rmse_mm"]
        scaled_row["abs_rel"] = metrics["abs_rel"]
        scaled_row["log10"] = metrics["log10"]
        scaled_row["a1"] = metrics["a1"]
        scaled_row["a2"] = metrics["a2"]
        rows.append(scaled_row)
    return scale, rows


def _write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        f"# {summary['run_id']} PolypSense3D Sparse Depth Summary",
        "",
        "## Dataset",
        "",
        f"- Frames: {summary['dataset']['frames']}",
        f"- Sparse points: {summary['dataset']['points']}",
        f"- Groups: {summary['dataset']['groups']}",
        "",
        "## Metrics",
        "",
        "| Scope | Count | MAE mm | RMSE mm | AbsRel | log10 | a1 | a2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("All points", "all_points"),
        ("Frame means", "frame_means"),
        ("Official groups", "official_groups"),
    ]:
        metrics = summary["metrics"][key]
        lines.append(
            "| {label} | {count:.0f} | {mae_mm:.4f} | {rmse_mm:.4f} | "
            "{abs_rel:.4f} | {log10:.4f} | {a1:.4f} | {a2:.4f} |".format(
                label=label,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Per-point JSONL: `{summary['outputs']['per_point_jsonl']}`",
            f"- Per-frame JSONL: `{summary['outputs']['per_frame_jsonl']}`",
            f"- Official group CSV: `{summary['outputs']['official_group_csv']}`",
            f"- Paper-style depth_all CSV: `{summary['outputs']['paper_depth_all_csv']}`",
            f"- Paper-style depth_summary CSV: `{summary['outputs']['paper_depth_summary_csv']}`",
            f"- Paper-style eval JSON: `{summary['outputs']['paper_depth_eval_json']}`",
        ]
    )
    paper = summary["polysense3d_paper_style"]
    lines.extend(
        [
            "",
            "## PolypSense3D Paper-Style Result",
            "",
            "Raw metric depth, no median scaling.",
            "",
            "| Count | MAE mm | RMSE mm | AbsRel | log10 | a1 % | a2 % |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            "| {count:.0f} | {mae:.4f} | {rmse:.4f} | {absrel:.4f} | "
            "{log10:.4f} | {a1_percent:.4f} | {a2_percent:.4f} |".format(
                **paper["raw_metrics"],
            ),
        ]
    )
    if "median_scaled_metrics" in paper:
        lines.extend(
            [
                "",
                "## Median-Scaled Diagnostic",
                "",
                f"- Global official-group median scale: {paper['median_scale']:.6f}",
                "",
                "| Count | MAE mm | RMSE mm | AbsRel | log10 | a1 % | a2 % |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                "| {count:.0f} | {mae:.4f} | {rmse:.4f} | {absrel:.4f} | "
                "{log10:.4f} | {a1_percent:.4f} | {a2_percent:.4f} |".format(
                    **paper["median_scaled_metrics"],
                ),
            ]
        )
    if summary["dataset"]["ignored_labels"]:
        lines.extend(["", "## Ignored Labels", ""])
        for label, count in sorted(summary["dataset"]["ignored_labels"].items()):
            lines.append(f"- `{label}`: {count}")
    path.write_text("\n".join(lines) + "\n")


def evaluate(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize depth-backend attributes so legacy callers / test stubs that
    # predate the depth_backend option still default to af_sfm.
    depth_backend = getattr(args, "depth_backend", "af_sfm") or "af_sfm"
    depth_anything_repo = getattr(args, "depth_anything_repo", None)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    weights_folder = Path(args.weights_folder).expanduser() if args.weights_folder else None
    if requires_weights_folder(depth_backend):
        if weights_folder is None:
            raise ValueError("--weights_folder is required for depth_backend=af_sfm")
        for filename in ("encoder.pth", "depth.pth"):
            if not (weights_folder / filename).is_file():
                raise FileNotFoundError(f"Missing {filename} under {weights_folder}")

    dataset = PolypSense3DSparseDepthDataset(
        args.data_root,
        target_size=(args.height, args.width),
        strict=not args.allow_manifest_errors,
    )
    model = _load_model(
        weights_folder,
        device,
        depth_backend=depth_backend,
        depth_anything_repo=depth_anything_repo,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    point_rows: list[dict] = []
    frame_rows: list[dict] = []
    frame_buffers: dict[str, dict] = {}
    ignored_labels = Counter()

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            color = sample["color_curr"].unsqueeze(0).to(device)
            pred_depth_m = _predict_depth_m(
                model,
                color,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                depth_backend=depth_backend,
            )
            value_points = sample["value_points"].numpy()
            top_points = sample["top_points"].numpy()
            pred_values_m = sample_depth_at_points(
                pred_depth_m,
                value_points,
                radius_px=args.sample_radius_px,
            )
            pred_values_mm = pred_values_m.astype(np.float64) * 1000.0
            if len(pred_values_mm) < args.paper_num_points:
                raise ValueError(
                    f"{sample['sample_id']} has {len(pred_values_mm)} value point(s), "
                    f"but paper-style depth evaluation needs {args.paper_num_points}."
                )
            gt_mm = float(sample["gt_depth_mm"])
            truth_values_mm = np.full_like(pred_values_mm, gt_mm, dtype=np.float64)

            point_metrics = _metrics_from_arrays(pred_values_mm, truth_values_mm)
            paper_pred_values_mm = pred_values_mm[: args.paper_num_points]
            paper_truth_values_mm = np.full_like(
                paper_pred_values_mm,
                gt_mm,
                dtype=np.float64,
            )
            paper_point_metrics = _metrics_from_arrays(
                paper_pred_values_mm,
                paper_truth_values_mm,
            )
            paper_pred_mean_mm = float(np.mean(paper_pred_values_mm))
            pred_mean_mm = float(np.mean(pred_values_mm))
            pred_median_mm = float(np.median(pred_values_mm))
            frame_metric_on_mean = _metrics_from_arrays([pred_mean_mm], [gt_mm])
            sample_id = str(sample["sample_id"])
            group_id = str(sample["group_id"])

            native_points = sample["value_points_native"].numpy()
            for point_idx, (pred_mm, point_model, point_native) in enumerate(
                zip(pred_values_mm, value_points, native_points),
                start=1,
            ):
                point_rows.append(
                    {
                        "run_id": args.run_id,
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "point_index": point_idx,
                        "x": float(point_model[0]),
                        "y": float(point_model[1]),
                        "x_native": float(point_native[0]),
                        "y_native": float(point_native[1]),
                        "pred_depth_mm": float(pred_mm),
                        "gt_depth_mm": gt_mm,
                        "abs_error_mm": float(abs(pred_mm - gt_mm)),
                        "abs_rel": float(abs(pred_mm - gt_mm) / max(gt_mm, 1e-8)),
                        "rgb_path": sample["rgb_path"],
                        "json_path": sample["json_path"],
                        "camera_id": sample["camera_id"],
                    }
                )

            frame_rows.append(
                {
                    "run_id": args.run_id,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "sequence": sample["sequence"],
                    "camera_id": sample["camera_id"],
                    "point_count": int(point_metrics["count"]),
                    "gt_depth_mm": gt_mm,
                    "pred_mean_mm": pred_mean_mm,
                    "pred_median_mm": pred_median_mm,
                    "pred_min_mm": float(np.min(pred_values_mm)),
                    "pred_max_mm": float(np.max(pred_values_mm)),
                    **{
                        f"pred{i}_mm": float(value)
                        for i, value in enumerate(paper_pred_values_mm, start=1)
                    },
                    "paper_num_points": int(args.paper_num_points),
                    "paper_pred_mean_mm": paper_pred_mean_mm,
                    "paper_mae_mm": float(paper_point_metrics["mae_mm"]),
                    "paper_rmse_mm": float(paper_point_metrics["rmse_mm"]),
                    "paper_abs_rel": float(paper_point_metrics["abs_rel"]),
                    "point_mae_mean_mm": float(point_metrics["mae_mm"]),
                    "mae_mm": float(frame_metric_on_mean["mae_mm"]),
                    "rmse_mm": float(frame_metric_on_mean["rmse_mm"]),
                    "abs_rel": float(frame_metric_on_mean["abs_rel"]),
                    "log10": float(frame_metric_on_mean["log10"]),
                    "a1": float(frame_metric_on_mean["a1"]),
                    "a2": float(frame_metric_on_mean["a2"]),
                    "rgb_path": sample["rgb_path"],
                    "json_path": sample["json_path"],
                }
            )

            ignored_labels.update(sample["ignored_labels"])
            if args.topk > 0:
                frame_buffers[sample_id] = {
                    "rgb_bgr": _rgb_tensor_to_bgr(sample["color_curr"]),
                    "pred_depth_m": pred_depth_m,
                    "value_points": value_points,
                    "top_points": top_points,
                }

    official_rows = _official_group_rows(frame_rows)
    paper_depth_all_rows = _paper_depth_all_rows(
        frame_rows,
        paper_num_points=args.paper_num_points,
    )
    paper_metrics = _paper_eval_metrics(official_rows)

    all_points_metrics = _metrics_from_arrays(
        [row["pred_depth_mm"] for row in point_rows],
        [row["gt_depth_mm"] for row in point_rows],
    )
    frame_mean_metrics = _metrics_from_arrays(
        [row["pred_mean_mm"] for row in frame_rows],
        [row["gt_depth_mm"] for row in frame_rows],
    )
    official_metrics = _metrics_from_arrays(
        [row["pred_mm"] for row in official_rows],
        [row["truth_mm"] for row in official_rows],
    )

    per_point_csv = out_dir / "per_point_metrics.csv"
    per_point_jsonl = out_dir / "per_point_metrics.jsonl"
    per_frame_csv = out_dir / "per_frame_metrics.csv"
    per_frame_jsonl = out_dir / "per_frame_metrics.jsonl"
    official_csv = out_dir / "official_group_summary.csv"
    official_jsonl = out_dir / "official_group_summary.jsonl"
    paper_depth_all_csv = out_dir / "paper_depth_all.csv"
    paper_depth_summary_csv = out_dir / "paper_depth_summary.csv"
    paper_depth_eval_json = out_dir / "paper_depth_eval_summary.json"

    _write_csv(per_point_csv, point_rows)
    _write_jsonl(per_point_jsonl, point_rows)
    _write_csv(per_frame_csv, frame_rows)
    _write_jsonl(per_frame_jsonl, frame_rows)
    _write_csv(official_csv, official_rows)
    _write_jsonl(official_jsonl, official_rows)
    _write_csv(paper_depth_all_csv, paper_depth_all_rows)
    _write_csv(
        paper_depth_summary_csv,
        [
            {
                "id": row["group_id"],
                "pred": row["pred_mm"],
                "truth": row["truth_mm"],
            }
            for row in official_rows
        ],
    )
    _write_json(paper_depth_eval_json, paper_metrics)

    paper_style_summary = {
        "protocol": "polysense3d_official_depth_raw",
        "median_scaling": False,
        "paper_num_points": args.paper_num_points,
        "raw_metrics": paper_metrics,
    }
    median_scaled_outputs = {}
    median_scaled_scope_metrics = {}
    if args.median_scaling:
        official_scale, official_scaled_rows = _median_scaled_group_rows(official_rows)
        official_scaled_metrics = _paper_eval_metrics(official_scaled_rows)
        paper_depth_summary_scaled_csv = out_dir / "paper_depth_summary_median_scaled.csv"
        paper_depth_eval_scaled_json = out_dir / "paper_depth_eval_median_scaled_summary.json"
        official_group_scaled_csv = out_dir / "official_group_summary_median_scaled.csv"
        official_group_scaled_jsonl = out_dir / "official_group_summary_median_scaled.jsonl"
        _write_csv(
            paper_depth_summary_scaled_csv,
            [
                {
                    "id": row["group_id"],
                    "pred": row["pred_mm"],
                    "truth": row["truth_mm"],
                    "median_scale": row["median_scale"],
                    "raw_pred": row["raw_pred_mm"],
                }
                for row in official_scaled_rows
            ],
        )
        _write_json(paper_depth_eval_scaled_json, official_scaled_metrics)
        _write_csv(official_group_scaled_csv, official_scaled_rows)
        _write_jsonl(official_group_scaled_jsonl, official_scaled_rows)

        all_points_scale, all_points_scaled_metrics = _median_scaled_metrics_from_arrays(
            [row["pred_depth_mm"] for row in point_rows],
            [row["gt_depth_mm"] for row in point_rows],
        )
        frame_scale, frame_scaled_metrics = _median_scaled_metrics_from_arrays(
            [row["pred_mean_mm"] for row in frame_rows],
            [row["gt_depth_mm"] for row in frame_rows],
        )
        median_scaled_scope_metrics = {
            "all_points": {
                "median_scale": all_points_scale,
                **all_points_scaled_metrics,
            },
            "frame_means": {
                "median_scale": frame_scale,
                **frame_scaled_metrics,
            },
            "official_groups": {
                "median_scale": official_scale,
                **_metrics_from_arrays(
                    [row["pred_mm"] for row in official_scaled_rows],
                    [row["truth_mm"] for row in official_scaled_rows],
                ),
            },
        }
        median_scaled_outputs = {
            "paper_depth_summary_csv": str(paper_depth_summary_scaled_csv),
            "paper_depth_eval_json": str(paper_depth_eval_scaled_json),
            "official_group_csv": str(official_group_scaled_csv),
            "official_group_jsonl": str(official_group_scaled_jsonl),
        }
        paper_style_summary.update(
            {
                "median_scale": official_scale,
                "median_scaled_metrics": official_scaled_metrics,
            }
        )

    if args.topk > 0:
        _save_topk_artifacts(
            out_dir / "topk",
            frame_rows,
            frame_buffers,
            topk=args.topk,
            depth_vis_min=args.depth_vis_min,
            depth_vis_max=args.depth_vis_max,
        )

    summary = {
        "run_id": args.run_id,
        "data_root": str(Path(args.data_root)),
        "weights_folder": str(weights_folder),
        "height": args.height,
        "width": args.width,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "sample_radius_px": args.sample_radius_px,
        "median_scaling": bool(args.median_scaling),
        "dataset": {
            "frames": len(frame_rows),
            "points": len(point_rows),
            "groups": len(official_rows),
            "ignored_labels": dict(ignored_labels),
            "manifest_errors": dataset.errors,
        },
        "metrics": {
            "all_points": all_points_metrics,
            "frame_means": frame_mean_metrics,
            "official_groups": official_metrics,
        },
        "median_scaled_metrics": median_scaled_scope_metrics,
        "polysense3d_paper_style": paper_style_summary,
        "outputs": {
            "per_point_csv": str(per_point_csv),
            "per_point_jsonl": str(per_point_jsonl),
            "per_frame_csv": str(per_frame_csv),
            "per_frame_jsonl": str(per_frame_jsonl),
            "official_group_csv": str(official_csv),
            "official_group_jsonl": str(official_jsonl),
            "paper_depth_all_csv": str(paper_depth_all_csv),
            "paper_depth_summary_csv": str(paper_depth_summary_csv),
            "paper_depth_eval_json": str(paper_depth_eval_json),
            "median_scaled": median_scaled_outputs,
        },
    }

    summary_json = out_dir / "summary.json"
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    _write_summary_md(out_dir / "summary.md", summary)

    print("\n========== PolypSense3D Sparse Depth Evaluation ==========")
    print(f"  Run ID:             {args.run_id}")
    print(f"  Frames evaluated:   {len(frame_rows)}")
    print(f"  Sparse points:      {len(point_rows)}")
    print(f"  Groups:             {len(official_rows)}")
    print(f"  Weights:            {weights_folder}")
    print(f"  Output:             {out_dir}")
    print("")
    for label, metrics in [
        ("All points", all_points_metrics),
        ("Frame means", frame_mean_metrics),
        ("Official groups", official_metrics),
    ]:
        print(
            f"  {label:<16} "
            f"MAE={metrics['mae_mm']:.4f} mm  "
            f"RMSE={metrics['rmse_mm']:.4f} mm  "
            f"AbsRel={metrics['abs_rel']:.4f}  "
            f"a1={metrics['a1']:.4f}"
        )
    print("")
    print("  PolypSense3D paper-style raw official groups:")
    print(
        f"    MAE={paper_metrics['mae']:.4f} mm  "
        f"RMSE={paper_metrics['rmse']:.4f} mm  "
        f"AbsRel={paper_metrics['absrel']:.4f}  "
        f"a1={paper_metrics['a1_percent']:.4f}%"
    )
    if args.median_scaling:
        scaled = paper_style_summary["median_scaled_metrics"]
        print("  Median-scaled official groups:")
        print(
            f"    scale={paper_style_summary['median_scale']:.6f}  "
            f"MAE={scaled['mae']:.4f} mm  "
            f"RMSE={scaled['rmse']:.4f} mm  "
            f"AbsRel={scaled['absrel']:.4f}  "
            f"a1={scaled['a1_percent']:.4f}%"
        )
    if ignored_labels:
        print(f"  Ignored labels:     {dict(ignored_labels)}")
    print("==========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate metric sparse depth on PolypSense3D clinical labels."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--weights_folder",
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
    parser.add_argument("--out_dir", default="reports/polysense3d/sparse_depth")
    parser.add_argument("--run_id", default="polysense3d_sparse")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.001)
    parser.add_argument("--max_depth", type=float, default=0.1)
    parser.add_argument(
        "--sample_radius_px",
        type=int,
        default=0,
        help="Median-sample a square patch around each point; 0 uses nearest point.",
    )
    parser.add_argument(
        "--paper_num_points",
        type=int,
        default=3,
        help="Number of LabelMe value points used for PolypSense3D paper-style pred columns.",
    )
    parser.add_argument(
        "--median_scaling",
        action="store_true",
        help=(
            "Also report one global official-group median-scaled diagnostic. "
            "Raw PolypSense3D paper-style metrics remain unchanged."
        ),
    )
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--depth_vis_min", type=float, default=0.001)
    parser.add_argument("--depth_vis_max", type=float, default=0.1)
    parser.add_argument("--allow_manifest_errors", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
