#!/usr/bin/env python3
"""Create fixed REALColon qualitative depth panels for checkpoint comparison.

This is intended for domain-adaptation diagnostics where REALColon has no
dense depth labels. It renders the same HF rows through multiple checkpoints,
saves side-by-side RGB/depth panels, and writes per-sample depth differences
against the first checkpoint.

Example:

    uv run python tools/qualitative_realcolon_panel.py \
        --hf_repo_id SmartWhatt/real-colon-SfM \
        --hf_split val \
        --hf_cache_dir /scratch/.../hf_cache/datasets \
        --weights run_d_base=logs/c3vdv2_hf/stage2_specular_pps/models/weights_best \
        --weights run_d_pose=logs/real_colon_hf/realcolon_run_d_diag/models/weights_0 \
        --out_dir reports/realcolon/run_d_pose_panel
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

from colon_datasets.hf_triplet_dataset import build_hf_triplet_dataset  # noqa: E402
from tools.depth_backend import (  # noqa: E402
    load_depth_model,
    predict_depth_m,
    requires_weights_folder,
)


def _parse_weight_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "--weights must use label=/path/to/weights format"
        )
    label, path_text = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("weight label must not be empty")
    path = Path(path_text).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"weights folder does not exist: {path}")
    for filename in ("encoder.pth", "depth.pth"):
        if not (path / filename).is_file():
            raise argparse.ArgumentTypeError(f"missing {filename} under {path}")
    return label, path


def _parse_indices(text: str) -> list[int]:
    """Parse comma indexes and simple start:stop:step ranges."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) if x else None for x in part.split(":")]
            if len(bits) not in {2, 3}:
                raise argparse.ArgumentTypeError(f"bad index range: {part}")
            start = 0 if bits[0] is None else bits[0]
            stop = bits[1]
            step = 1 if len(bits) == 2 or bits[2] is None else bits[2]
            if stop is None:
                raise argparse.ArgumentTypeError(
                    f"range stop is required for fixed panels: {part}"
                )
            out.extend(range(start, stop, step))
        else:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("no indices parsed")
    return out


def _safe_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    if hasattr(value, "item"):
        return str(value.item())
    return str(value)


def _load_model(
    weights_folder: Path,
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


def _predict_depth(
    model,
    color: torch.Tensor,
    min_depth: float,
    max_depth: float,
    *,
    depth_backend: str = "af_sfm",
):
    depth = predict_depth_m(
        model,
        color,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_backend=depth_backend,
    )
    return depth.squeeze(0).squeeze(0).cpu().numpy()


def _rgb_tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _depth_to_bgr(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-8)
    norm = np.clip((depth - vmin) / denom, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _labeled_tile(image_bgr: np.ndarray, label: str) -> np.ndarray:
    tile = image_bgr.copy()
    bar_h = 24
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], bar_h), (0, 0, 0), -1)
    tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)
    cv2.putText(
        tile,
        label[:80],
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _make_panel(rgb_bgr: np.ndarray, depths: list[tuple[str, np.ndarray]], args) -> np.ndarray:
    tiles = [_labeled_tile(rgb_bgr, "rgb")]
    for label, depth in depths:
        tiles.append(
            _labeled_tile(
                _depth_to_bgr(depth, args.depth_vis_min, args.depth_vis_max),
                f"{label} depth",
            )
        )
    return np.concatenate(tiles, axis=1)


def _select_indices(args, dataset_len: int) -> list[int]:
    if args.indices:
        indices = _parse_indices(args.indices)
    else:
        indices = [
            args.start_index + i * args.stride for i in range(args.num_samples)
        ]
    bad = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if bad:
        raise ValueError(
            f"indices outside split length {dataset_len}: "
            + ", ".join(str(x) for x in bad[:10])
        )
    return indices


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    weight_specs = [_parse_weight_spec(spec) for spec in (args.weights or [])]
    # Each entry: (label, model, path, backend)
    models: list[tuple[str, object, object, str]] = [
        (label, _load_model(path, device), path, "af_sfm")
        for label, path in weight_specs
    ]
    if args.depth_backend == "depth_anything":
        dav2_model = _load_model(
            None,
            device,
            depth_backend="depth_anything",
            depth_anything_repo=args.depth_anything_repo,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        models.append(
            (args.depth_anything_label, dav2_model, args.depth_anything_repo, "depth_anything")
        )

    if not models:
        raise ValueError(
            "Provide at least one --weights spec (af_sfm) or --depth_backend depth_anything."
        )

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
    dataset_len = len(dataset)
    indices = _select_indices(args, dataset_len)

    rows = []
    pair_sums: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for panel_rank, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            color = sample["color_curr"].unsqueeze(0).to(device)
            rgb_bgr = _rgb_tensor_to_bgr(sample["color_curr"])
            seq = _safe_text(sample.get("seq", ""))
            frame_idx = _safe_text(sample.get("frame_idx", ""))

            depths: list[tuple[str, np.ndarray]] = []
            for label, model, _path, backend in models:
                depths.append(
                    (
                        label,
                        _predict_depth(
                            model,
                            color,
                            args.min_depth,
                            args.max_depth,
                            depth_backend=backend,
                        ),
                    )
                )

            stem = f"{panel_rank:03d}_idx{sample_idx}_{seq}_{frame_idx}".replace("/", "_")
            panel_path = out_dir / f"{stem}_panel.png"
            cv2.imwrite(str(panel_path), _make_panel(rgb_bgr, depths, args))

            base_label, base_depth = depths[0]
            for label, depth in depths:
                diff = np.abs(depth - base_depth)
                pair_key = f"{base_label}_vs_{label}"
                stats = pair_sums.setdefault(
                    pair_key,
                    {"samples": 0, "mean_abs_diff_m": 0.0, "max_abs_diff_m": 0.0},
                )
                stats["samples"] += 1
                stats["mean_abs_diff_m"] += float(diff.mean())
                stats["max_abs_diff_m"] = max(
                    stats["max_abs_diff_m"], float(diff.max())
                )

                rows.append(
                    {
                        "sample_index": sample_idx,
                        "sequence": seq,
                        "frame_idx": frame_idx,
                        "panel": panel_path.name,
                        "baseline": base_label,
                        "checkpoint": label,
                        "depth_min_m": float(depth.min()),
                        "depth_median_m": float(np.median(depth)),
                        "depth_max_m": float(depth.max()),
                        "mean_abs_diff_from_baseline_m": float(diff.mean()),
                        "max_abs_diff_from_baseline_m": float(diff.max()),
                        "allclose_to_baseline": bool(
                            np.allclose(depth, base_depth, rtol=args.rtol, atol=args.atol)
                        ),
                    }
                )

    for stats in pair_sums.values():
        if stats["samples"]:
            stats["mean_abs_diff_m"] /= stats["samples"]

    csv_path = out_dir / "panel_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "hf_repo_id": args.hf_repo_id,
        "hf_split": args.hf_split,
        "indices": indices,
        "height": args.height,
        "width": args.width,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "weights": {label: str(path) for label, _model, path in models},
        "pair_summary": pair_sums,
        "panel_metrics": str(csv_path),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(indices)} panels to {out_dir}")
    print(f"Metrics: {csv_path}")
    print(f"Summary: {summary_path}")
    for key, stats in pair_sums.items():
        print(
            f"{key}: mean_abs_diff={stats['mean_abs_diff_m']:.8f} m "
            f"max_abs_diff={stats['max_abs_diff_m']:.8f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render fixed REALColon qualitative depth panels from HF rows."
    )
    parser.add_argument("--hf_repo_id", default="SmartWhatt/real-colon-SfM")
    parser.add_argument("--hf_split", default="val")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument(
        "--weights",
        action="append",
        default=None,
        help="Checkpoint spec in label=/path/to/weights format. Repeat for each model. "
        "Optional when --depth_backend depth_anything is used.",
    )
    parser.add_argument(
        "--depth_backend",
        choices=["af_sfm", "depth_anything"],
        default="af_sfm",
        help="Add a DA-v2 tile alongside (or instead of) --weights checkpoints.",
    )
    parser.add_argument(
        "--depth_anything_repo",
        default=None,
        help="HF repo id for DA-v2 when --depth_backend depth_anything.",
    )
    parser.add_argument(
        "--depth_anything_label",
        default="da_v2",
        help="Panel label for the DA-v2 depth tile.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--indices", default=None, help="Comma list or ranges like 0:16000:1000")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.001)
    parser.add_argument("--max_depth", type=float, default=0.1)
    parser.add_argument("--depth_vis_min", type=float, default=0.001)
    parser.add_argument("--depth_vis_max", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.depth_backend == "depth_anything" and not args.depth_anything_repo:
        parser.error("--depth_anything_repo is required when --depth_backend depth_anything")
    if not args.weights and args.depth_backend != "depth_anything":
        parser.error("at least one --weights spec is required when --depth_backend af_sfm")
    run(args)


if __name__ == "__main__":
    main()
