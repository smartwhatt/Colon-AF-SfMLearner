#!/usr/bin/env python3
"""Convert an extracted REAL-Colon tree to streaming-ready HF triplet shards.

Each Parquet row has the same unified triplet schema as C3VDv2 HF shards.
REAL-Colon has no depth or pose GT, so supervised label columns are null and
the availability flags are false.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo
from PIL import Image
from tqdm import tqdm


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.generate_splits_realcolon import (  # noqa: E402
    SequenceInfo,
    available_frame_indices,
    frame_path,
    load_video_info,
    partition_sequences,
    sample_available_frame_indices,
    validate_sequences,
)


README_TEMPLATE = """\
---
license: cc-by-4.0
task_categories:
- depth-estimation
language:
- en
tags:
- endoscopy
- colonoscopy
- self-supervised
- medical
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/**/*.parquet
  - split: val
    path: data/val/**/*.parquet
  - split: test
    path: data/test/**/*.parquet
dataset_info:
  features:
  - name: dataset_name
    dtype: string
  - name: sequence
    dtype: string
  - name: frame_idx
    dtype: int32
  - name: frame_idx_prev
    dtype: int32
  - name: frame_idx_curr
    dtype: int32
  - name: frame_idx_next
    dtype: int32
  - name: rgb_prev
    dtype: image
  - name: rgb_curr
    dtype: image
  - name: rgb_next
    dtype: image
  - name: depth
    dtype: image
  - name: occlusion
    dtype: image
  - name: pose_curr2prev
    sequence: float64
  - name: pose_curr2next
    sequence: float64
  - name: K
    sequence: float32
  - name: has_depth
    dtype: bool
  - name: has_occlusion
    dtype: bool
  - name: has_pose
    dtype: bool
  - name: source_fps
    dtype: float32
  - name: target_fps
    dtype: float32
  - name: frame_stride
    dtype: int32
  splits:
  - name: train
  - name: val
  - name: test
---

# REAL-Colon HF Triplets

This dataset is a re-packaged version of REAL-Colon for streaming
self-supervised AF-SfMLearner training. The original dataset is distributed
under CC BY 4.0.

Original dataset DOI: https://doi.org/10.25452/figshare.plus.22202866

Rows are lower-fps temporal triplets sampled from the extracted frame files that
exist on disk. If the official extraction is already subsampled, the requested
`target_fps` is approximated by the nearest integer step on that available frame
grid. `frame_stride` records the actual original-frame distance between each
row's previous and current image. REAL-Colon does not provide depth or pose
ground truth, so `depth`, `occlusion`, `pose_curr2prev`, and `pose_curr2next`
are null and `has_depth`, `has_occlusion`, and `has_pose` are false.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train", streaming=True)
row = next(iter(ds))
```
"""


REAL_COLON_K_NORM = np.array(
    [[0.82, 0.0, 0.5], [0.0, 1.02, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
).flatten().tolist()

_HF_IMAGE_TYPE = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])

PARQUET_SCHEMA = pa.schema([
    pa.field("dataset_name",    pa.string()),
    pa.field("sequence",        pa.string()),
    pa.field("frame_idx",       pa.int32()),
    pa.field("frame_idx_prev",  pa.int32()),
    pa.field("frame_idx_curr",  pa.int32()),
    pa.field("frame_idx_next",  pa.int32()),
    pa.field("rgb_prev",        _HF_IMAGE_TYPE),
    pa.field("rgb_curr",        _HF_IMAGE_TYPE),
    pa.field("rgb_next",        _HF_IMAGE_TYPE),
    pa.field("depth",           _HF_IMAGE_TYPE),
    pa.field("occlusion",       _HF_IMAGE_TYPE),
    pa.field("pose_curr2prev",  pa.list_(pa.float64())),
    pa.field("pose_curr2next",  pa.list_(pa.float64())),
    pa.field("K",               pa.list_(pa.float32())),
    pa.field("has_depth",       pa.bool_()),
    pa.field("has_occlusion",   pa.bool_()),
    pa.field("has_pose",        pa.bool_()),
    pa.field("source_fps",      pa.float32()),
    pa.field("target_fps",      pa.float32()),
    pa.field("frame_stride",    pa.int32()),
])


QUALITY_REPORT_FIELDS = [
    "split",
    "sequence",
    "frame_idx",
    "path",
    "reason",
    "width",
    "height",
    "mean_luma",
    "luma_std",
    "dark_fraction",
    "saturated_fraction",
    "edge_mean",
    "noise_score",
]


@dataclass(frozen=True)
class QualityFilterConfig:
    downsample_size: int
    min_mean_luma: float
    max_dark_fraction: float
    max_saturated_fraction: float
    min_luma_std: float
    min_edge_mean: float
    max_noise_score: float | None


@dataclass(frozen=True)
class FrameQuality:
    ok: bool
    reason: str
    width: int
    height: int
    mean_luma: float
    luma_std: float
    dark_fraction: float
    saturated_fraction: float
    edge_mean: float
    noise_score: float


@dataclass
class QualityFilterState:
    config: QualityFilterConfig
    cache: dict[str, FrameQuality]
    stats: Counter
    report_writer: csv.DictWriter | None = None
    reported_paths: set[str] | None = None

    def __post_init__(self) -> None:
        if self.reported_paths is None:
            self.reported_paths = set()


def _encode_image_file(path: str) -> dict:
    with open(path, "rb") as f:
        return {"bytes": f.read(), "path": None}


def _box_blur3(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _measure_frame_quality(path: str, config: QualityFilterConfig) -> FrameQuality:
    try:
        with Image.open(path) as image:
            width, height = image.size
            gray_image = image.convert("L")
            gray_image.thumbnail((config.downsample_size, config.downsample_size))
            gray = np.asarray(gray_image, dtype=np.float32)
    except Exception as exc:
        return FrameQuality(
            ok=False,
            reason=f"decode_error:{type(exc).__name__}",
            width=0,
            height=0,
            mean_luma=0.0,
            luma_std=0.0,
            dark_fraction=1.0,
            saturated_fraction=0.0,
            edge_mean=0.0,
            noise_score=0.0,
        )

    if gray.size == 0:
        return FrameQuality(
            ok=False,
            reason="empty_image",
            width=width,
            height=height,
            mean_luma=0.0,
            luma_std=0.0,
            dark_fraction=1.0,
            saturated_fraction=0.0,
            edge_mean=0.0,
            noise_score=0.0,
        )

    mean_luma = float(gray.mean())
    luma_std = float(gray.std())
    dark_fraction = float((gray <= 8).mean())
    saturated_fraction = float((gray >= 248).mean())
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_mean = float((dx.mean() + dy.mean()) / 2.0) if dx.size and dy.size else 0.0
    noise_score = float(np.abs(gray - _box_blur3(gray)).std())

    reasons = []
    if mean_luma < config.min_mean_luma:
        reasons.append("too_dark")
    if dark_fraction > config.max_dark_fraction:
        reasons.append("mostly_black")
    if saturated_fraction > config.max_saturated_fraction:
        reasons.append("mostly_saturated")
    if luma_std < config.min_luma_std and edge_mean < config.min_edge_mean:
        reasons.append("low_texture")
    if config.max_noise_score is not None and noise_score > config.max_noise_score:
        reasons.append("high_noise")

    return FrameQuality(
        ok=not reasons,
        reason=";".join(reasons) if reasons else "ok",
        width=width,
        height=height,
        mean_luma=mean_luma,
        luma_std=luma_std,
        dark_fraction=dark_fraction,
        saturated_fraction=saturated_fraction,
        edge_mean=edge_mean,
        noise_score=noise_score,
    )


def _get_frame_quality(
    path: str,
    split_name: str,
    seq_id: str,
    frame_idx: int,
    state: QualityFilterState,
) -> FrameQuality:
    quality = state.cache.get(path)
    if quality is None:
        quality = _measure_frame_quality(path, state.config)
        state.cache[path] = quality
        state.stats["quality_frames_checked"] += 1
        if not quality.ok:
            state.stats["quality_bad_frames"] += 1
            for reason in quality.reason.split(";"):
                state.stats[f"quality_bad_frame_{reason}"] += 1

    if not quality.ok and state.report_writer is not None and path not in state.reported_paths:
        state.report_writer.writerow(
            {
                "split": split_name,
                "sequence": seq_id,
                "frame_idx": frame_idx,
                "path": path,
                "reason": quality.reason,
                "width": quality.width,
                "height": quality.height,
                "mean_luma": f"{quality.mean_luma:.4f}",
                "luma_std": f"{quality.luma_std:.4f}",
                "dark_fraction": f"{quality.dark_fraction:.6f}",
                "saturated_fraction": f"{quality.saturated_fraction:.6f}",
                "edge_mean": f"{quality.edge_mean:.4f}",
                "noise_score": f"{quality.noise_score:.4f}",
            }
        )
        state.reported_paths.add(path)

    return quality


def _triplet_rows_for_sequence(
    data_root: str,
    seq: SequenceInfo,
    target_fps: float,
    fallback_stride: int,
    quality_state: QualityFilterState | None = None,
    split_name: str = "",
):
    frame_indices = sample_available_frame_indices(
        seq,
        available_frame_indices(data_root, seq.seq_id),
        target_fps,
        fallback_stride,
    )
    if len(frame_indices) < 3:
        return

    for pos in range(1, len(frame_indices) - 1):
        idx_prev = frame_indices[pos - 1]
        idx_curr = frame_indices[pos]
        idx_next = frame_indices[pos + 1]
        frame_stride = idx_curr - idx_prev

        prev_path = frame_path(data_root, seq.seq_id, idx_prev)
        curr_path = frame_path(data_root, seq.seq_id, idx_curr)
        next_path = frame_path(data_root, seq.seq_id, idx_next)
        if not (
            os.path.isfile(prev_path)
            and os.path.isfile(curr_path)
            and os.path.isfile(next_path)
        ):
            continue
        if quality_state is not None:
            quality_state.stats["quality_candidate_triplets"] += 1
            qualities = [
                _get_frame_quality(prev_path, split_name, seq.seq_id, idx_prev, quality_state),
                _get_frame_quality(curr_path, split_name, seq.seq_id, idx_curr, quality_state),
                _get_frame_quality(next_path, split_name, seq.seq_id, idx_next, quality_state),
            ]
            if not all(quality.ok for quality in qualities):
                quality_state.stats["quality_skipped_triplets"] += 1
                continue

        yield {
            "dataset_name":    "real_colon",
            "sequence":        seq.seq_id,
            "frame_idx":       idx_curr,
            "frame_idx_prev":  idx_prev,
            "frame_idx_curr":  idx_curr,
            "frame_idx_next":  idx_next,
            "rgb_prev":        _encode_image_file(prev_path),
            "rgb_curr":        _encode_image_file(curr_path),
            "rgb_next":        _encode_image_file(next_path),
            "depth":           None,
            "occlusion":       None,
            "pose_curr2prev":  None,
            "pose_curr2next":  None,
            "K":               REAL_COLON_K_NORM,
            "has_depth":       False,
            "has_occlusion":   False,
            "has_pose":        False,
            "source_fps":      seq.fps,
            "target_fps":      target_fps,
            "frame_stride":    frame_stride,
        }


def write_parquet(rows: list[dict], path: str) -> None:
    cols = {k: [r[k] for r in rows] for k in PARQUET_SCHEMA.names}
    pq.write_table(
        pa.table(cols, schema=PARQUET_SCHEMA),
        path,
        compression="snappy",
        row_group_size=100,
        write_statistics=True,
        data_page_size=1024**2,
    )


def process_split(
    split_name: str,
    sequences: list[SequenceInfo],
    args: argparse.Namespace,
    staging_dir: str,
    quality_state: QualityFilterState | None = None,
) -> int:
    split_dir = os.path.join(staging_dir, "data", split_name)
    os.makedirs(split_dir, exist_ok=True)
    rows = []
    shard_n = 0
    total_rows = 0

    def flush() -> None:
        nonlocal rows, shard_n, total_rows
        if not rows:
            return
        local_path = os.path.join(split_dir, f"shard_{shard_n:05d}.parquet")
        write_parquet(rows, local_path)
        total_rows += len(rows)
        rows = []
        shard_n += 1

    for seq in tqdm(sequences, desc=f"{split_name} sequences", ncols=90):
        for row in _triplet_rows_for_sequence(
            args.data_root,
            seq,
            args.target_fps,
            args.fallback_stride,
            quality_state=quality_state,
            split_name=split_name,
        ):
            rows.append(row)
            if len(rows) >= args.shard_size:
                flush()

    flush()
    print(f"  {split_name}: {total_rows:,} triplets")
    return total_rows


def _quality_config_from_args(args: argparse.Namespace) -> QualityFilterConfig:
    return QualityFilterConfig(
        downsample_size=args.quality_downsample_size,
        min_mean_luma=args.min_mean_luma,
        max_dark_fraction=args.max_dark_fraction,
        max_saturated_fraction=args.max_saturated_fraction,
        min_luma_std=args.min_luma_std,
        min_edge_mean=args.min_edge_mean,
        max_noise_score=args.max_noise_score,
    )


def _default_staging_dir(args: argparse.Namespace) -> str:
    safe_repo_id = args.repo_id.replace("/", "__")
    return os.path.join(args.tmp_dir, "real_colon_hf_upload", safe_repo_id)


def _prepare_staging_dir(args: argparse.Namespace) -> str:
    staging_dir = args.staging_dir or _default_staging_dir(args)
    if args.upload_staged_only:
        if not os.path.isdir(staging_dir):
            raise SystemExit(f"Staging directory does not exist: {staging_dir}")
        return staging_dir

    if os.path.exists(staging_dir):
        if args.staging_dir is not None and not args.overwrite_staging:
            raise SystemExit(
                f"Staging directory already exists: {staging_dir}\n"
                "Pass --overwrite-staging to replace it, or choose another --staging-dir."
            )
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir


def _write_readme(staging_dir: str, repo_id: str) -> None:
    readme_path = os.path.join(staging_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_TEMPLATE.format(repo_id=repo_id))


def _summarize_existing_remote(api: HfApi, args: argparse.Namespace) -> set[str]:
    if args.skip_remote_check:
        print("Skipping remote file check.")
        return set()

    print("Checking existing HF repo files...")
    try:
        existing_hf = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    except Exception as exc:
        print(f"  warning: could not list remote files: {type(exc).__name__}: {exc}")
        return set()

    split_counts = Counter(
        parts[1]
        for path in existing_hf
        if path.startswith("data/") and path.endswith(".parquet")
        for parts in [path.split("/")]
        if len(parts) >= 3
    )
    print(f"  remote files: {len(existing_hf):,}")
    if split_counts:
        print(
            "  remote parquet shards: "
            + ", ".join(f"{split}={count:,}" for split, count in sorted(split_counts.items()))
        )
        if args.upload_mode == "folder" and not args.overwrite_existing:
            print(
                "  warning: existing data shards will be overwritten only when names match; "
                "unmatched stale shards remain unless --overwrite-existing is set"
            )
    else:
        print("  remote parquet shards: none")
    return existing_hf


def _upload_staged_folder(
    api: HfApi,
    args: argparse.Namespace,
    hf_token: str | None,
    staging_dir: str,
) -> None:
    if args.upload_mode == "none":
        print(f"Upload skipped. Staged dataset folder: {staging_dir}")
        return

    if args.upload_mode == "folder":
        delete_patterns = ["data/**"] if args.overwrite_existing else None
        print(f"Uploading staged folder in one commit: {staging_dir}")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=staging_dir,
            repo_type="dataset",
            token=hf_token,
            commit_message="Upload REAL-Colon HF dataset",
            delete_patterns=delete_patterns,
        )
        return

    if args.overwrite_existing:
        print(
            "  warning: --overwrite-existing cannot delete stale remote files with "
            "--upload-mode large-folder; use --upload-mode folder for replacement uploads"
        )
    print(f"Uploading staged folder with upload_large_folder: {staging_dir}")
    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=staging_dir,
        repo_type="dataset",
        num_workers=args.upload_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted REAL-Colon frames to HF triplet Parquet shards"
    )
    parser.add_argument("repo_id", help="HF dataset repo, e.g. username/real-colon")
    parser.add_argument("--data-root", help="Extracted REAL-Colon root")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--tmp-dir", default="/tmp/real_colon_hf")
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Local folder to stage README.md and data/ before bulk upload",
    )
    parser.add_argument(
        "--overwrite-staging",
        action="store_true",
        help="Replace an existing user-provided --staging-dir before generating shards",
    )
    parser.add_argument(
        "--upload-staged-only",
        action="store_true",
        help="Upload an existing --staging-dir without regenerating shards",
    )
    parser.add_argument(
        "--upload-mode",
        choices=("folder", "large-folder", "none"),
        default="folder",
        help="folder = one upload_folder commit; large-folder = resumable large upload; none = stage only",
    )
    parser.add_argument("--upload-workers", type=int, default=None)
    parser.add_argument("--skip-remote-check", action="store_true")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--fallback-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--verify-sample-paths", action="store_true")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace existing HF shard paths instead of treating them as resume-complete",
    )
    parser.add_argument(
        "--filter-quality",
        action="store_true",
        help="Skip triplets containing corrupt, black, saturated, or blank frames",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="CSV path for rejected-frame quality metrics; defaults under --tmp-dir",
    )
    parser.add_argument("--quality-downsample-size", type=int, default=256)
    parser.add_argument("--min-mean-luma", type=float, default=8.0)
    parser.add_argument("--max-dark-fraction", type=float, default=0.98)
    parser.add_argument("--max-saturated-fraction", type=float, default=0.98)
    parser.add_argument("--min-luma-std", type=float, default=3.0)
    parser.add_argument("--min-edge-mean", type=float, default=1.0)
    parser.add_argument(
        "--max-noise-score",
        type=float,
        default=None,
        help="Optional high-frequency noise cutoff; disabled by default",
    )
    args = parser.parse_args()
    if not args.upload_staged_only and not args.data_root:
        parser.error("--data-root is required unless --upload-staged-only is set")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    staging_dir = _prepare_staging_dir(args)

    print(f"Creating/verifying HF repo: {args.repo_id}")
    api = HfApi(token=hf_token)
    create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True, token=hf_token)
    _summarize_existing_remote(api, args)

    if args.upload_staged_only:
        _upload_staged_folder(api, args, hf_token, staging_dir)
        print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")
        return

    print("Reading REAL-Colon metadata...")
    sequences = load_video_info(args.data_root)
    print(f"  video_info.csv: {len(sequences)} usable videos")

    print("Validating extracted frame folders...")
    sequences, warnings = validate_sequences(
        args.data_root,
        sequences,
        verify_sample_paths=args.verify_sample_paths,
        sample_stride=1 if args.verify_sample_paths else args.fallback_stride,
        require_contiguous_neighbors=False,
    )
    for warning in warnings[:20]:
        print(f"  warning: {warning}")
    if len(warnings) > 20:
        print(f"  warning: {len(warnings) - 20} more omitted")
    if not sequences:
        raise SystemExit("No valid REAL-Colon frame folders found.")

    train, val, test = partition_sequences(
        sequences,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"Split videos: train={len(train)} val={len(val)} test={len(test)}")

    print(f"Staging dataset folder: {staging_dir}")
    _write_readme(staging_dir, args.repo_id)

    quality_state = None
    quality_report_file = None
    if args.filter_quality:
        quality_report_path = args.quality_report or os.path.join(
            args.tmp_dir, "real_colon_quality_report.csv"
        )
        quality_report_dir = os.path.dirname(quality_report_path)
        if quality_report_dir:
            os.makedirs(quality_report_dir, exist_ok=True)
        quality_report_file = open(quality_report_path, "w", newline="")
        quality_report_writer = csv.DictWriter(
            quality_report_file, fieldnames=QUALITY_REPORT_FIELDS
        )
        quality_report_writer.writeheader()
        quality_state = QualityFilterState(
            config=_quality_config_from_args(args),
            cache={},
            stats=Counter(),
            report_writer=quality_report_writer,
        )
        print(f"Quality filtering enabled; rejected-frame report: {quality_report_path}")

    try:
        totals = {
            "train": process_split("train", train, args, staging_dir, quality_state),
            "val": process_split("val", val, args, staging_dir, quality_state),
            "test": process_split("test", test, args, staging_dir, quality_state),
        }
    finally:
        if quality_report_file is not None:
            quality_report_file.close()

    if quality_state is not None:
        stats = quality_state.stats
        print("Quality filter summary:")
        print(f"  frames checked: {stats['quality_frames_checked']:,}")
        print(f"  bad frames: {stats['quality_bad_frames']:,}")
        print(f"  candidate triplets: {stats['quality_candidate_triplets']:,}")
        print(f"  skipped triplets: {stats['quality_skipped_triplets']:,}")
        for key, value in sorted(stats.items()):
            if key.startswith("quality_bad_frame_"):
                print(f"  {key.removeprefix('quality_bad_frame_')}: {value:,}")

    print("Done.")
    print(totals)
    _upload_staged_folder(api, args, hf_token, staging_dir)
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
