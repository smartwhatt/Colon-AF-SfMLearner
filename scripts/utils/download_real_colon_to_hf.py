#!/usr/bin/env python3
"""Convert an extracted REAL-Colon tree to streaming-ready HF triplet shards.

Each Parquet row has the same unified triplet schema as C3VDv2 HF shards.
REAL-Colon has no depth or pose GT, so supervised label columns are null and
the availability flags are false.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo
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


def _encode_image_file(path: str) -> dict:
    with open(path, "rb") as f:
        return {"bytes": f.read(), "path": None}


def _triplet_rows_for_sequence(
    data_root: str,
    seq: SequenceInfo,
    target_fps: float,
    fallback_stride: int,
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
    api: HfApi,
    existing_hf: set[str],
) -> int:
    os.makedirs(args.tmp_dir, exist_ok=True)
    rows = []
    shard_n = 0
    total_rows = 0

    def flush() -> None:
        nonlocal rows, shard_n, total_rows
        if not rows:
            return
        path_in_repo = f"data/{split_name}/shard_{shard_n:05d}.parquet"
        local_path = os.path.join(args.tmp_dir, f"real_colon_{split_name}_{shard_n:05d}.parquet")
        write_parquet(rows, local_path)
        if path_in_repo in existing_hf and not args.overwrite_existing:
            print(f"  {path_in_repo} already on HF, skipping upload")
        else:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Add REAL-Colon {split_name} shard {shard_n:05d}",
            )
        os.remove(local_path)
        total_rows += len(rows)
        rows = []
        shard_n += 1

    for seq in tqdm(sequences, desc=f"{split_name} sequences", ncols=90):
        for row in _triplet_rows_for_sequence(
            args.data_root, seq, args.target_fps, args.fallback_stride
        ):
            rows.append(row)
            if len(rows) >= args.shard_size:
                flush()

    flush()
    print(f"  {split_name}: {total_rows:,} triplets")
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted REAL-Colon frames to HF triplet Parquet shards"
    )
    parser.add_argument("repo_id", help="HF dataset repo, e.g. username/real-colon")
    parser.add_argument("--data-root", required=True, help="Extracted REAL-Colon root")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--tmp-dir", default="/tmp/real_colon_hf")
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
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

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

    print(f"Creating/verifying HF repo: {args.repo_id}")
    api = HfApi(token=hf_token)
    create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True, token=hf_token)
    try:
        existing_hf = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    except Exception:
        existing_hf = set()

    readme = README_TEMPLATE.format(repo_id=args.repo_id)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Update README with REAL-Colon HF schema",
    )

    totals = {
        "train": process_split("train", train, args, api, existing_hf),
        "val": process_split("val", val, args, api, existing_hf),
        "test": process_split("test", test, args, api, existing_hf),
    }
    print("Done.")
    print(totals)
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
