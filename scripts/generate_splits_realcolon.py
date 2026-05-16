"""
generate_splits_realcolon.py
----------------------------
Generate AF-SfMLearner split files for the official REAL-Colon download.

Expected extracted layout:
  <data_root>/
    001-001_frames/
      001-001_1.jpg
      001-001_2.jpg
      ...
    001-001_annotation/
    video_info.csv
    lesion_info.csv
    dataset_description.md

Note: this local split format is for contiguous frame extractions because
MonoDataset resolves temporal neighbours as frame_index - 1 and frame_index + 1.
If the official REAL-Colon extraction is already subsampled, use
scripts/utils/download_real_colon_to_hf.py instead; its HF triplet rows carry
explicit prev/curr/next images.

The output follows MonoDataset's default split convention:
  splits/real_colon/train_files.txt
  splits/real_colon/val_files.txt
  splits/real_colon/test_files.txt

Each row has the form:
  <sequence_id> <frame_index> l
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SequenceInfo:
    seq_id: str
    num_frames: int
    fps: float | None
    endoscope_brand: str
    bbps: str


def _parse_int(value: str, field_name: str, seq_id: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field_name!r} for sequence {seq_id!r}: {value!r}"
        ) from exc


def _parse_float_or_none(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_video_info(data_root: str) -> list[SequenceInfo]:
    csv_path = os.path.join(data_root, "video_info.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing REAL-Colon metadata file: {csv_path}")

    sequences = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"unique_video_name", "num_frames"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "video_info.csv is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            seq_id = (row.get("unique_video_name") or "").strip()
            if not seq_id:
                continue

            num_frames = _parse_int(row.get("num_frames", ""), "num_frames", seq_id)
            if num_frames < 3:
                continue

            sequences.append(
                SequenceInfo(
                    seq_id=seq_id,
                    num_frames=num_frames,
                    fps=_parse_float_or_none(row.get("fps", "")),
                    endoscope_brand=(row.get("endoscope_brand") or "").strip(),
                    bbps=(row.get("bbps") or "").strip(),
                )
            )

    return sorted(sequences, key=lambda s: s.seq_id)


def frame_dir(data_root: str, seq_id: str) -> str:
    return os.path.join(data_root, f"{seq_id}_frames")


def frame_path(data_root: str, seq_id: str, frame_index: int) -> str:
    return os.path.join(frame_dir(data_root, seq_id), f"{seq_id}_{frame_index}.jpg")


def available_frame_indices(data_root: str, seq_id: str) -> list[int]:
    seq_frame_dir = frame_dir(data_root, seq_id)
    if not os.path.isdir(seq_frame_dir):
        return []

    prefix = f"{seq_id}_"
    suffix = ".jpg"
    indices = []
    for filename in os.listdir(seq_frame_dir):
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            continue
        raw_index = filename[len(prefix) : -len(suffix)]
        try:
            indices.append(int(raw_index))
        except ValueError:
            continue
    return sorted(set(indices))


def infer_extracted_frame_step(frame_indices: list[int]) -> int:
    deltas = [
        right - left
        for left, right in zip(frame_indices, frame_indices[1:])
        if right > left
    ]
    if not deltas:
        return 1
    return max(1, round(statistics.median(deltas)))


def target_native_stride(
    seq: SequenceInfo,
    target_fps: float,
    fallback_stride: int,
) -> int:
    if seq.fps is None or seq.fps <= 0 or target_fps <= 0:
        return max(1, fallback_stride)
    return max(1, round(seq.fps / target_fps))


def sample_available_frame_indices(
    seq: SequenceInfo,
    frame_indices: list[int],
    target_fps: float,
    fallback_stride: int,
) -> list[int]:
    if len(frame_indices) < 3:
        return []

    extracted_step = infer_extracted_frame_step(frame_indices)
    desired_stride = target_native_stride(seq, target_fps, fallback_stride)
    available_step = max(1, round(desired_stride / extracted_step))
    return frame_indices[::available_step]


def validate_sequences(
    data_root: str,
    sequences: list[SequenceInfo],
    verify_sample_paths: bool,
    sample_stride: int,
    require_contiguous_neighbors: bool = True,
) -> tuple[list[SequenceInfo], list[str]]:
    valid = []
    warnings = []

    for seq in sequences:
        seq_frame_dir = frame_dir(data_root, seq.seq_id)
        if not os.path.isdir(seq_frame_dir):
            zip_hint = seq_frame_dir + ".zip"
            if os.path.isfile(zip_hint):
                warnings.append(
                    f"{seq.seq_id}: found {zip_hint}, but frames must be extracted"
                )
            else:
                warnings.append(f"{seq.seq_id}: missing frame directory {seq_frame_dir}")
            continue

        frame_indices = available_frame_indices(data_root, seq.seq_id)
        if len(frame_indices) < 3:
            warnings.append(
                f"{seq.seq_id}: only {len(frame_indices)} extracted frame(s) found"
            )
            continue

        if require_contiguous_neighbors:
            anchors = [1, 2, seq.num_frames - 1, seq.num_frames]
            missing_anchors = [
                idx
                for idx in anchors
                if not os.path.isfile(frame_path(data_root, seq.seq_id, idx))
            ]
            if missing_anchors:
                warnings.append(
                    f"{seq.seq_id}: missing expected contiguous frame(s) "
                    + ", ".join(str(idx) for idx in missing_anchors)
                )
                continue

        if verify_sample_paths:
            if require_contiguous_neighbors:
                sampled_indices = list(range(2, seq.num_frames, sample_stride))
                missing_sample = next(
                    (
                        idx
                        for center_idx in sampled_indices
                        for idx in (center_idx - 1, center_idx, center_idx + 1)
                        if not os.path.isfile(frame_path(data_root, seq.seq_id, idx))
                    ),
                    None,
                )
                if missing_sample is not None:
                    warnings.append(f"{seq.seq_id}: missing sampled frame {missing_sample}")
                    continue
            else:
                sampled_indices = frame_indices[:: max(1, sample_stride)]
                if len(sampled_indices) < 3:
                    warnings.append(
                        f"{seq.seq_id}: fewer than 3 frames remain after sample stride "
                        f"{sample_stride}"
                    )
                    continue

        valid.append(seq)

    return valid, warnings


def partition_sequences(
    sequences: list[SequenceInfo],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[SequenceInfo], list[SequenceInfo], list[SequenceInfo]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("--val_fraction must be in [0, 1)")
    if not 0 <= test_fraction < 1:
        raise ValueError("--test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("--val_fraction + --test_fraction must be less than 1")

    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    if n == 0:
        return [], [], []

    test_count = round(n * test_fraction)
    val_count = round(n * val_fraction)

    if test_fraction > 0 and n >= 3:
        test_count = max(1, test_count)
    if val_fraction > 0 and n - test_count >= 2:
        val_count = max(1, val_count)

    while test_count + val_count >= n and (test_count > 0 or val_count > 0):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1

    test = shuffled[:test_count]
    val = shuffled[test_count : test_count + val_count]
    train = shuffled[test_count + val_count :]
    return train, val, test


def write_split(
    sequences: list[SequenceInfo],
    out_path: str,
    stride: int,
    seed: int,
) -> int:
    if stride < 1:
        raise ValueError("stride must be >= 1")

    lines = []
    for seq in sequences:
        for idx in range(2, seq.num_frames, stride):
            lines.append(f"{seq.seq_id} {idx} l")

    random.Random(seed).shuffle(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    print(f"  {out_path}: {len(lines):,} samples from {len(sequences)} videos")
    return len(lines)


def write_manifest(
    args: argparse.Namespace,
    train: list[SequenceInfo],
    val: list[SequenceInfo],
    test: list[SequenceInfo],
    sample_counts: dict[str, int],
    warnings: list[str],
) -> None:
    manifest = {
        "dataset": "REAL-Colon",
        "data_root": os.path.abspath(args.data_root),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "train_stride": args.train_stride,
        "val_stride": args.val_stride,
        "test_stride": args.test_stride,
        "verify_sample_paths": args.verify_sample_paths,
        "split_video_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "split_sample_counts": sample_counts,
        "sequences": {
            "train": [asdict(seq) for seq in train],
            "val": [asdict(seq) for seq in val],
            "test": [asdict(seq) for seq in test],
        },
        "warnings": warnings,
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="splits/real_colon")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument(
        "--train_stride",
        type=int,
        default=10,
        help="Sample every Nth valid frame for training",
    )
    parser.add_argument(
        "--val_stride",
        type=int,
        default=10,
        help="Sample every Nth valid frame for validation",
    )
    parser.add_argument(
        "--test_stride",
        type=int,
        default=10,
        help="Sample every Nth valid frame for testing",
    )
    parser.add_argument(
        "--verify_sample_paths",
        action="store_true",
        help="Stat every sampled frame path instead of only checking anchor frames",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("Reading REAL-Colon metadata...")
    sequences = load_video_info(args.data_root)
    print(f"  video_info.csv: {len(sequences)} usable videos")

    print("Validating extracted frame folders...")
    sample_stride = 1 if args.verify_sample_paths else max(
        args.train_stride, args.val_stride, args.test_stride
    )
    sequences, warnings = validate_sequences(
        args.data_root,
        sequences,
        verify_sample_paths=args.verify_sample_paths,
        sample_stride=sample_stride,
    )
    print(f"  valid videos: {len(sequences)}")
    for warning in warnings[:20]:
        print(f"  warning: {warning}")
    if len(warnings) > 20:
        print(f"  warning: {len(warnings) - 20} more omitted; see manifest.json")

    if not sequences:
        raise SystemExit("No valid REAL-Colon frame folders found.")

    train, val, test = partition_sequences(
        sequences, args.val_fraction, args.test_fraction, args.seed
    )
    print(
        f"  Split videos: train={len(train)} val={len(val)} test={len(test)}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    sample_counts = {
        "train": write_split(
            train,
            os.path.join(args.output_dir, "train_files.txt"),
            args.train_stride,
            args.seed,
        ),
        "val": write_split(
            val,
            os.path.join(args.output_dir, "val_files.txt"),
            args.val_stride,
            args.seed + 1,
        ),
        "test": write_split(
            test,
            os.path.join(args.output_dir, "test_files.txt"),
            args.test_stride,
            args.seed + 2,
        ),
    }
    write_manifest(args, train, val, test, sample_counts, warnings)
    print("Done.")


if __name__ == "__main__":
    main()
