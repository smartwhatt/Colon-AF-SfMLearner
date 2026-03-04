"""
generate_splits.py
------------------
Scans a C3VDv2 data root and writes train/val/test split files
compatible with C3VDv2Dataset (format: "<sequence_name> <frame_idx>" per line).

Strategy:
  - Only uses Registered (pixel-GT) sequences (v1/v2/v3 videos).
  - Holds out 2 full colon geometries for val, 2 for test, rest for train.
  - Skips the first and last frame of every sequence (no valid triplet).
  - Optionally skips "debris" (v3) sequences for training if desired.

Usage:
    python generate_splits.py \
        --data_root /path/to/C3VDv2 \
        --output_dir splits/c3vdv2 \
        --val_segs c1_seg6 c1_seg7 \
        --test_segs c2_seg7 c2_seg8
"""

import argparse
import os
import random


def count_frames(seq_path: str) -> int:
    """Count RGB frames in a sequence directory."""
    rgb_dir = os.path.join(seq_path, "rgb")
    if not os.path.isdir(rgb_dir):
        return 0
    return len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])


def discover_sequences(data_root: str) -> list:
    """
    Return all sequence directory names that have both rgb/ and depth/ subdirs.
    C3VDv2 sequences are organised as <colon>/<segment>/<phantom>/<video>/
    We flatten to a single path string relative to data_root.
    """
    seqs = []
    for root, dirs, files in os.walk(data_root):
        dirs.sort()
        if "rgb" in dirs and "depth" in dirs and "pose.txt" in files:
            rel = os.path.relpath(root, data_root)
            n = count_frames(root)
            if n > 2:  # need at least 3 frames for a triplet
                seqs.append((rel, n))
    return seqs


def write_split(seqs: list, out_path: str, stride: int = 1):
    """Write a split file. stride > 1 sub-samples frames."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    for seq_name, n_frames in seqs:
        # Skip first and last frame (no valid prev/next)
        for idx in range(1, n_frames - 1, stride):
            lines.append(f"{seq_name} {idx}")
    random.shuffle(lines)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {out_path}: {len(lines)} samples from {len(seqs)} sequences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="splits/c3vdv2")
    parser.add_argument(
        "--val_segs",
        nargs="*",
        default=[],
        help="Sequence path fragments to assign to val set",
    )
    parser.add_argument(
        "--test_segs",
        nargs="*",
        default=[],
        help="Sequence path fragments to assign to test set",
    )
    parser.add_argument(
        "--skip_debris",
        action="store_true",
        help="Exclude v3 (debris) sequences from train/val",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=1,
        help="Sample every N frames for train (reduces dataset size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Discovering sequences...")
    all_seqs = discover_sequences(args.data_root)
    print(f"  Found {len(all_seqs)} sequences")

    # Partition
    train_seqs, val_seqs, test_seqs = [], [], []
    for seq_name, n in all_seqs:
        if args.skip_debris and "_v3" in seq_name:
            continue
        if any(seg in seq_name for seg in args.test_segs):
            test_seqs.append((seq_name, n))
        elif any(seg in seq_name for seg in args.val_segs):
            val_seqs.append((seq_name, n))
        else:
            train_seqs.append((seq_name, n))

    print(
        f"  Train: {len(train_seqs)} seqs | Val: {len(val_seqs)} | Test: {len(test_seqs)}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    write_split(
        train_seqs, os.path.join(args.output_dir, "train.txt"), stride=args.train_stride
    )
    write_split(val_seqs, os.path.join(args.output_dir, "val.txt"), stride=1)
    write_split(test_seqs, os.path.join(args.output_dir, "test.txt"), stride=1)

    print("Done.")


if __name__ == "__main__":
    main()
