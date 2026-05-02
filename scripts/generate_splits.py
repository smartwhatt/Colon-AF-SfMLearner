"""
generate_splits.py
------------------
Scans a C3VDv2 data root and writes train/val/test split files compatible
with C3VDv2Dataset (format: "<sequence_name> <frame_idx>" per line).

Default strategy:
  - Only uses regional pixel-GT sequences; skips model, mold, full, and camera files.
  - Keeps every variant of a trajectory token in the same split.
  - Train: t1/t2 trajectories; val: t4 trajectories; test: t3 trajectories.
  - Skips the first and last frame of every sequence (no valid triplet).
  - Optionally skips v3/debris sequences.

Usage:
    python scripts/generate_splits.py \
        --data_root /path/to/C3VDv2 \
        --output_dir splits/c3vdv2

    python scripts/generate_splits.py --print_download_indices
"""

import argparse
import json
import os
import random
import re


REGIONAL_SEQUENCE_RE = re.compile(
    r"^(?P<colon>c[12])_"
    r"(?P<region>ascending|cecum|descending|rectum|sigmoid1|sigmoid2|sigmoid|transverse1|transverse2)_"
    r"(?P<trial>t[1-4])_"
    r"(?P<variant>v[1-4])$"
)

CURATED_DOWNLOAD_INDICES = {
    "train": [
        2, 3, 4, 5, 6,
        16, 17, 18, 19, 20, 21, 22,
        30, 31, 32, 33, 34,
        48, 49, 50, 51, 52, 53,
        63, 64, 65, 66, 67, 68, 69,
        79, 80, 81, 82, 83, 84,
        94, 95, 96, 97, 98, 99, 100,
        110, 111, 112, 113, 114, 115,
        124, 125, 126, 127, 128, 129, 130,
        139, 140, 141, 142, 143, 144,
        165, 166, 167, 168, 169, 170, 171,
        180, 181, 182, 183, 184, 185, 186,
        195, 196, 197, 198, 199, 200, 201,
        211, 212, 213, 214, 215, 216,
    ],
    "val": [
        10, 11, 12, 13,
        25, 26, 27,
        38, 39, 40, 41,
        57, 58, 59, 60,
        73, 74, 75, 76,
        88, 89, 90, 91,
        104, 105, 106, 107,
        119, 120, 121,
        134, 135, 136,
        148, 149, 150,
        156, 157, 158,
        175, 176, 177,
        190, 191, 192,
        206, 207, 208,
        220, 221,
    ],
    "test": [
        7, 8, 9,
        23, 24,
        35, 36, 37,
        54, 55, 56,
        70, 71, 72,
        85, 86, 87,
        101, 102, 103,
        116, 117, 118,
        131, 132, 133,
        145, 146, 147,
        153, 154, 155,
        172, 173, 174,
        187, 188, 189,
        202, 203, 204, 205,
        217, 218, 219,
    ],
    "metadata": [225],
}

# Approximate totals from the C3VDv2 metadata list printed by
# scripts/utils/download_c3vdv2.py. Values use the displayed MB values.
CURATED_DOWNLOAD_SIZE_MB = {
    "train": 309808.15,
    "val": 197478.04,
    "test": 158442.18,
    "metadata": 0.00,
}


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
    return sorted(seqs)


def sequence_stem(seq_name: str) -> str:
    """Return the C3VDv2 archive-like sequence stem from a relative path."""
    for part in reversed(seq_name.split(os.sep)):
        if REGIONAL_SEQUENCE_RE.match(part):
            return part
    return os.path.basename(seq_name)


def sequence_trial(seq_name: str) -> str | None:
    match = REGIONAL_SEQUENCE_RE.match(sequence_stem(seq_name))
    if match is None:
        return None
    return match.group("trial")


def is_debris_variant(seq_name: str) -> bool:
    return sequence_stem(seq_name).endswith("_v3")


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


def write_manifest(args, protocol: str, train_seqs: list, val_seqs: list, test_seqs: list):
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    total_size_mb = sum(CURATED_DOWNLOAD_SIZE_MB.values())
    manifest = {
        "protocol": protocol,
        "data_root": os.path.abspath(args.data_root),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "train_trials": args.train_trials,
        "val_trials": args.val_trials,
        "test_trials": args.test_trials,
        "train_stride": args.train_stride,
        "split_sequence_counts": {
            "train": len(train_seqs),
            "val": len(val_seqs),
            "test": len(test_seqs),
        },
        "sequences": {
            "train": [seq for seq, _ in train_seqs],
            "val": [seq for seq, _ in val_seqs],
            "test": [seq for seq, _ in test_seqs],
        },
        "download_indices": CURATED_DOWNLOAD_INDICES,
        "download_size_mb": CURATED_DOWNLOAD_SIZE_MB,
        "download_size_gb": {
            name: round(size_mb / 1024, 2)
            for name, size_mb in CURATED_DOWNLOAD_SIZE_MB.items()
        },
        "download_total_size_mb": round(total_size_mb, 2),
        "download_total_size_gb": round(total_size_mb / 1024, 2),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def print_download_indices():
    all_indices = (
        CURATED_DOWNLOAD_INDICES["train"]
        + CURATED_DOWNLOAD_INDICES["val"]
        + CURATED_DOWNLOAD_INDICES["test"]
        + CURATED_DOWNLOAD_INDICES["metadata"]
    )
    total_size_mb = sum(CURATED_DOWNLOAD_SIZE_MB.values())
    print("train:", " ".join(map(str, CURATED_DOWNLOAD_INDICES["train"])))
    print("val:", " ".join(map(str, CURATED_DOWNLOAD_INDICES["val"])))
    print("test:", " ".join(map(str, CURATED_DOWNLOAD_INDICES["test"])))
    print("metadata:", " ".join(map(str, CURATED_DOWNLOAD_INDICES["metadata"])))
    print("all:", " ".join(map(str, all_indices)))
    print()
    print("Approximate sizes from listed metadata MB values:")
    for split_name in ("train", "val", "test", "metadata"):
        size_mb = CURATED_DOWNLOAD_SIZE_MB[split_name]
        print(f"{split_name}: {size_mb:.2f} MB ({size_mb / 1024:.2f} GB)")
    print(f"all: {total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root")
    parser.add_argument("--output_dir", default="splits/c3vdv2")
    parser.add_argument(
        "--split_protocol",
        choices=["auto", "c3vdv2_curated", "fragments"],
        default="auto",
        help="auto uses fragments when --val_segs/--test_segs are set; otherwise c3vdv2_curated",
    )
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
        "--train_trials",
        nargs="+",
        default=["t1", "t2"],
        help="C3VDv2 trajectory trials assigned to train for c3vdv2_curated",
    )
    parser.add_argument(
        "--val_trials",
        nargs="+",
        default=["t4"],
        help="C3VDv2 trajectory trials assigned to val for c3vdv2_curated",
    )
    parser.add_argument(
        "--test_trials",
        nargs="+",
        default=["t3"],
        help="C3VDv2 trajectory trials assigned to test for c3vdv2_curated",
    )
    parser.add_argument(
        "--skip_debris",
        action="store_true",
        help="Exclude v3/debris sequences",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=1,
        help="Sample every N frames for train (reduces dataset size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--print_download_indices",
        action="store_true",
        help="Print archive metadata indices for the curated C3VDv2 regional plan",
    )
    args = parser.parse_args()

    if args.print_download_indices:
        print_download_indices()
        if not args.data_root:
            return

    if not args.data_root:
        parser.error("--data_root is required unless only --print_download_indices is used")

    random.seed(args.seed)

    print("Discovering sequences...")
    all_seqs = discover_sequences(args.data_root)
    print(f"  Found {len(all_seqs)} sequences")

    # Partition
    protocol = args.split_protocol
    if protocol == "auto":
        protocol = "fragments" if args.val_segs or args.test_segs else "c3vdv2_curated"

    train_seqs, val_seqs, test_seqs = [], [], []
    for seq_name, n in all_seqs:
        if args.skip_debris and is_debris_variant(seq_name):
            continue

        if protocol == "fragments":
            if any(seg in seq_name for seg in args.test_segs):
                test_seqs.append((seq_name, n))
            elif any(seg in seq_name for seg in args.val_segs):
                val_seqs.append((seq_name, n))
            else:
                train_seqs.append((seq_name, n))
        else:
            trial = sequence_trial(seq_name)
            if trial in args.test_trials:
                test_seqs.append((seq_name, n))
            elif trial in args.val_trials:
                val_seqs.append((seq_name, n))
            elif trial in args.train_trials:
                train_seqs.append((seq_name, n))

    if protocol == "c3vdv2_curated":
        missing = [
            split_name
            for split_name, seqs in (
                ("train", train_seqs),
                ("val", val_seqs),
                ("test", test_seqs),
            )
            if not seqs
        ]
        if missing:
            raise SystemExit(
                "No sequences found for split(s): {}. Check downloaded C3VDv2 regional zips.".format(
                    ", ".join(missing)
                )
            )

    print(
        f"  Protocol: {protocol}\n"
        f"  Train: {len(train_seqs)} seqs | Val: {len(val_seqs)} | Test: {len(test_seqs)}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    write_split(
        train_seqs, os.path.join(args.output_dir, "train.txt"), stride=args.train_stride
    )
    write_split(val_seqs, os.path.join(args.output_dir, "val.txt"), stride=1)
    write_split(test_seqs, os.path.join(args.output_dir, "test.txt"), stride=1)
    write_manifest(args, protocol, train_seqs, val_seqs, test_seqs)

    print("Done.")


if __name__ == "__main__":
    main()
