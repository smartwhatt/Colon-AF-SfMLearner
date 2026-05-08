#!/usr/bin/env python3
"""Download C3VDv2 from JHU DataVerse, convert to Parquet triplet shards,
and push to a public HuggingFace dataset repo for load_dataset(..., streaming=True).

No Cloudflare bypass needed — this instance is on the JHU allowed list.

Each Parquet row is one training sample: a (prev, curr, next) RGB triplet with
pre-undistorted 512×512 frames, ground-truth depth, occlusion mask, and relative poses.

Resume: already-uploaded shards are skipped (checked against HF file list at startup).
"""

import os
import re
import sys
import json
import shutil
import zipfile
import tempfile
import argparse
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

# Import dataset helpers from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.c3vdv2_dataset import (
    load_undistort_maps,
    apply_undistort,
    decode_depth_tiff,
    undistort_depth,
    load_poses,
    relative_pose,
    K_UNDISTORTED,
    UNDISTORT_WIDTH,
    UNDISTORT_HEIGHT,
)

DOI = "10.7281/T1/JC64MK"
BASE_URL = "https://archive.data.jhu.edu"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}

REGIONAL_SEQUENCE_RE = re.compile(
    r"^(?P<colon>c[12])_"
    r"(?P<region>ascending|cecum|descending|rectum|sigmoid1|sigmoid2|sigmoid|transverse1|transverse2)_"
    r"(?P<trial>t[1-4])_"
    r"(?P<variant>v[1-4])$"
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
- depth
- pose
- medical
---

# C3VDv2 — Colonoscopy 3D Video Dataset v2

This dataset is a re-packaged version of **C3VDv2** originally published by
Johns Hopkins University, distributed under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

**Original dataset DOI:** https://doi.org/10.7281/T1/JC64MK
**Dataset archive:** https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi:10.7281/T1/JC64MK

## Attribution

This re-packaged version was created to facilitate streaming access. The original
data and all intellectual property rights belong to the original authors and
Johns Hopkins University. When using this dataset, you must comply with the
CC BY 4.0 license terms, which require attribution to the original creators.

Please cite the original work (check the dataset page above for the full citation):

```bibtex
@dataset{{c3vdv2,
  title  = {{C3VDv2: Colonoscopy 3D Video Dataset v2}},
  doi    = {{10.7281/T1/JC64MK}},
  url    = {{https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi:10.7281/T1/JC64MK}},
  publisher = {{Johns Hopkins University Data Archive}},
}}
```

## Re-packaging

Raw omnidirectional fisheye frames (1350×1080) are converted to undistorted
perspective crops (512×512) using the Scaramuzza camera model from
`camera_intrinsics.txt`. Each **Parquet row** is one training sample:

| column | dtype | description |
|--------|-------|-------------|
| `sequence` | string | sequence name, e.g. `c1_ascending_t1_v1` |
| `frame_idx` | int32 | centre frame index |
| `rgb_prev` | binary | PNG bytes, undistorted 512×512 |
| `rgb_curr` | binary | PNG bytes, undistorted 512×512 |
| `rgb_next` | binary | PNG bytes, undistorted 512×512 |
| `depth` | binary | float32 `.tobytes()`, 512×512, metric metres |
| `occlusion` | binary | uint8 `.tobytes()`, 512×512 binary mask (1 = occluded) |
| `pose_curr2prev` | list[float64] | 16-value row-major 4×4 relative pose |
| `pose_curr2next` | list[float64] | 16-value row-major 4×4 relative pose |
| `K` | list[float32] | 9-value row-major 3×3 normalised camera intrinsics |

## Splits

| split | trials | description |
|-------|--------|-------------|
| train | t1, t2 | two trajectories per region |
| val   | t4     | held-out trajectory |
| test  | t3     | held-out trajectory |

## Usage

```python
from datasets import load_dataset
import numpy as np
from PIL import Image
import io

ds = load_dataset("{repo_id}", split="train", streaming=True)

for row in ds:
    rgb = np.array(Image.open(io.BytesIO(row["rgb_curr"])))            # 512×512×3 uint8
    depth = np.frombuffer(row["depth"], np.float32).reshape(512, 512)  # metres
    occ = np.frombuffer(row["occlusion"], np.uint8).reshape(512, 512)  # binary mask
    T_curr2prev = np.array(row["pose_curr2prev"]).reshape(4, 4)
    K = np.array(row["K"]).reshape(3, 3)
```
"""


# ---------------------------------------------------------------------------
# JHU DataVerse helpers
# ---------------------------------------------------------------------------

def fetch_metadata():
    url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_files(metadata):
    try:
        return metadata["data"]["latestVersion"]["files"]
    except KeyError:
        print("ERROR: Unexpected metadata format")
        sys.exit(1)


def fmt_size(n):
    return f"{n / 1024 / 1024:.2f} MB"


def download_zip(file_id, filename, expected_size, tmp_dir):
    """Download zip to tmp_dir with byte-range resume. Returns local path or None."""
    out = os.path.join(tmp_dir, filename)
    existing = os.path.getsize(out) if os.path.exists(out) else 0

    if existing == expected_size:
        print(f"  [{filename}] already downloaded, skipping.")
        return out

    hdrs = dict(HEADERS)
    if existing > 0:
        hdrs["Range"] = f"bytes={existing}-"
        print(f"  [{filename}] resuming from {existing / 1024**2:.1f} MB")

    r = requests.get(
        f"{BASE_URL}/api/access/datafile/{file_id}",
        headers=hdrs, stream=True, timeout=3600,
    )
    if r.status_code not in (200, 206):
        print(f"  ERROR: HTTP {r.status_code} for {filename}")
        return None

    mode = "ab" if r.status_code == 206 else "wb"
    done = existing if r.status_code == 206 else 0

    with tqdm(total=expected_size, initial=done, unit="B", unit_scale=True,
              unit_divisor=1024, desc=filename[:50], ncols=100) as pbar:
        with open(out, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    return out


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def build_split_map_from_files(splits_dir):
    """Read train/val/test.txt → {seq_name: split_name}."""
    split_map = {}
    for split_name in ("train", "val", "test"):
        txt = os.path.join(splits_dir, f"{split_name}.txt")
        if not os.path.exists(txt):
            continue
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    seq = parts[0]
                    split_map[os.path.basename(seq)] = split_name
                    split_map[seq] = split_name
    return split_map


def assign_split(seq_name, split_map, train_trials, val_trials, test_trials):
    leaf = os.path.basename(seq_name)
    for key in (seq_name, leaf):
        if key in split_map:
            return split_map[key]
    m = REGIONAL_SEQUENCE_RE.match(leaf)
    if m:
        trial = m.group("trial")
        if trial in train_trials:
            return "train"
        if trial in val_trials:
            return "val"
        if trial in test_trials:
            return "test"
    return None


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def discover_sequences(extract_dir):
    """Find all sequence dirs (rgb/ + depth/ + pose.txt) under extract_dir."""
    seqs = []
    for root, dirs, files in os.walk(extract_dir):
        dirs.sort()
        if "rgb" in dirs and "depth" in dirs and "pose.txt" in files:
            seqs.append(root)
    return sorted(seqs)


def sorted_frame_indices(seq_dir):
    """Return sorted list of integer frame indices from rgb/ filenames."""
    rgb_dir = os.path.join(seq_dir, "rgb")
    indices = []
    for fn in os.listdir(rgb_dir):
        if fn.endswith(".png"):
            try:
                indices.append(int(os.path.splitext(fn)[0]))
            except ValueError:
                pass
    return sorted(indices)


# ---------------------------------------------------------------------------
# Frame loading helpers
# ---------------------------------------------------------------------------

def _encode_png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return bytes(buf)


def _load_rgb(seq_dir, idx, map1, map2):
    path = os.path.join(seq_dir, "rgb", f"{idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return apply_undistort(img, map1, map2)


def _load_depth(seq_dir, idx, map1, map2):
    path = os.path.join(seq_dir, "depth", f"{idx:04d}_depth.tiff")
    return undistort_depth(decode_depth_tiff(path), map1, map2)


def _load_occlusion(seq_dir, idx, map1, map2):
    base = os.path.join(seq_dir, "occlusions", f"{idx:04d}_occlusion")
    for ext in (".tiff", ".png"):
        candidate = base + ext
        if os.path.exists(candidate):
            occ = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            if occ is not None:
                occ = cv2.remap(occ, map1, map2, interpolation=cv2.INTER_NEAREST)
                return (occ == 255).astype(np.uint8)
    return np.zeros((UNDISTORT_HEIGHT, UNDISTORT_WIDTH), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Parquet schema + writing
# ---------------------------------------------------------------------------

_K_NORM = np.array(
    [
        [K_UNDISTORTED[0, 0] / UNDISTORT_WIDTH,  0,  K_UNDISTORTED[0, 2] / UNDISTORT_WIDTH],
        [0,  K_UNDISTORTED[1, 1] / UNDISTORT_HEIGHT,  K_UNDISTORTED[1, 2] / UNDISTORT_HEIGHT],
        [0,  0,  1],
    ],
    dtype=np.float32,
).flatten().tolist()

PARQUET_SCHEMA = pa.schema([
    pa.field("sequence",       pa.string()),
    pa.field("frame_idx",      pa.int32()),
    pa.field("rgb_prev",       pa.binary()),
    pa.field("rgb_curr",       pa.binary()),
    pa.field("rgb_next",       pa.binary()),
    pa.field("depth",          pa.binary()),
    pa.field("occlusion",      pa.binary()),
    pa.field("pose_curr2prev", pa.list_(pa.float64())),
    pa.field("pose_curr2next", pa.list_(pa.float64())),
    pa.field("K",              pa.list_(pa.float32())),
])


def write_parquet(rows, path):
    cols = {k: [r[k] for r in rows] for k in PARQUET_SCHEMA.names}
    pq.write_table(pa.table(cols, schema=PARQUET_SCHEMA), path, compression="snappy")


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(seq_dir, seq_name, split, map1, map2,
                     shard_size, repo_id, api, existing_hf, tmp_dir):
    """Convert one sequence to Parquet shards and upload to HF. Returns True on success."""
    frame_indices = sorted_frame_indices(seq_dir)
    n_frames = len(frame_indices)
    if n_frames < 3:
        print(f"    WARNING: {seq_name} has {n_frames} frames — skipping")
        return True

    try:
        poses = load_poses(os.path.join(seq_dir, "pose.txt"))
    except Exception as e:
        print(f"    ERROR loading poses for {seq_name}: {e}")
        return False

    # Frame index → position mapping for O(1) prev/next lookup
    idx_to_pos = {fi: pos for pos, fi in enumerate(frame_indices)}

    valid_centers = frame_indices[1:-1]  # skip first and last (no valid triplet)
    n_shards = (len(valid_centers) + shard_size - 1) // shard_size

    print(f"    {seq_name} → {split} ({n_frames} frames, {n_shards} shards)")

    for shard_n in range(n_shards):
        shard_path_hf = f"data/{split}/{seq_name}/shard_{shard_n:04d}.parquet"
        if shard_path_hf in existing_hf:
            tqdm.write(f"      shard_{shard_n:04d} already on HF, skipping")
            continue

        chunk = valid_centers[shard_n * shard_size : (shard_n + 1) * shard_size]
        rows = []

        for idx_curr in tqdm(chunk, desc=f"      shard_{shard_n:04d}", ncols=80, leave=False):
            pos = idx_to_pos[idx_curr]
            idx_prev = frame_indices[pos - 1]
            idx_next = frame_indices[pos + 1]

            try:
                rgb_prev = _encode_png(_load_rgb(seq_dir, idx_prev, map1, map2))
                rgb_curr = _encode_png(_load_rgb(seq_dir, idx_curr, map1, map2))
                rgb_next = _encode_png(_load_rgb(seq_dir, idx_next, map1, map2))
                depth = _load_depth(seq_dir, idx_curr, map1, map2).tobytes()
                occ = _load_occlusion(seq_dir, idx_curr, map1, map2).tobytes()
            except FileNotFoundError as exc:
                tqdm.write(f"      WARNING: missing file, skipping frame {idx_curr}: {exc}")
                continue

            # Clamp pose indices to valid range
            pi_curr = min(idx_curr, len(poses) - 1)
            pi_prev = min(idx_prev, len(poses) - 1)
            pi_next = min(idx_next, len(poses) - 1)
            T_c2p = relative_pose(poses[pi_curr], poses[pi_prev]).flatten().tolist()
            T_c2n = relative_pose(poses[pi_curr], poses[pi_next]).flatten().tolist()

            rows.append({
                "sequence":       seq_name,
                "frame_idx":      idx_curr,
                "rgb_prev":       rgb_prev,
                "rgb_curr":       rgb_curr,
                "rgb_next":       rgb_next,
                "depth":          depth,
                "occlusion":      occ,
                "pose_curr2prev": T_c2p,
                "pose_curr2next": T_c2n,
                "K":              _K_NORM,
            })

        if not rows:
            continue

        local_shard = os.path.join(tmp_dir, f"_shard_{shard_n:04d}.parquet")
        write_parquet(rows, local_shard)
        try:
            api.upload_file(
                path_or_fileobj=local_shard,
                path_in_repo=shard_path_hf,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {seq_name} shard {shard_n:04d}",
            )
        finally:
            if os.path.exists(local_shard):
                os.remove(local_shard)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download C3VDv2, convert to Parquet triplet shards, push to HuggingFace"
    )
    parser.add_argument("repo_id", help="HF dataset repo, e.g. username/c3vdv2")
    parser.add_argument("--undistort_maps", required=True,
                        help="Dir with undistort_map1.npy / undistort_map2.npy")
    parser.add_argument("--hf-token", default=None,
                        help="HF token (default: ~/.cache/huggingface/token or HF_TOKEN env)")
    parser.add_argument("--tmp-dir", default=None,
                        help="Temp dir for downloads/extraction (default: system temp)")
    parser.add_argument("--shard-size", type=int, default=500,
                        help="Rows per Parquet shard (default: 500)")
    parser.add_argument("--indices", nargs="*", type=int, default=None,
                        help="File indices to process (skip interactive menu)")
    parser.add_argument("--splits-dir", default=None,
                        help="Dir with train.txt/val.txt/test.txt for split assignment")
    parser.add_argument("--train-trials", nargs="+", default=["t1", "t2"])
    parser.add_argument("--val-trials",   nargs="+", default=["t4"])
    parser.add_argument("--test-trials",  nargs="+", default=["t3"])
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # ---- Metadata & file selection ----------------------------------------
    print("Fetching C3VDv2 metadata from JHU DataVerse...")
    metadata = fetch_metadata()
    files = get_files(metadata)

    total_size = sum(f["dataFile"]["filesize"] for f in files)
    print("\nAvailable files:")
    print("-" * 100)
    for i, f in enumerate(files):
        print(f"[{i:3d}] {f['label']:<60} {fmt_size(f['dataFile']['filesize'])}")
    print("-" * 100)
    print(f"Total: {total_size / 1024**3:.2f} GB\n")

    if args.indices is not None:
        selected = args.indices
    else:
        print("Options:  1) All  2) First N  3) Manual")
        choice = input("Choose [1-3]: ").strip()
        if choice == "1":
            selected = list(range(len(files)))
        elif choice == "2":
            N = int(input("Enter N: "))
            selected = list(range(min(N, len(files))))
        elif choice == "3":
            selected = [int(x) for x in input("Indices (space-separated): ").split()]
        else:
            print("Invalid choice")
            sys.exit(1)

    # ---- HF setup ---------------------------------------------------------
    print(f"\nCreating/verifying HF repo: {args.repo_id}")
    api = HfApi(token=hf_token)
    create_repo(args.repo_id, repo_type="dataset", private=False,
                exist_ok=True, token=hf_token)

    split_map = {}
    if args.splits_dir:
        split_map = build_split_map_from_files(args.splits_dir)
        print(f"Split map loaded from {args.splits_dir} ({len(split_map)} entries)")

    print("Querying existing HF files (for resume)...")
    try:
        existing_hf = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    except Exception:
        existing_hf = set()
    print(f"  {len(existing_hf)} files already present")

    if "README.md" not in existing_hf:
        readme = README_TEMPLATE.format(repo_id=args.repo_id)
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add README with CC BY 4.0 attribution",
        )
        print("  Uploaded README.md")

    # ---- Undistortion maps ------------------------------------------------
    print(f"\nLoading undistortion maps from {args.undistort_maps}...")
    map1, map2 = load_undistort_maps(args.undistort_maps)
    print(f"  Maps loaded (output: {UNDISTORT_WIDTH}×{UNDISTORT_HEIGHT})")

    # ---- Temp dir ---------------------------------------------------------
    own_tmp = args.tmp_dir is None
    tmp_base = tempfile.mkdtemp(prefix="c3vdv2_hf_") if own_tmp else args.tmp_dir
    os.makedirs(tmp_base, exist_ok=True)
    print(f"Temp dir: {tmp_base}\n")

    total = len(selected)
    failed = []

    try:
        for n, i in enumerate(selected, 1):
            f = files[i]
            file_id = f["dataFile"]["id"]
            filename = f["label"]
            expected_size = f["dataFile"]["filesize"]

            print(f"\n[{n}/{total}] {filename} ({fmt_size(expected_size)})")

            zip_path = download_zip(file_id, filename, expected_size, tmp_base)
            if zip_path is None:
                failed.append(filename)
                continue

            extract_dir = os.path.join(tmp_base, filename.replace(".zip", "_extracted"))
            if not os.path.isdir(extract_dir):
                print("  Extracting...")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_dir)
                except Exception as e:
                    print(f"  ERROR extracting: {e}")
                    failed.append(filename)
                    continue
            else:
                print("  Already extracted.")

            seqs = discover_sequences(extract_dir)
            if not seqs:
                print(f"  WARNING: no sequences found in {filename}")

            zip_ok = True
            for seq_dir in seqs:
                seq_name = os.path.basename(seq_dir)
                split = assign_split(
                    seq_name, split_map,
                    args.train_trials, args.val_trials, args.test_trials,
                )
                if split is None:
                    print(f"  WARNING: {seq_name} has no split assignment — skipping")
                    continue

                ok = process_sequence(
                    seq_dir, seq_name, split, map1, map2,
                    args.shard_size, args.repo_id, api, existing_hf, tmp_base,
                )
                if not ok:
                    zip_ok = False

            if zip_ok:
                shutil.rmtree(extract_dir, ignore_errors=True)
                os.remove(zip_path)
            else:
                failed.append(filename)

    finally:
        if own_tmp and os.path.isdir(tmp_base):
            shutil.rmtree(tmp_base, ignore_errors=True)

    print("\n" + "=" * 100)
    print(f"Done. {total - len(failed)}/{total} zips processed successfully.")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
