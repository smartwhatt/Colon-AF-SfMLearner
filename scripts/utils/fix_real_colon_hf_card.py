#!/usr/bin/env python3
"""Patch the REAL-Colon HF dataset card split sizes.

The `datasets` library verifies cached split row counts against
`dataset_info.splits` in README frontmatter. If the card lists split names
without `num_examples`, non-streaming `load_dataset()` can fail with
NonMatchingSplitsSizesError after cache preparation. This helper writes the
known split counts into the card.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_COUNTS = {
    "train": {"num_examples": 201543, "num_bytes": 195536299901},
    "val": {"num_examples": 30525, "num_bytes": 35043892146},
    "test": {"num_examples": 18454, "num_bytes": 21911115823},
}


def _token_from_args(args: argparse.Namespace) -> str | None:
    if args.hf_token:
        return args.hf_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if args.prompt_token:
        return getpass.getpass("HF token: ").strip()
    return None


def _split_block(counts: dict[str, dict[str, int]]) -> str:
    lines = ["  splits:"]
    for split in ("train", "val", "test"):
        lines.append(f"  - name: {split}")
        lines.append(f"    num_examples: {counts[split]['num_examples']}")
        lines.append(f"    num_bytes: {counts[split]['num_bytes']}")
    return "\n".join(lines)


def _replace_split_block(readme: str, counts: dict[str, dict[str, int]]) -> str:
    if not readme.startswith("---\n"):
        raise ValueError("README does not start with YAML frontmatter")
    end = readme.find("\n---", 4)
    if end == -1:
        raise ValueError("README frontmatter closing marker not found")

    frontmatter = readme[4:end]
    body = readme[end:]

    lines = frontmatter.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        if lines[i] == "  splits:":
            out.extend(_split_block(counts).splitlines())
            replaced = True
            i += 1
            while i < len(lines):
                line = lines[i]
                if line.startswith("  - name: ") or line.startswith("    "):
                    i += 1
                    continue
                break
            continue
        out.append(lines[i])
        i += 1

    if not replaced:
        raise ValueError("Could not replace dataset_info.splits block")
    return "---\n" + "\n".join(out) + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="SmartWhatt/real-colon-SfM")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--prompt-token", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = _token_from_args(args)
    readme_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename="README.md",
        token=token,
    )
    readme = Path(readme_path).read_text(encoding="utf-8")
    updated = _replace_split_block(readme, DEFAULT_COUNTS)

    if args.dry_run:
        print(updated[: updated.find("\n---", 4) + 4])
        return

    if not token:
        raise SystemExit("HF token required for upload. Set HF_TOKEN or pass --prompt-token.")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(updated)
        tmp_path = f.name

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Fix dataset_info split sizes",
    )
    print(f"Updated README.md for {args.repo_id}")


if __name__ == "__main__":
    main()
