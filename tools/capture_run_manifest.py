#!/usr/bin/env python3
"""Capture a reproducibility manifest for a checkpoint folder.

Writes ``run_manifest.json`` recording repo state, environment, file hashes,
checkpoint contents, and the evaluation command needed to reproduce results.

Usage::

    uv run python tools/capture_run_manifest.py \
        --run_id run_d_specular_pps_weights_best \
        --weights_folder logs/c3vdv2_hf/stage2_specular_pps/models/weights_best \
        --hf_repo_id SmartWhatt/c3vdv2-SfM \
        --hf_split test \
        --training_script scripts/submit_specularnet_pps_hf.sh \
        --out reports/c3vdv2/run_d_manifest.json

If ``--out`` is omitted the manifest is written to ``<weights_folder>/run_manifest.json``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root: Path) -> dict:
    """Collect git commit, branch, and dirty-status from *repo_root*."""
    info: dict = {"commit": None, "branch": None, "status_short": None, "diff_hash": None}

    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except Exception:
        pass

    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            text=True,
        ).strip()
        info["status_short"] = status if status else "clean"
    except Exception:
        info["status_short"] = "unknown"

    if info["status_short"] not in (None, "clean", "unknown"):
        try:
            diff = subprocess.check_output(
                ["git", "diff"],
                cwd=str(repo_root),
                text=True,
            )
            info["diff_hash"] = hashlib.sha256(diff.encode()).hexdigest()
        except Exception:
            pass

    return info


def _env_info() -> dict:
    """Collect Python / platform / PyTorch info.

    If PyTorch or CUDA imports fail, record ``null`` and add an
    ``environment_warnings`` entry rather than failing.
    """
    info: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
    }
    # Ensure warnings list is typed correctly for mypy / type checkers
    warnings: list[str] = []

    try:
        import torch
        info["torch_version"] = torch.__version__
    except Exception as exc:
        info["torch_version"] = None
        warnings.append(f"torch import failed: {exc}")

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
    except Exception as exc:
        info["cuda_available"] = None
        warnings.append(f"CUDA check failed: {exc}")

    if warnings:
        info["environment_warnings"] = warnings

    return info


def _parse_script_env(script_path: str | None) -> dict:
    """Extract key config vars from a bash training script's env exports."""
    config: dict = {}

    if script_path is None or not os.path.isfile(script_path):
        return config

    with open(script_path) as f:
        text = f.read()

    # Parse `export KEY="${KEY:-default}"` lines
    import re
    for m in re.finditer(
        r'export\s+(\w+)=\"\$\{\w+:-(.*)\}\"',
        text,
    ):
        config[m.group(1)] = m.group(2)

    # Also handle non-default exports: export KEY=value
    for m in re.finditer(r'export\s+(\w+)="([^"]+)"', text):
        if m.group(1) not in config:
            config[m.group(1)] = m.group(2)

    return config


def _known_config(script_path: str | None, repo_root: Path) -> dict:
    """Build the ``known_config`` block for Run D."""
    env = _parse_script_env(script_path)

    return {
        "specular_net": _truthy(env.get("USE_SPECULAR_NET")),
        "specular_loss_weight": _as_float(env.get("SPECULAR_LOSS_WEIGHT")),
        "specular_threshold": _as_float(env.get("SPECULAR_THRESHOLD")),
        "pps_supervised_weight": _as_float(env.get("PPS_SUP_WEIGHT")),
        "pps_self_weight": _as_float(env.get("PPS_SELF_WEIGHT")),
        "pps_light_baseline_mm": _as_float(env.get("PPS_LIGHT_BASELINE_MM")),
        "s1_weights_path": env.get("S1_FINAL_WEIGHTS"),
        "s2_epochs": _as_int(env.get("S2_EPOCHS")),
        "s2_lr": _as_float(env.get("S2_LR")),
        "w_depth_start": _as_float(env.get("W_DEPTH_START")),
        "w_depth_end": _as_float(env.get("W_DEPTH_END")),
        "w_pose_start": _as_float(env.get("W_POSE_START")),
        "w_pose_end": _as_float(env.get("W_POSE_END")),
        "script_defaults": env,
    }


def _truthy(v: str | None) -> bool | None:
    if v is None:
        return None
    return v.strip() in ("1", "true", "True", "yes", "YES")


def _as_float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return v  # type: ignore[return-value]


def _as_int(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _find_weight_files(weights_folder: Path) -> dict[str, str]:
    """Return ``{basename: sha256}`` for every ``.pth`` file found."""
    hashes: dict[str, str] = {}
    for p in sorted(weights_folder.glob("*.pth")):
        hashes[p.name] = _sha256(p)
    return hashes


def capture(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    weights_folder = Path(args.weights_folder).resolve()

    if not weights_folder.is_dir():
        sys.exit(f"ERROR: weights_folder does not exist: {weights_folder}")

    manifest: dict = {
        "run_id": args.run_id,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo": _git_info(repo_root),
        "environment": _env_info(),
        "files": {},
        "data": {
            "hf_repo_id": args.hf_repo_id,
            "hf_split": args.hf_split,
            "hf_streaming": args.hf_streaming,
            "hf_cache_dir": args.hf_cache_dir,
        },
        "checkpoint": {
            "weights_folder": str(weights_folder),
            "weight_files": sorted(
                [p.name for p in weights_folder.glob("*.pth")]
            ),
            "weight_hashes": _find_weight_files(weights_folder),
        },
        "known_config": _known_config(args.training_script, repo_root),
        "evaluation_command": (
            f"uv run python tools/evaluate_c3vdv2.py "
            f"--data_source hf "
            f"--hf_repo_id {args.hf_repo_id} "
            f"--hf_split {args.hf_split} "
            f"--weights_folder {args.weights_folder}"
        ),
    }

    # File hashes for key source-of-truth files
    for label, rel_path in [
        ("pyproject.toml", "pyproject.toml"),
        ("uv.lock", "uv.lock"),
    ]:
        p = repo_root / rel_path
        if p.is_file():
            manifest["files"][label] = _sha256(p)

    if args.training_script:
        script_path = Path(args.training_script)
        if not script_path.is_absolute():
            script_path = repo_root / script_path
        if script_path.is_file():
            manifest["files"]["training_script"] = _sha256(script_path)

    out_path = args.out or str(weights_folder / "run_manifest.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Manifest written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture a reproducibility manifest for a checkpoint folder."
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--weights_folder", required=True)
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_split", default="test")
    parser.add_argument("--hf_streaming", action="store_true")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--training_script", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    capture(args)
