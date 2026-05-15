"""Tests for tools/capture_run_manifest.py — synthetic / temp-dir only, no GPU."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module under test so we can call its helpers directly.
from tools.capture_run_manifest import (
    _sha256,
    _git_info,
    _env_info,
    _find_weight_files,
    _known_config,
    capture,
)


# ---------------------------------------------------------------------------
# SHA256 hashing
# ---------------------------------------------------------------------------

def test_sha256_produces_stable_hash():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
        f.write(b"hello world")
        tmp = f.name
    try:
        h1 = _sha256(tmp)
        h2 = _sha256(tmp)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        os.unlink(tmp)


def test_sha256_different_content_different_hash():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
        f.write(b"aaa")
        tmp_a = f.name
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
        f.write(b"bbb")
        tmp_b = f.name
    try:
        assert _sha256(tmp_a) != _sha256(tmp_b)
    finally:
        os.unlink(tmp_a)
        os.unlink(tmp_b)


# ---------------------------------------------------------------------------
# Weight-file discovery
# ---------------------------------------------------------------------------

def test_find_weight_files_hashes_pth_files():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "encoder.pth").write_text("encoder-data")
        (root / "depth.pth").write_text("depth-data")
        (root / "notes.txt").write_text("not a weight")

        hashes = _find_weight_files(root)
        assert "encoder.pth" in hashes
        assert "depth.pth" in hashes
        assert "notes.txt" not in hashes
        assert len(hashes["encoder.pth"]) == 64


# ---------------------------------------------------------------------------
# Manifest capture (temp checkpoint)
# ---------------------------------------------------------------------------

def _make_dummy_pth(path: Path, name: str) -> None:
    (path / name).write_bytes(b"\x00" * 64)


def test_capture_writes_valid_json():
    repo_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        weights = Path(td) / "weights_best"
        weights.mkdir()
        _make_dummy_pth(weights, "encoder.pth")
        _make_dummy_pth(weights, "depth.pth")

        out = Path(td) / "manifest.json"

        # Build a minimal argparse-like namespace
        out_str = str(out)

        class Args:
            run_id = "test_run"
            weights_folder = str(weights)
            hf_repo_id = "test/repo"
            hf_split = "test"
            hf_streaming = False
            hf_cache_dir = None
            training_script = None
            out = out_str

        capture(Args())

        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["run_id"] == "test_run"
        assert data["checkpoint"]["weight_hashes"]["encoder.pth"]
        assert data["checkpoint"]["weight_hashes"]["depth.pth"]
        assert data["data"]["hf_repo_id"] == "test/repo"
        assert "captured_at" in data
        assert "repo" in data
        assert "environment" in data


def test_capture_missing_checkpoint_exits():
    with tempfile.TemporaryDirectory() as td:
        class Args:
            run_id = "test"
            weights_folder = os.path.join(td, "does_not_exist")
            hf_repo_id = None
            hf_split = "test"
            hf_streaming = False
            hf_cache_dir = None
            training_script = None
            out = None

        with pytest.raises(SystemExit) as exc:
            capture(Args())
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Git / environment info (smoke tests — always runnable)
# ---------------------------------------------------------------------------

def test_git_info_returns_dict_with_keys():
    repo_root = Path(__file__).resolve().parent.parent
    info = _git_info(repo_root)
    assert isinstance(info, dict)
    for key in ("commit", "branch", "status_short", "diff_hash"):
        assert key in info
    # commit should be non-None since we are in a repo
    assert info["commit"] is not None


def test_env_info_returns_dict():
    info = _env_info()
    assert "python_version" in info
    assert "platform" in info


# ---------------------------------------------------------------------------
# Known config parsing
# ---------------------------------------------------------------------------

def test_known_config_no_script():
    """When no script is provided, known_config returns defaults."""
    repo_root = Path(__file__).resolve().parent.parent
    cfg = _known_config(None, repo_root)
    assert cfg["specular_net"] is None
    assert "script_defaults" in cfg


def test_known_config_parses_exports():
    """Parse a synthetic bash script and check extracted values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(
            'export USE_SPECULAR_NET="1"\n'
            'export PPS_SUP_WEIGHT="${PPS_SUP_WEIGHT:-0.01}"\n'
            'export PPS_LIGHT_BASELINE_MM="${PPS_LIGHT_BASELINE_MM:-3.0}"\n'
        )
        script = f.name
    try:
        repo_root = Path(__file__).resolve().parent.parent
        cfg = _known_config(script, repo_root)
        assert cfg["specular_net"] is True
        assert cfg["pps_supervised_weight"] == 0.01
        assert cfg["pps_light_baseline_mm"] == 3.0
    finally:
        os.unlink(script)
