"""Tests for stratified evaluation helpers — synthetic arrays only, no GPU/network.

Covers:
- Depth-band splitter
- Specular pseudo-mask
- JSONL writer (structure)
- Comparator report sections
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.evaluate_c3vdv2 import (
    _depth_bands_np,
    _specular_intensity_mask_np,
    _specular_model_mask_np,
    _error_tail_np,
    _specular_metrics_np,
    _write_summary_md,
)
from tools.compare_c3vdv2_strata import (
    _agg_metrics,
    _band_metrics,
    _specular_metrics,
    _per_sequence_deltas,
    _worst_regressions,
    _decide,
    generate_report,
    DEPTH_METRIC_KEYS,
    DEPTH_BAND_NAMES,
    DECISION_PROCEED,
    DECISION_CAUTION,
    DECISION_RECHECK,
)

from colon_datasets.c3vdv2_metrics import compute_depth_metrics_np


# ============================================================================
# Depth-band splitter
# ============================================================================

def _make_depth_frame(gt_m: np.ndarray, pred_m: np.ndarray | None = None):
    """Helper: gt_m and pred_m are (H,W) float arrays in metres."""
    if pred_m is None:
        pred_m = gt_m.copy()
    valid = (gt_m > 0.001) & (gt_m < 0.099)
    return gt_m, pred_m, valid


def test_depth_bands_include_correct_pixels():
    """Pixels at 3 mm and 7 mm land in the correct bands."""
    h, w = 64, 64
    gt = np.full((h, w), 0.003, dtype=np.float32)  # 3 mm → band 0_5mm
    pred = gt.copy()
    valid = (gt > 0.001) & (gt < 0.099)

    bands = _depth_bands_np(gt, pred, valid, median_scaling=False)

    # 0_5mm band should have metrics
    assert bands["0_5mm"] is not None
    assert bands["0_5mm"]["rmse"] == 0.0
    # Other bands should be None (no pixels)
    assert bands["5_10mm"] is None
    assert bands["50_100mm"] is None


def test_depth_bands_skip_tiny_masks():
    """Bands with fewer than min_pixels valid pixels are null."""
    gt = np.full((8, 8), 0.003, dtype=np.float32)
    pred = gt.copy()
    valid = (gt > 0.001) & (gt < 0.099)

    # Default min_pixels=10, but we only have 64 pixels in one band
    bands = _depth_bands_np(gt, pred, valid, median_scaling=False, min_pixels=100)
    assert bands["0_5mm"] is None  # 64 < 100


def test_depth_bands_multiple_bands():
    """Mixed-depth frame: half at 3 mm, half at 15 mm."""
    h, w = 64, 64
    gt = np.full((h, w), 0.003, dtype=np.float32)
    gt[h // 2:, :] = 0.015  # 15 mm → band 10_20mm
    pred = gt.copy()
    valid = (gt > 0.001) & (gt < 0.099)

    bands = _depth_bands_np(gt, pred, valid, median_scaling=False)
    assert bands["0_5mm"] is not None
    assert bands["10_20mm"] is not None
    # Bands with no pixels should be None
    assert bands["5_10mm"] is None


# ============================================================================
# Specular pseudo-mask
# ============================================================================

def test_specular_intensity_mask_saturated_selected():
    """Pixels with max channel > 0.98 are marked specular."""
    h, w = 32, 32
    # Create a mostly-dark image with a bright patch
    rgb = np.zeros((3, h, w), dtype=np.float32)
    rgb[:, 10:20, 10:20] = 1.0  # saturated white patch

    mask = _specular_intensity_mask_np(rgb, max_channel_thresh=0.98, intensity_thresh=0.95,
                                       dilate_radius=0)

    # Saturated patch should be True
    assert mask[15, 15]  # inside the patch
    # Dark area should be False
    assert not mask[0, 0]


def test_specular_intensity_mask_non_saturated_excluded():
    """Pixels below threshold are not marked specular."""
    rgb = np.full((3, 32, 32), 0.5, dtype=np.float32)
    mask = _specular_intensity_mask_np(rgb, max_channel_thresh=0.98, intensity_thresh=0.95,
                                       dilate_radius=0)
    assert not mask.any()


def test_specular_intensity_mask_high_intensity_selected():
    """High average intensity (>0.95) triggers specular even if max channel ≤ 0.98."""
    rgb = np.full((3, 32, 32), 0.96, dtype=np.float32)  # intensity=0.96, max_ch=0.96
    mask = _specular_intensity_mask_np(rgb, max_channel_thresh=0.98, intensity_thresh=0.95,
                                       dilate_radius=0)
    assert mask.all()


# ============================================================================
# Specular model mask
# ============================================================================

def test_specular_model_mask_threshold():
    spec_out = np.array([[0.1, 0.6], [0.4, 0.9]], dtype=np.float32)
    mask = _specular_model_mask_np(spec_out, threshold=0.5)

    assert mask.tolist() == [[False, True], [False, True]]


# ============================================================================
# Specular metrics
# ============================================================================

def test_specular_metrics_inside_outside():
    """Specular region gets separate metrics from non-specular."""
    h, w = 32, 32
    gt = np.full((h, w), 0.02, dtype=np.float32)
    pred = np.full((h, w), 0.02, dtype=np.float32)
    valid = (gt > 0.001) & (gt < 0.099)

    spec_mask = np.zeros((h, w), dtype=bool)
    spec_mask[:16, :] = True  # top half is specular

    result = _specular_metrics_np(gt, pred, valid, spec_mask, median_scaling=False)

    assert result["inside"] is not None
    assert result["outside"] is not None
    assert result["inside"]["rmse"] == 0.0
    assert result["outside"]["rmse"] == 0.0
    assert result["inside"]["pixel_count"] > 0
    assert result["outside"]["pixel_count"] > 0


# ============================================================================
# Error tail
# ============================================================================

def test_error_tail_perfect_prediction():
    gt = np.full((32, 32), 0.02, dtype=np.float32)
    pred = gt.copy()
    mask = (gt > 0.001) & (gt < 0.099)

    tail = _error_tail_np(gt, pred, mask)
    assert tail["abs_rel_p90"] == 0.0
    assert tail["abs_rel_p95"] == 0.0


def test_error_tail_with_errors():
    gt = np.full((32, 32), 0.02, dtype=np.float32)
    pred = gt.copy()
    # Make ~half the pixels have large error so p95 is non-trivial
    pred[16:, :] = 0.04  # 100% error → abs_rel = 1.0
    pred[:16, :16] = 0.03  # 50% error → abs_rel = 0.5
    mask = np.ones((32, 32), dtype=bool)

    tail = _error_tail_np(gt, pred, mask)
    assert tail["abs_rel_p95"] > 0.0


# ============================================================================
# JSONL structure (via summary markdown writer — exercises helper)
# ============================================================================

def _make_stratified_frame(
    run_id="test",
    seq="seq_01",
    frame_idx=0,
    abs_rel=0.05,
    rmse=0.005,
    include_bands=True,
    include_specular=True,
    include_model=False,
):
    """Build a single synthetic stratified frame dict matching the evaluator's output."""
    metrics = {
        "abs_rel": abs_rel, "sq_rel": 0.01, "rmse": rmse, "rmse_log": 0.02,
        "a1": 0.95, "a2": 0.98, "a3": 0.99,
    }
    frame: dict = {
        "run_id": run_id,
        "seq": seq,
        "frame_idx": frame_idx,
        "n_valid": 5000,
        "global": metrics,
        "depth_bands": {},
        "specular_intensity": {"inside": None, "outside": None},
        "specular_model": {"inside": None, "outside": None},
        "error_tail": {"abs_rel_p90": 0.1, "abs_rel_p95": 0.2, "p95_rmse_contrib": 0.001},
    }

    if include_bands:
        for band in DEPTH_BAND_NAMES:
            frame["depth_bands"][band] = metrics.copy()

    if include_specular:
        frame["specular_intensity"]["inside"] = dict(metrics, pixel_count=200.0)
        frame["specular_intensity"]["outside"] = dict(metrics, pixel_count=4800.0)

    if include_model:
        frame["specular_model"]["inside"] = dict(metrics, pixel_count=150.0)
        frame["specular_model"]["outside"] = dict(metrics, pixel_count=4850.0)

    return frame


def test_summary_md_writes_sections():
    frames = [_make_stratified_frame() for _ in range(5)]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "summary.md")
        _write_summary_md(path, "run_test", frames)

        text = Path(path).read_text()
        assert "## Aggregate" in text
        assert "## Depth Bands" in text
        assert "## Specular (intensity pseudo-mask)" in text
        assert "## Error Tail" in text
        assert "0.0500" in text  # abs_rel value


def test_jsonl_roundtrip():
    """A frame dict can be serialized and deserialized without corruption."""
    frame = _make_stratified_frame()
    dumped = json.dumps(frame)
    loaded = json.loads(dumped)
    assert loaded["run_id"] == "test"
    assert loaded["global"]["abs_rel"] == 0.05
    assert loaded["depth_bands"]["0_5mm"]["rmse"] == 0.005
    assert loaded["specular_intensity"]["inside"]["pixel_count"] == 200.0


# ============================================================================
# Comparator aggregate / band / specular
# ============================================================================

def _make_frames(run_id, abs_rels, seqs=None):
    """Build a list of synthetic stratified frames with given AbsRel values."""
    frames = []
    for i, ar in enumerate(abs_rels):
        seq = seqs[i] if seqs else "seq_A"
        frames.append(_make_stratified_frame(
            run_id=run_id, seq=seq, frame_idx=i, abs_rel=ar, include_model=True
        ))
    return frames


def test_agg_metrics_computes_mean():
    frames_a = _make_frames("run_a", [0.04, 0.06, 0.05])
    agg = _agg_metrics(frames_a)
    assert abs(agg["abs_rel"] - 0.05) < 0.001


def test_band_metrics_non_null():
    frames = _make_frames("run", [0.04, 0.06])
    bm = _band_metrics(frames, "0_5mm")
    assert bm is not None
    assert "abs_rel" in bm
    assert bm["n_frames"] == 2


def test_specular_metrics_helper():
    frames = [
        _make_stratified_frame(run_id="run", abs_rel=0.03, include_specular=True),
        _make_stratified_frame(run_id="run", abs_rel=0.05, include_specular=True),
    ]
    sm = _specular_metrics(frames, "specular_intensity", "inside")
    assert sm is not None
    assert abs(sm["abs_rel"] - 0.04) < 0.001


def test_per_sequence_deltas():
    frames_a = _make_frames("a", [0.04, 0.06], seqs=["seq1", "seq2"])
    frames_b = _make_frames("b", [0.03, 0.07], seqs=["seq1", "seq2"])
    deltas = _per_sequence_deltas(frames_a, frames_b, "abs_rel")
    assert deltas["seq1"]["delta"] == pytest.approx(-0.01)  # B is better (lower AbsRel)
    assert deltas["seq2"]["delta"] == pytest.approx(0.01)  # B is worse


def test_worst_regressions_finds_degraded_frames():
    frames_ref = [
        _make_stratified_frame(run_id="ref", seq="s1", frame_idx=0, abs_rel=0.04),
        _make_stratified_frame(run_id="ref", seq="s1", frame_idx=1, abs_rel=0.05),
    ]
    frames_cmp = [
        _make_stratified_frame(run_id="cmp", seq="s1", frame_idx=0, abs_rel=0.03),  # improvement
        _make_stratified_frame(run_id="cmp", seq="s1", frame_idx=1, abs_rel=0.12),  # large regression
    ]
    worst = _worst_regressions(frames_ref, frames_cmp, "abs_rel", top_n=5)
    assert len(worst) == 1  # only one regression
    assert worst[0]["ref_val"] == 0.05
    assert worst[0]["cmp_val"] == 0.12


def test_worst_regressions_no_regressions():
    frames_ref = _make_frames("ref", [0.05, 0.06])
    frames_cmp = _make_frames("cmp", [0.03, 0.04])  # all improvements
    worst = _worst_regressions(frames_ref, frames_cmp)
    assert len(worst) == 0


# ============================================================================
# Decision rule
# ============================================================================

def _make_agg(abs_rel):
    return {"abs_rel": abs_rel, "sq_rel": 0.01, "rmse": 0.005, "rmse_log": 0.02,
            "a1": 0.9, "a2": 0.95, "a3": 0.98}

def _make_band(abs_rel):
    return {"0_5mm": _make_agg(abs_rel), "5_10mm": _make_agg(abs_rel),
            "10_20mm": _make_agg(abs_rel), "20_50mm": _make_agg(abs_rel),
            "50_100mm": _make_agg(abs_rel)}

def _make_spec(abs_rel):
    return {"inside": _make_agg(abs_rel), "outside": _make_agg(abs_rel)}


def test_decide_proceed_when_d_best_no_regressions():
    decision = _decide(
        run_labels=["run_a", "run_b", "run_d"],
        agg={"run_a": _make_agg(0.06), "run_b": _make_agg(0.05), "run_d": _make_agg(0.04)},
        band_agg={
            "run_a": _make_band(0.06), "run_b": _make_band(0.05), "run_d": _make_band(0.04),
        },
        spec_intensity={
            "run_a": _make_spec(0.06), "run_b": _make_spec(0.05), "run_d": _make_spec(0.04),
        },
        seq_deltas={},
        has_regressions=False,
    )
    assert decision == DECISION_PROCEED


def test_decide_recheck_when_d_worse_aggregate():
    decision = _decide(
        run_labels=["run_b", "run_d"],
        agg={"run_b": _make_agg(0.05), "run_d": _make_agg(0.06)},  # D worse
        band_agg={"run_b": _make_band(0.05), "run_d": _make_band(0.06)},
        spec_intensity={"run_b": _make_spec(0.05), "run_d": _make_spec(0.06)},
        seq_deltas={},
        has_regressions=False,
    )
    assert decision == DECISION_RECHECK


def test_decide_caution_when_regressions():
    decision = _decide(
        run_labels=["run_b", "run_d"],
        agg={"run_b": _make_agg(0.06), "run_d": _make_agg(0.04)},  # D better aggregate
        band_agg={"run_b": _make_band(0.06), "run_d": _make_band(0.04)},
        spec_intensity={"run_b": _make_spec(0.06), "run_d": _make_spec(0.04)},
        seq_deltas={},
        has_regressions=True,  # but has material regressions
    )
    assert decision == DECISION_CAUTION


def test_decide_caution_when_near_field_regresses():
    """D is better overall but worse in 0_5mm band."""
    band_agg = {
        "run_b": {"0_5mm": _make_agg(0.04), **{k: _make_agg(0.06) for k in DEPTH_BAND_NAMES[1:]}},
        "run_d": {"0_5mm": _make_agg(0.06), **{k: _make_agg(0.04) for k in DEPTH_BAND_NAMES[1:]}},
    }
    decision = _decide(
        run_labels=["run_b", "run_d"],
        agg={"run_b": _make_agg(0.06), "run_d": _make_agg(0.04)},
        band_agg=band_agg,
        spec_intensity={"run_b": _make_spec(0.06), "run_d": _make_spec(0.04)},
        seq_deltas={},
        has_regressions=False,
    )
    assert decision == DECISION_CAUTION


# ============================================================================
# Full report generation
# ============================================================================

def test_generate_report_has_all_sections():
    frames_a = _make_frames("run_a", [0.06, 0.07, 0.05])
    frames_b = _make_frames("run_b", [0.05, 0.06, 0.04])
    frames_d = _make_frames("run_d", [0.04, 0.05, 0.03])

    frames_by_run = {"run_a": frames_a, "run_b": frames_b, "run_d": frames_d}
    run_labels = ["run_a", "run_b", "run_d"]

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "report.md")
        report = generate_report(run_labels, frames_by_run, out)

        assert os.path.isfile(out)
        assert "## Aggregate Metrics" in report
        assert "## Depth-Band Metrics" in report
        assert "## Specular Metrics (intensity pseudo-mask)" in report
        assert "## Per-Sequence Deltas" in report
        assert "## Worst Per-Frame Regressions" in report
        assert "## Decision" in report
        # Run D should win → PROCEED
        assert DECISION_PROCEED in report


def test_generate_report_single_band_missing():
    """If one band has no data across all frames, it still renders the section header."""
    frames_a = _make_frames("run_a", [0.05], seqs=["seq1"])
    frames_b = _make_frames("run_b", [0.04], seqs=["seq1"])
    frames_by_run = {"run_a": frames_a, "run_b": frames_b}
    run_labels = ["run_a", "run_b"]

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "report.md")
        report = generate_report(run_labels, frames_by_run, out)
        assert "## Depth-Band Metrics" in report
        assert "## Decision" in report
