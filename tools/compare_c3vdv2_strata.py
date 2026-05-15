#!/usr/bin/env python3
"""Compare stratified C3VDv2 evaluation outputs across runs.

Reads per-frame JSONL files produced by ``evaluate_c3vdv2.py --stratify`` and
writes a markdown report comparing aggregate, depth-band, specular-region, and
per-sequence metrics across runs, with a go/no-go recommendation for REALColon
SSL adaptation.

Usage::

    uv run python tools/compare_c3vdv2_strata.py \\
        --inputs \\
            reports/c3vdv2/run_a_metrics.jsonl \\
            reports/c3vdv2/run_b_metrics.jsonl \\
            reports/c3vdv2/run_d_metrics.jsonl \\
        --out reports/c3vdv2/run_d_vs_ab_stratified.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

DEPTH_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
DEPTH_BAND_NAMES = ["0_5mm", "5_10mm", "10_20mm", "20_50mm", "50_100mm"]

DECISION_PROCEED = "PROCEED_REALCOLON_SSL"
DECISION_CAUTION = "PROCEED_WITH_CAUTION"
DECISION_RECHECK = "RECHECK_RUN_D"


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of per-frame dicts."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _agg_metrics(frames: list[dict]) -> dict[str, float]:
    """Compute aggregate metrics across all frames."""
    global_metrics = [f["global"] for f in frames if f.get("global")]
    if not global_metrics:
        return {}
    arr = np.array([[m.get(k, float("nan")) for k in DEPTH_METRIC_KEYS]
                     for m in global_metrics])
    mean = np.nanmean(arr, axis=0)
    return {k: float(mean[i]) for i, k in enumerate(DEPTH_METRIC_KEYS)}


def _band_metrics(frames: list[dict], band_name: str) -> dict[str, float] | None:
    """Compute aggregate metrics for a specific depth band across frames."""
    band_vals = [
        f["depth_bands"][band_name]
        for f in frames
        if f.get("depth_bands", {}).get(band_name) is not None
    ]
    if not band_vals:
        return None
    arr = np.array([[m.get(k, float("nan")) for k in DEPTH_METRIC_KEYS]
                     for m in band_vals])
    mean = np.nanmean(arr, axis=0)
    return {k: float(mean[i]) for i, k in enumerate(DEPTH_METRIC_KEYS)} | {
        "n_frames": len(band_vals)
    }


def _specular_metrics(frames: list[dict], source: str, region: str) -> dict[str, float] | None:
    """Compute aggregate metrics for specular inside/outside from a given source."""
    vals = [
        f[source][region]
        for f in frames
        if f.get(source, {}).get(region) is not None
    ]
    if not vals:
        return None
    arr = np.array([[m.get(k, float("nan")) for k in DEPTH_METRIC_KEYS]
                     for m in vals])
    mean = np.nanmean(arr, axis=0)
    return {k: float(mean[i]) for i, k in enumerate(DEPTH_METRIC_KEYS)} | {
        "n_frames": len(vals)
    }


def _per_sequence_deltas(
    frames_a: list[dict],
    frames_b: list[dict],
    metric_key: str = "abs_rel",
) -> dict[str, dict]:
    """Compute per-sequence delta (B - A) for a given metric.

    Returns ``{seq: {"a": val, "b": val, "delta": b - a, "n_a": int, "n_b": int}}``.
    """
    def _seq_map(frames):
        out: dict[str, list[float]] = defaultdict(list)
        for f in frames:
            seq = f.get("seq", "unknown")
            gm = f.get("global", {})
            if metric_key in gm:
                out[seq].append(gm[metric_key])
        return {seq: float(np.mean(vals)) for seq, vals in out.items()}

    seq_a = _seq_map(frames_a)
    seq_b = _seq_map(frames_b)

    all_seqs = sorted(set(seq_a.keys()) | set(seq_b.keys()))
    result = {}
    for seq in all_seqs:
        va = seq_a.get(seq)
        vb = seq_b.get(seq)
        if va is not None and vb is not None:
            result[seq] = {"a": va, "b": vb, "delta": vb - va}
        elif va is not None:
            result[seq] = {"a": va, "b": None, "delta": None}
        else:
            result[seq] = {"a": None, "b": vb, "delta": None}
    return result


def _worst_regressions(
    frames_ref: list[dict],
    frames_cmp: list[dict],
    metric_key: str = "abs_rel",
    top_n: int = 10,
) -> list[dict]:
    """Find individual frames where *cmp* is materially worse than *ref*.

    Returns up to *top_n* entries sorted by largest regression.
    """
    def _frame_map(frames):
        out = {}
        for f in frames:
            seq = f.get("seq", "unknown")
            fidx = f.get("frame_idx")
            gm = f.get("global", {})
            if metric_key in gm:
                out[(seq, fidx)] = gm[metric_key]
        return out

    ref_map = _frame_map(frames_ref)
    cmp_map = _frame_map(frames_cmp)

    regressions = []
    for key, cmp_val in cmp_map.items():
        ref_val = ref_map.get(key)
        if ref_val is not None and cmp_val is not None:
            delta = cmp_val - ref_val
            if delta > 0:  # cmp is worse (higher AbsRel)
                regressions.append({
                    "seq": key[0],
                    "frame_idx": key[1],
                    "ref_val": ref_val,
                    "cmp_val": cmp_val,
                    "delta": delta,
                })

    regressions.sort(key=lambda x: x["delta"], reverse=True)
    return regressions[:top_n]


def _decide(
    run_labels: list[str],
    agg: dict[str, dict],
    band_agg: dict[str, dict],
    spec_intensity: dict[str, dict],
    seq_deltas: dict[str, dict],
    has_regressions: bool,
) -> str:
    """Apply the decision rule to recommend go / caution / recheck.

    Assumes the last run in *run_labels* is the candidate (Run D) and earlier
    runs include the baseline (Run B).
    """
    candidate = run_labels[-1]
    baseline = run_labels[0] if len(run_labels) > 1 else None

    # Find the best run by aggregate AbsRel
    best_run = min(agg, key=lambda r: agg[r].get("abs_rel", float("inf")))

    # RECHECK: candidate loses to baseline in aggregate
    if baseline and baseline != candidate:
        cand_absrel = agg.get(candidate, {}).get("abs_rel", float("inf"))
        base_absrel = agg.get(baseline, {}).get("abs_rel", float("inf"))
        if not math.isnan(cand_absrel) and not math.isnan(base_absrel):
            if cand_absrel > base_absrel:
                return DECISION_RECHECK

    # CAUTION: candidate is best aggregate but has material regressions
    if has_regressions:
        return DECISION_CAUTION

    # CAUTION: candidate is best but gains not concentrated in expected strata
    # Check near-field bands (0_5mm, 5_10mm) — if candidate is worse here
    if baseline:
        for band in ["0_5mm", "5_10mm"]:
            b = band_agg.get(baseline, {}).get(band)
            c = band_agg.get(candidate, {}).get(band)
            if b and c:
                b_absrel = b.get("abs_rel", float("inf"))
                c_absrel = c.get("abs_rel", float("inf"))
                if c_absrel > b_absrel:
                    return DECISION_CAUTION

        # Check specular inside — if candidate is worse here
        for region in ["inside", "outside"]:
            si_b = spec_intensity.get(baseline, {}).get(region)
            si_c = spec_intensity.get(candidate, {}).get(region)
            if si_b and si_c:
                b_absrel = si_b.get("abs_rel", float("inf"))
                c_absrel = si_c.get("abs_rel", float("inf"))
                if c_absrel > b_absrel:
                    return DECISION_CAUTION

    return DECISION_PROCEED


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _fmt_metrics_row(label: str, metrics: dict | None, keys=DEPTH_METRIC_KEYS) -> str:
    if metrics is None:
        return f"| {label} | " + " | ".join(["-"] * len(keys)) + " |"
    vals = [f"{metrics.get(k, float('nan')):.4f}" for k in keys]
    return f"| {label} | " + " | ".join(vals) + " |"


def generate_report(
    run_labels: list[str],
    frames_by_run: dict[str, list[dict]],
    out_path: str,
) -> str:
    """Build a complete markdown comparison report and write it to *out_path*."""
    lines: list[str] = []

    # Precompute
    agg = {label: _agg_metrics(frames_by_run[label]) for label in run_labels}

    band_agg: dict[str, dict] = {}
    for label in run_labels:
        band_agg[label] = {}
        for band_name in DEPTH_BAND_NAMES:
            bm = _band_metrics(frames_by_run[label], band_name)
            if bm:
                band_agg[label][band_name] = bm

    spec_intensity: dict[str, dict] = {}
    for label in run_labels:
        spec_intensity[label] = {}
        for region in ("inside", "outside"):
            sm = _specular_metrics(frames_by_run[label], "specular_intensity", region)
            if sm:
                spec_intensity[label][region] = sm

    spec_model: dict[str, dict] = {}
    has_spec_model = any(
        any(f.get("specular_model", {}).get("inside") is not None
            for f in frames_by_run[label])
        for label in run_labels
    )
    if has_spec_model:
        for label in run_labels:
            spec_model[label] = {}
            for region in ("inside", "outside"):
                sm = _specular_metrics(frames_by_run[label], "specular_model", region)
                if sm:
                    spec_model[label][region] = sm

    # Per-sequence deltas: last run vs each earlier run
    seq_deltas: dict[str, dict] = {}
    candidate = run_labels[-1]
    for baseline in run_labels[:-1]:
        key = f"{candidate}_vs_{baseline}"
        seq_deltas[key] = _per_sequence_deltas(
            frames_by_run[baseline], frames_by_run[candidate], "abs_rel"
        )

    # Worst regressions: candidate vs earlier runs
    worst_regressions: dict[str, list] = {}
    regressions_exist = False
    for baseline in run_labels[:-1]:
        key = f"{candidate}_vs_{baseline}"
        wr = _worst_regressions(
            frames_by_run[baseline], frames_by_run[candidate], "abs_rel", top_n=10
        )
        worst_regressions[key] = wr
        if wr:
            regressions_exist = True

    # Decision
    decision = _decide(
        run_labels, agg, band_agg, spec_intensity, seq_deltas, regressions_exist
    )

    # ---- Render report ---------------------------------------------------

    lines.append("# C3VDv2 Stratified Comparison")
    lines.append("")
    lines.append(f"**Runs:** {', '.join(run_labels)}")
    lines.append("")

    # --- Aggregate ---
    lines.append("## Aggregate Metrics")
    lines.append("")
    header = "| Run | " + " | ".join(DEPTH_METRIC_KEYS) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(DEPTH_METRIC_KEYS) + 1)) + "|")
    for label in run_labels:
        lines.append(_fmt_metrics_row(label, agg.get(label)))
    lines.append("")

    # --- Depth bands ---
    lines.append("## Depth-Band Metrics")
    lines.append("")
    for band_name in DEPTH_BAND_NAMES:
        lines.append(f"### {band_name}")
        lines.append("")
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(DEPTH_METRIC_KEYS) + 1)) + "|")
        for label in run_labels:
            lines.append(_fmt_metrics_row(label, band_agg.get(label, {}).get(band_name)))
        lines.append("")

    # --- Specular intensity ---
    lines.append("## Specular Metrics (intensity pseudo-mask)")
    lines.append("")
    for region in ("inside", "outside"):
        lines.append(f"### {region}")
        lines.append("")
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(DEPTH_METRIC_KEYS) + 1)) + "|")
        for label in run_labels:
            lines.append(_fmt_metrics_row(label, spec_intensity.get(label, {}).get(region)))
        lines.append("")

    # --- Specular model ---
    if has_spec_model:
        lines.append("## Specular Metrics (model mask)")
        lines.append("")
        for region in ("inside", "outside"):
            lines.append(f"### {region}")
            lines.append("")
            lines.append(header)
            lines.append("|" + "|".join(["---"] * (len(DEPTH_METRIC_KEYS) + 1)) + "|")
            for label in run_labels:
                lines.append(_fmt_metrics_row(label, spec_model.get(label, {}).get(region)))
            lines.append("")

    # --- Per-sequence deltas ---
    lines.append("## Per-Sequence Deltas (AbsRel)")
    lines.append("")
    for key, deltas in seq_deltas.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| Sequence | A | B | Delta (B - A) |")
        lines.append("|---|---|---|---|")
        for seq in sorted(deltas.keys()):
            d = deltas[seq]
            a_str = f"{d['a']:.4f}" if d["a"] is not None else "-"
            b_str = f"{d['b']:.4f}" if d["b"] is not None else "-"
            delta_str = f"{d['delta']:+.4f}" if d["delta"] is not None else "-"
            lines.append(f"| {seq} | {a_str} | {b_str} | {delta_str} |")
        lines.append("")

    # --- Worst regressions ---
    lines.append("## Worst Per-Frame Regressions")
    lines.append("")
    for key, regs in worst_regressions.items():
        lines.append(f"### {key}")
        lines.append("")
        if not regs:
            lines.append("*(no regressions found)*")
            lines.append("")
        else:
            lines.append("| Seq | Frame | Ref AbsRel | Cmp AbsRel | Delta |")
            lines.append("|---|---|---|---|---|")
            for r in regs:
                lines.append(
                    f"| {r['seq']} | {r['frame_idx']} | "
                    f"{r['ref_val']:.4f} | {r['cmp_val']:.4f} | "
                    f"{r['delta']:+.4f} |"
                )
            lines.append("")

    # --- Decision ---
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**Recommendation: `{decision}`**")
    lines.append("")
    if decision == DECISION_PROCEED:
        lines.append(
            "Run D remains the best aggregate performer and does not show broad "
            "regressions in near-field or specular strata. Safe to proceed with "
            "REALColon SSL using Run D as the starting checkpoint."
        )
    elif decision == DECISION_CAUTION:
        lines.append(
            "Run D is best aggregate but gains are not uniformly concentrated in "
            "expected failure modes, or some sequences regress materially. "
            "Proceed with REALColon SSL but include Run B as a fallback starting "
            "checkpoint in the diagnostic design."
        )
    else:
        lines.append(
            "Run D does not outperform the baseline in aggregate or has severe "
            "systematic regressions in near-field/specular strata. Re-run aggregate "
            "evaluation and inspect checkpoint/config mismatch before REALColon training."
        )

    report = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare stratified C3VDv2 evaluation outputs across runs."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more JSONL files from evaluate_c3vdv2.py --stratify."
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Optional run labels (one per --input). Default: derived from run_id in JSONL."
    )
    parser.add_argument("--out", required=True, help="Output markdown report path.")
    args = parser.parse_args()

    frames_by_run: dict[str, list[dict]] = {}
    run_labels: list[str] = []

    for i, path in enumerate(args.inputs):
        if not os.path.isfile(path):
            print(f"WARNING: input file not found, skipping: {path}")
            continue
        frames = _load_jsonl(path)
        if not frames:
            print(f"WARNING: empty file, skipping: {path}")
            continue

        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            label = frames[0].get("run_id") or os.path.basename(path).replace("_metrics.jsonl", "")

        frames_by_run[label] = frames
        run_labels.append(label)

    if len(run_labels) < 2:
        sys.exit("ERROR: need at least two valid input files for comparison.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = generate_report(run_labels, frames_by_run, args.out)
    print(report)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
