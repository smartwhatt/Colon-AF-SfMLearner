# Research Playbook

## Before running anything

1. Read the other handoff documents and then the canonical topic wiki listed in
   `AGENTS.md`.
2. Confirm the current git commit, checkpoint availability, data-access rights,
   and compute environment with the project owner.
3. Treat an experiment as provisional until its raw note, experiment card,
   index entry, claims impact, and topic log have been updated in the wiki.
4. Do not turn a depth metric, a visual panel, or a model prediction into
   medical advice or a clinical performance claim.

## Environment setup

The tracked project targets Python 3.11 or newer and uses `uv` for dependency
management. Start from a clean virtual environment:

```bash
uv sync
uv run pytest tests
```

If `uv` is unavailable on a local machine, install it through the supported
project workflow rather than mixing an unmanaged global Python into runs. The
maintained test suite is `tests/`; top-level inference scripts are not part of
that suite.

## Repository map

| Location | Responsibility |
| --- | --- |
| `train.py` | Unified training entry point and mode dispatch. |
| `options.py` | Pydantic `TrainingConfig`; add new opt-in knobs here. |
| `trainers/` | Training loops, source replay, teacher consistency, C3VDv2 losses, and evaluation logging. |
| `colon_datasets/` | Dataset contracts for C3VDv2, REALColon, HF triplets, and sparse PolypSense3D labels. |
| `networks/` | AF-SfMLearner components and opt-in Depth Anything V2 adapter. |
| `scripts/` | Reusable local/HF/Lanta wrappers. |
| `tools/` | C3VDv2 depth/pose evaluation, qualitative panels, manifests, and analyses. |
| `tests/` | Maintained unit and behaviour tests; no HF-data download should be required. |

### Key terms

| Term | Meaning |
| --- | --- |
| Run A | Plain C3VDv2 Stage 2 baseline. |
| Run B | SpecularNet C3VDv2 checkpoint and fallback depth prior. |
| Run D | SpecularNet plus PPS C3VDv2 baseline; preferred aggregate phantom-depth checkpoint. |
| `weights_best` | Validation-selected checkpoint. It is not automatically the clinically best checkpoint. |
| `weights_3` / `weights_5` | Final-epoch checkpoints from adaptation campaigns. They require the same gates as `weights_best`. |
| `pose_transform` | Depth-producing modules are frozen; a no-forgetting control, not depth adaptation. |
| `teacher_pose_transform` | Another frozen-depth teacher control. |
| `teacher_source_replay` | Trainable depth/pose branch with frozen teacher consistency and C3VDv2 source replay. |

## Data access and privacy

| Data | Access and use |
| --- | --- |
| C3VDv2 | Public/research benchmark and the source guardrail. The project can consume its HF repackaging where access is configured. |
| REALColon | Research data for clinical appearance adaptation. Follow its license and avoid treating its metadata as calibration-grade size ground truth. |
| PolypSense3D | Sparse clinical depth evaluation only; report its limited label coverage. |
| Private forceps data | Evaluation-first. Build and review a point-label manifest before any training or tuning. |
| Private candidate size archive | Do not copy, expose paths, or use as a benchmark until duplicate/conflict/split/anchor audit is complete. |

Never commit raw images, patient identifiers, credentials, local private-data
paths, or checkpoints. Keep manifests de-identified and store raw evidence only
in the access-controlled project workflow.

The [dataset catalogue and loading guide](datasets_and_access.md) records each
set's camera/calibration context, modality, sample and storage scales, prepared
Hugging Face loading commands, and available depth/pose/size signals.

## Required guardrails for a depth-adapting REALColon checkpoint

Run all four gates before selecting or describing a checkpoint:

```bash
# Absolute scale: no median scaling
uv run python tools/evaluate_c3vd2.py \
  --data_source hf --hf_repo_id SmartWhatt/c3vdv2-SfM --hf_split test \
  --weights_folder <CHECKPOINT_DIR> --load_specular_weights

# Relative shape: median-scaled diagnostic
uv run python tools/evaluate_c3vd2.py \
  --data_source hf --hf_repo_id SmartWhatt/c3vdv2-SfM --hf_split test \
  --weights_folder <CHECKPOINT_DIR> --median_scaling --load_specular_weights

# Pose guardrail
uv run python tools/evaluate_c3vdv2_pose.py \
  --data_source hf --hf_repo_id SmartWhatt/c3vdv2-SfM --hf_split test \
  --weights_folder <CHECKPOINT_DIR>
```

Then run a fixed REALColon qualitative panel through
`scripts/submit_realcolon_panel_lanta.sh` and compare the same rows across the
source and candidate checkpoints. Add PolypSense3D and the private forceps gate
only as held-out, raw-scale clinical evidence. Median scaling is for relative
shape diagnosis, not clinical metric accuracy.

## Current decision flow

1. Do **not** start another long source-selected replay campaign by default.
2. Compare the clinically interesting source-replay final checkpoint with its
   `weights_best` counterpart on a fixed REALColon panel.
3. Design a target-domain selector that is not replay loss alone. It must use
   C3VDv2 scale and shape diagnostics, pose, panel quality, and an
   independently held-out scale-aware sparse gate.
4. If the candidate fails a guardrail, retain Run D as the phantom baseline and
   record the failure. Do not average it away with median scaling.
5. In parallel, make the clinical datasets benchmark-ready before training:
   manifest, label provenance, anchor state, leakage checks, and polyp-level
   splits first.
6. Begin direct sizing with ordinal/abstaining output. Enable millimetres only
   where a validated anchor or scope calibration supports them.

## Reproducibility checklist

Before a compute submission, capture:

- git commit and clean/dirty state;
- exact command, environment variables, package lock, data revision, split,
  resolution, and checkpoint hashes;
- source and target datasets, units, camera preprocessing, and scale policy;
- model-selection criterion and which checkpoints will be evaluated;
- fixed qualitative-panel rows; and
- the full gate matrix, including failures and abstentions.

Use `tools/capture_run_manifest.py` when applicable. When an experiment matters,
write an immutable raw note and then promote the result through the canonical
wiki workflow described in `AGENTS.md`.
