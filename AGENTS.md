# AGENTS.md

Guidance for coding agents working in this repository.

## Scope

This repository is the active implementation workspace for the colonoscopy
polyp-size research project. Treat it as a research codebase with experiment
provenance requirements, not just a generic AF-SfMLearner fork.

Current project focus:
- Keep Run D (`stage2_specular_pps/weights_best`) as the preferred C3VDv2
  phantom-depth baseline.
- Treat REALColon adaptation as the active blocker. Target-only adaptation has
  shown forgetting or scale drift.
- Use frozen `pose_transform` as a no-forgetting control only; it does not prove
  clinical-domain depth adaptation.
- Run teacher-anchored REALColon diagnostics before replay or longer campaigns.
- Do not claim metric polyp sizing without explicit scale-anchor evidence.

## Wiki Source Of Truth

Use this topic wiki for research state:

```text
/Users/smartwatt/wiki/topics/colonscopy-polyp-size-research
```

Start here:
- `inventory/views/active-actions.md` for current blockers and next work.
- `wiki/experiments/_index.md` for experiment decisions.
- `wiki/experiments/c3vdv2-hf-depth-benchmark.md` for comparable C3VDv2 runs.
- `output/projects/polyp-size-paper/claims-matrix.md` for paper-safe claims.

When an experiment result matters, document it in the wiki:
1. Add an immutable raw note under `raw/notes/`.
2. Add or update an experiment card under `wiki/experiments/`.
3. Update `wiki/experiments/_index.md`.
4. Update paper project files if the result changes claims or figures.
5. Append to the topic `log.md`.

## Active REALColon Workflow

Next long-running diagnostic:

```bash
ADAPTATION_PRESET=teacher_pose_transform START_RUN=run_b \
EPOCHS=1 STEPS_PER_EPOCH=500 LR=1e-5 \
MODEL_NAME=realcolon_run_b_teacher_pose_500 \
bash scripts/submit_real_colon_hf_lanta.sh
```

Every depth-adapting REALColon checkpoint must be gated with:
- C3VDv2 no-median depth evaluation for scale drift.
- C3VDv2 median-scaled depth evaluation for relative shape.
- C3VDv2 pose evaluation.
- Fixed REALColon qualitative panel.

Do not start a full REALColon campaign until the diagnostic branch passes.

## Code Practices

- Prefer the existing Pydantic `TrainingConfig` in `options.py` for new CLI
  knobs.
- Keep defaults backwards compatible. New research branches should be opt-in.
- Use `trainable_modules` for freeze/allowlist behavior instead of ad hoc
  parameter filtering.
- Do not save frozen teacher models in checkpoints.
- Add focused tests for new trainer behavior; avoid tests that download HF data.

## Verification

Run these before committing code changes:

```bash
bash -n scripts/run_finetune_real_colon.sh scripts/submit_real_colon_hf.sh scripts/submit_real_colon_hf_lanta.sh scripts/submit_realcolon_panel_lanta.sh
.venv/bin/python -m py_compile options.py trainers/stage_two.py trainers/base.py scripts/utils/download_real_colon_to_hf.py
uv run pytest tests
git diff --check
```

`uv run pytest` may fail during collection if top-level `test_simple.py` imports
missing `matplotlib`; report that separately and rely on `uv run pytest tests`
for the maintained suite unless the environment is updated.

