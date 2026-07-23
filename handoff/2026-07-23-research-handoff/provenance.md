# Provenance, Boundaries, and Handoff Integrity

## Handoff snapshot

| Field | Value |
| --- | --- |
| Handoff date | 2026-07-23 |
| Repository commit consulted | `d48d63baa1803eadbf44d26cd7cc1b65bc119c23` |
| Baseline maintained tests | `166 passed, 1 warning` with `.venv/bin/python -m pytest tests --tb=no -q` |
| Package purpose | Documentation, reproducibility, and safe research continuation |
| Canonical live record | The topic wiki named in `AGENTS.md` |
| Project repository | [github.com/smartwhatt/Colon-AF-SfMLearner](https://github.com/smartwhatt/Colon-AF-SfMLearner) |
| Handoff contact | Smart Wattanapornmongkol; student ID 6738249221; phone 0639390653; [LINE](https://line.me/ti/p/nqlvyY38hD) |
| Compiled papers | `preliminary_findings/preliminary-findings.pdf` and `literature_review/literature-review.pdf` |

The package is a point-in-time synthesis. Recheck the topic wiki, current git
state, data revisions, and checkpoint manifests before using any number here.

## Evidence used

### Canonical project evidence

- Active actions and current experiment index in the topic wiki.
- C3VDv2 HF depth benchmark and its Run A-E ablation cards.
- REALColon target-only, frozen-depth control, teacher, and source-replay
  experiment cards.
- Paper claims matrix and results register.
- Dataset records for C3VDv2, REALColon, PolypSense3D, the Polyp-Size
  Dataset, and the private candidate size archive. The dataset catalogue also
  records the public Hugging Face mirror revisions and used-storage snapshots
  checked on the handoff date.

Session digests were not used as scientific evidence. They only confirmed the
user's handoff request and must not be promoted into project claims.

### External literature

The preliminary paper cites primary papers or official proceedings for
AF-SfMLearner, ENDOANGEL-CPS, Poseidon, the Yii et al. direct system, and MPSE.
The literature-review paper has its own BibTeX database and links primary
papers, official proceedings, DOI pages, or author manuscripts for direct-video
and temporal methods. Its two arXiv entries are labelled as preprints both in
the bibliography and in the discussion.

Publication status is material. The 2026 temporal SSL work is an arXiv
manuscript, and any online-preview work must be checked again before it is used
as settled clinical evidence. Report direct-method results with their cohort,
scope, label, and validation design rather than treating a single headline MAE
as a portable benchmark.

## Known missing artifacts and evidence

- There is no leakage-checked, anchor-annotated clinical polyp-size benchmark.
- There is no validated cross-scope calibration or scale-anchor protocol for
  routine monocular video.
- The fixed REALColon qualitative-panel comparison and scale-aware selector
  remain incomplete for the clinically interesting source-replay final
  checkpoint.
- Checkpoints, raw experiment notes, raw clinical images, private manifests,
  and private label payloads are intentionally excluded. Consult the canonical
  wiki and the approved data environment rather than reconstructing them from
  this package.

## Explicitly allowed claims

- Run D is the strongest tested C3VDv2 phantom-depth baseline in this project.
- SpecularNet plus PPS outperformed the tested baseline/PPS-only alternatives on
  the comparable C3VDv2 protocol.
- Unconstrained REALColon adaptation produced forgetting or scale drift in the
  documented diagnostics.
- Frozen `pose_transform` and `teacher_pose_transform` are no-forgetting
  controls, not clinical depth-adaptation results.
- Source replay preserved C3VDv2 guardrails more effectively than earlier
  trainable-depth attempts, and one final checkpoint improved a sparse
  PolypSense3D raw gate.

## Claims that remain prohibited

- Metric polyp sizing from standard monocular video without an explicit,
  validated scale anchor.
- Clinical 5-mm or 10-mm threshold accuracy.
- Broad clinical generalization or a selected real-domain metric-depth winner.
- Equating median-scaled depth with metric accuracy.
- Treating private candidate size labels as a leakage-free benchmark before
  audit.
- Clinical deployment readiness, regulatory compliance, or patient benefit.

## Privacy and access rules

This package contains no clinical media, patient identifiers, private-data
location, access token, or checkpoint. Do not add them. De-identify manifests,
store sensitive evidence only in the approved data environment, and obtain the
project owner's approval before accessing private datasets or compute accounts.

## How to update this package

Use this package as a stable handoff artifact, not as a second experiment log.
For a material new result, first follow the canonical wiki provenance workflow.
Then create a dated successor handoff or explicitly update this package with the
new commit, evidence card, claim status, bibliography check, and PDF review.
