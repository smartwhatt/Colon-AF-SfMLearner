# Research Handoff - 2026-07-23

This directory is the starting point for the next researcher on this project.
It is written for an ML/DL engineer who may be unfamiliar with colonoscopy. It
summarizes the state of the project as of the handoff date; the topic wiki
named in `AGENTS.md` remains the live, canonical record for future results.

## Read in this order

1. [Preliminary findings short paper](preliminary_findings/README.md) for the
   claim-safe results, methods, interpretation, and limitations.
2. [Literature review paper](literature_review/README.md) for the cited
   clinical, optical, and methodological background and the recommended
   direct, depth-pose-free research track. The accompanying
   [Markdown quick reference](literature_and_direction.md) is useful for fast
   browsing.
3. [Dataset catalogue and loading guide](datasets_and_access.md) for the
   prepared Hugging Face mirrors, public and restricted datasets, camera and
   calibration context, sample/storage scales, and available annotations.
4. [Research playbook](research_playbook.md) before running code or proposing
   a new experiment.
5. [Provenance and claim boundaries](provenance.md) before using a result in a
   paper, talk, or clinical discussion.

## What this package is and is not

It is a documentation and reproducibility handoff. It intentionally contains
no checkpoints, clinical images, patient identifiers, private-data paths, or
credentials. Checkpoint availability and data access must be confirmed with the
project owner before a run is started.

The package does not establish clinical performance, regulatory readiness, or
metric polyp sizing. In particular, a standard monocular colonoscopy video has
an unresolved metric-scale ambiguity unless a validated scale anchor is present.

## Handoff owner and contact

- Smart Wattanapornmongkol (student ID 6738249221)
- Phone: 0639390653
- [LINE contact](https://line.me/ti/p/nqlvyY38hD)
- [Project GitHub repository](https://github.com/smartwhatt/Colon-AF-SfMLearner)
