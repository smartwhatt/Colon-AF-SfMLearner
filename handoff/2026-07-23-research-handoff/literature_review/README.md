# Literature Review Paper

This directory contains the citable survey-style literature review for the
handoff. It is written for ML/DL researchers who are new to colonoscopy and
explains the clinical workflow, image-formation constraints, direct-sizing
methods, and the proposed research direction.

## Contents

- `main.tex` — LaTeX source.
- `references.bib` — BibTeX database, with peer-reviewed papers, proceedings,
  and clearly labelled preprints.
- `literature-review.pdf` — tracked compiled paper.
- `Makefile` — reproducible build entry point.

## Build

```bash
cd handoff/2026-07-23-research-handoff/literature_review
make pdf
```

The Makefile prefers `latexmk` and falls back to Tectonic. Generated
intermediates are deliberately ignored in `build/`; the compiled PDF is
tracked. The Markdown companion at `../literature_and_direction.md` remains a
quick-reference version of the review.
