# Preliminary Findings Report

`main.tex` is the authoritative editable report and `references.bib` contains
its BibTeX citations. `preliminary-findings.pdf` is the reviewed build artifact.

## Build

The report uses standard LaTeX and BibTeX. From this directory, run:

```bash
make pdf
```

This prefers `latexmk -pdf` and falls back to Tectonic, which runs BibTeX for
this document. It writes temporary files under `build/`. Install either a TeX
distribution with `latexmk`, `pdflatex`, and BibTeX, or the standalone Tectonic
compiler. The report uses standard packages (`geometry`, `hyperref`,
`tabularx`, `booktabs`, and `url`).

To remove local build products without removing the tracked PDF:

```bash
make clean
```

The PDF must be regenerated after changes to `main.tex` or `references.bib` and
visually inspected page by page before handoff.
