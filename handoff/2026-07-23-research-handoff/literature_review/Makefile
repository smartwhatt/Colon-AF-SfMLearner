PDF := literature-review.pdf
TEX := main.tex
BUILD := build
LATEXMK ?= latexmk
TECTONIC ?= tectonic

.PHONY: pdf clean

pdf:
	@mkdir -p $(BUILD)
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(TEX); \
	elif command -v $(TECTONIC) >/dev/null 2>&1; then \
		$(TECTONIC) -X compile --keep-logs --keep-intermediates --outdir $(BUILD) $(TEX); \
	else \
		echo "Need latexmk (preferred) or Tectonic to build the report."; exit 1; \
	fi
	cp $(BUILD)/main.pdf $(PDF)

clean:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -C -outdir=$(BUILD) $(TEX); \
	fi
	rm -rf $(BUILD)
