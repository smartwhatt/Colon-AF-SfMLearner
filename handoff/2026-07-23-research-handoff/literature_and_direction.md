# Literature Review and Research Direction (Quick Reference)

The citable, survey-style version of this review is the
[LaTeX/BibTeX literature review paper](literature_review/README.md). This
Markdown document is kept as a fast, searchable companion for the next
researcher; the paper is the authoritative handoff literature artifact.

## Executive recommendation

Do not make another unanchored depth-pose reconstruction campaign the default
next step. The recommended primary direction is an **evaluation-first,
anchor-gated short-video sizing system**:

1. detect and track one polyp across a brief clip;
2. select stable, high-quality frames;
3. predict ordinal size probabilities and, when an anchor is available, a
   continuous diameter with uncertainty;
4. aggregate at the polyp level rather than reporting the noisiest frame; and
5. abstain from metric output when scale evidence is absent or unreliable.

This is not a claim that direct learning solves the monocular scale problem. It
changes the endpoint from "reconstruct a 3D point map, then measure it" to
"estimate the clinically useful measurement directly while exposing the evidence
that supports millimetres." The existing depth model can remain an optional
feature or quality cue, but it should not be the only route to a size estimate.

## 1. Clinical primer for ML researchers

### Why size matters

A colorectal polyp is a growth on the inner lining of the large bowel.
Endoscopists inspect it during colonoscopy, often remove it, and document a
diameter estimate. Size influences surveillance planning and the choice of
resection technique. The common research thresholds are:

| Category | Diameter | Why the boundary is useful in research |
| --- | ---: | --- |
| Diminutive | `<=5 mm` | Small lesions are difficult to measure visually and can occupy few pixels. |
| Small | `>5 mm` to `<10 mm` | A clinically consequential middle range. |
| Large | `>=10 mm` | A common decision threshold, but not a substitute for clinical guidelines. |

These categories are useful model endpoints, not medical advice. Any deployed
system would require clinical validation, workflow study, risk management,
regulatory review, and a clear policy for uncertain cases.

### Measurement labels are not interchangeable

The meaning of `size_mm` depends on how it was produced. Preserve the source in
every manifest and report subgroup performance by it.

| Label source | What it measures | Main limitation |
| --- | --- | --- |
| Fresh-specimen caliper | Excised physical specimen | Tissue handling and shrinkage can differ from in-vivo appearance. |
| Snare or forceps ratio | In-vivo image relative to a tool | Tool opening, orientation, depth mismatch, and visibility matter. |
| Pathology report | Post-resection clinical record | May reflect fixation, fragmentation, and site-specific reporting. |
| Endoscopist visual estimate | Human in-procedure estimate | It is not independent metric ground truth. |
| Phantom geometry | Known physical test object | Excellent for controlled geometry, but not real tissue or clinical workflow. |

### What a real colonoscopy video contains

Full procedures include insertion, withdrawal, washing, suction, instruments,
and treatment. Lesion views are a small, unevenly sampled subset. Bowel
preparation, mucus, bubbles, blood, motion blur, folds, and partial occlusion
can dominate the pixels. A model trained from still images must be tested on
these conditions instead of assuming a curated image is a normal deployment
frame.

## 2. Camera and optical context

Colonoscopy cameras are wide-angle endoscopes with a light source near the lens.
The image is not a textbook, fixed pinhole camera view.

- **Wide field of view and distortion:** C3VDv2 starts with a fisheye-like
  camera and this repository undistorts it to a perspective training view.
  An image size in pixels cannot be compared across scopes or fields of view
  without calibration.
- **Near-light illumination:** brightness changes sharply with distance and
  surface orientation. It can be informative, but is confounded by exposure,
  white balance, tissue reflectance, saturation, and specular highlights.
- **Non-rigid scene:** insufflation, contact, fluid, peristalsis, and scope
  pressure change tissue shape and invalidate many rigid-scene assumptions.
- **Video instability:** adjacent frames may be blurred, occluded, or look
  nearly identical. More frames reduce random noise only if the track and
  frame-quality policy are reliable.
- **Scope shift:** lens, processor, illumination, colour pipeline, resolution,
  and clinical technique differ by manufacturer, model, and operator. Treat
  scope and site as environments when splitting or evaluating data.

### Monocular scale ambiguity

For a pinhole projection, a 3D point `(X, Y, Z)` maps to an image coordinate
through `x = f X / Z`. Multiplying `(X, Y, Z)` and camera translation by the
same positive constant leaves the projection unchanged. Monocular SfM and
photometric self-supervision therefore recover shape only up to a global scale.

The practical consequence is simple: a video model can draw a convincing depth
map and still be wrong by a constant factor in millimetres. Median scaling is a
relative-shape diagnostic; it is never evidence of metric size. A metric output
needs an independent scale anchor, calibrated prior, or empirically validated
scope-specific calibration. The project wiki's near-light analysis further
shows that unconstrained radiometry cannot be assumed to resolve this scale.

## 3. Current project evidence and data roles

| Resource | Role in this project | What it can support | What it cannot support alone |
| --- | --- | --- | --- |
| C3VDv2 | Dense phantom depth/pose guardrail | Controlled depth, pose, specular, and artifact ablations | Clinical tissue robustness or patient polyp sizing |
| REALColon | Clinical appearance and video adaptation | Target-domain qualitative panels and self-supervised adaptation | Dense depth or calibration-grade size labels |
| PolypSense3D | Sparse labelled clinical depth gate | Point-depth comparison and scale-drift diagnosis | Lesion-axis size or 10-mm accuracy |
| Private forceps data | Proposed evaluation-only sparse clinical gate | Stronger clinical depth validation after a manifest is frozen | Training evidence before split and label review |
| Chulalongkorn Polyp Size 2 | Private candidate size benchmark | Future in-vivo size evaluation after cleanup | Headline accuracy now: it has duplicates, conflicts, mixed schemas, and unknown anchor reliability |

Run D (`stage2_specular_pps/weights_best`) is the preferred C3VDv2 baseline.
The current blocker is not a new generic pretraining run. It is the selection
and scale-aware validation of source-replay adaptation checkpoints, together
with a fixed REALColon qualitative comparison.

## 4. What other systems do

The literature uses several different routes to a size estimate. Compare their
scale mechanism and validation design, not only their reported aggregate error.

| Family | Example | Input and output | Scale mechanism | Main lesson |
| --- | --- | --- | --- | --- |
| Lens-calibrated depth pipeline | [ENDOANGEL-CPS](https://pubmed.ncbi.nlm.nih.gov/37827513/) | Frame -> depth and mask -> diameter | Lens parameters and learned depth | Prospective multicentre evidence is possible, but optical calibration is a real dependency. |
| In-scene reference geometry | [Poseidon](https://pmc.ncbi.nlm.nih.gov/articles/PMC10465238/) | Frame with nearby waterjet -> 2D ratio measurement | Known waterjet diameter | A direct 2D system can avoid 3D reconstruction when a reference is visible and near the lesion. |
| Direct regression with learned calibration | [Yii et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12080523/) | Segmentation and depth features -> millimetres | Snare-based metric bootstrap during fine tuning | Good technical precision can still be scope- and protocol-specific. |
| Direct multitask still-image model | [Sunthornwetchapong et al.](https://doi.org/10.1109/ACCESS.2025.3527720) | Detection, log-size regression, bins, coarse depth | Tool-ratio labels during annotation | Ordinal and regression targets can coexist, but label provenance and large-polyp balance remain limiting. |
| Direct RGB or RGB-D classification | [Itoh et al. 2018](https://www.cs.jhu.edu/~lelu/publication/MICCAI2018_Colonoscopy.pdf) and [Hard-Attention Gates](https://arxiv.org/abs/2407.04400) | Frame or clip -> threshold/bin | Learned data prior, not an explicit anchor | A clinically useful category can be easier to validate than a millimetre output, but shortcuts remain possible. |
| Direct video aggregation | [MPSE](https://proceedings.mlr.press/v315/lawrence26a.html) | Per-frame regression -> polyp-level aggregation | Learned scale and depth cues | Quality-aware aggregation is promising, but small clinical cohorts and rare large polyps must be stated. |
| Temporal representation learning | [Temporal SSL](https://arxiv.org/abs/2605.12320) | Short tracklets -> representation -> size probe | No metric scale mechanism | Useful for clip features and invariance, not a stand-alone measurement solution. |

The direct-learning literature supports trying a video endpoint without a
point-cloud stage. It does not validate an unanchored monocular millimetre
claim across scopes or institutions. The direct-video review previously created
for this project is incorporated here as a research direction, not as evidence
of clinical readiness.

## 5. Recommended direct short-video design

### Input and tracking contract

Start with a short, fixed-duration lesion track instead of arbitrary full-video
windows. A practical first protocol is 8 quality-filtered frames sampled from a
stable 1-5 second lesion view. The tracker must retain a per-frame box or mask,
track identifier, timestamps, scope metadata when available, and a reason when
the clip is rejected.

### Model heads

Use a shared visual encoder and temporal aggregation module. A small temporal
transformer, gated average, or LSTM is enough for the first comparison. Attach:

- an ordinal head for `<=5 mm`, `5-10 mm`, and `>=10 mm` probabilities;
- a continuous diameter head trained only where the scale/label protocol is
  appropriate;
- a quality and uncertainty head based on blur, specular fraction, mask
  stability, track consistency, and anchor reliability; and
- optional depth or geometry features as uncertain inputs, not as the sole
  mechanism for scale.

The output contract must expose `metric_available`, `anchor_type`,
`anchor_reliability`, interval/uncertainty, and abstention reason. When the
anchor state is absent or poor, publish ordinal probabilities or abstain; do not
silently emit a millimetre value.

### Scale-anchor options that do not require depth-pose reconstruction

1. **Waterjet reference:** detect a known waterjet feature adjacent to the
   polyp and use pixel-ratio geometry, following Poseidon. Quantify depth and
   orientation mismatch instead of assuming the reference is coplanar.
2. **Forceps or snare reference:** use a fully opened, identified instrument
   only when its physical dimension and image state are validated. Record
   partial opening and occlusion as invalid-anchor conditions.
3. **Scope-specific calibration:** train and evaluate separately by verified
   scope/lens configuration, then require held-out-scope testing before calling
   the calibration portable.
4. **Anchor-gated learning:** let the direct model learn robust visual and
   temporal features, but gate metric supervision and metric output on a
   recorded anchor state.
5. **Ordinal-only baseline:** evaluate the size category as a clinically useful
   non-metric endpoint while the scale protocol and continuous labels mature.

## 6. Evaluation protocol

The unit of inference and splitting is the **polyp**, with patient/procedure
grouping whenever available. Never split adjacent frames of one lesion across
training and test. Report the number of patients, procedures, polyps, clips,
and frames separately.

Required experiments:

1. Compare one frame, 8 frames, 1 second, 3 seconds, and 5 seconds using the
   same polyp-level split and aggregation rule.
2. Compare RGB-only, RGB plus optional depth cue, and anchor-aware variants.
3. Evaluate with ground-truth masks/boxes, predicted masks/boxes, and
   deliberately perturbed masks/boxes to expose dependence on localisation.
4. Report MAE, median absolute error, RMSE, bias and limits of agreement when
   sample size permits, threshold sensitivity/specificity, macro-F1,
   calibration, abstention rate, and latency.
5. Stratify by size bin, morphology, scope model, site, frame quality,
   instrument/anchor state, location, preparation, blur, specularity, and
   missing or unstable masks.
6. Keep an oracle-scale ladder: compare no anchor, known-perfect anchor, and
   detected anchor. This separates representation, localisation, and scale
   failure rather than hiding them in one error number.

## 7. Immediate research sequence

1. Finish the existing depth-adaptation selection gates: fixed REALColon panel,
   C3VDv2 absolute and median-scaled depth checks, pose check, and a
   scale-aware sparse clinical gate.
2. Freeze the private sparse-depth manifest as evaluation-only; do not tune on
   it.
3. Clean the private candidate size benchmark into an anchor-annotated,
   duplicate-free, polyp-level evaluation split.
4. Establish an ordinal still-image baseline and a direct 8-frame baseline on
   that split.
5. Add anchor-gated continuous regression only after the anchor and label
   provenance are demonstrably reliable.

This sequence makes failure interpretable. It avoids mistaking a better
photometric model, a scope shortcut, or a leaky frame split for a clinically
reliable measurement system.
