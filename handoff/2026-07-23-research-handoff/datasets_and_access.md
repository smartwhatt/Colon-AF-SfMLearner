# Dataset Catalogue and Loading Guide

This guide is the data-access companion to the [research playbook](research_playbook.md). It is written for a researcher who may be familiar with machine learning but not colonoscopy data collection. It distinguishes a source video/frame count from a self-supervised temporal-triplet row count: they are not interchangeable.

**Snapshot date:** 2026-07-23. Hugging Face storage and revisions are mutable; re-check the dataset card and repository revision before a large run. Do not place patient data, credentials, local data paths, or checkpoint payloads in this handoff.

## Fast selection guide

| Dataset | Data type | Metric supervision / calibration | Best handoff use |
| --- | --- | --- | --- |
| C3VDv2 (project HF mirror) | Silicone-colon phantom video, clean and degraded pairs | Dense depth, relative pose, and per-sequence camera calibration | Depth/pose baseline and scale-drift gate |
| REAL-Colon (project HF mirror) | Clinical colonoscopy video | No depth, pose, or device-specific intrinsics in the mirror | Appearance adaptation and fixed qualitative panels |
| PolypSense3D | Virtual, physical-phantom, and clinical images/video | Dense virtual depth; sparse clinical forceps-derived depth and size | Anchor-gated sizing evaluation and calibration research |
| Polyp-Size Dataset | Clinical videos of excised specimens | Vernier-caliper reference size; no depth or intrinsics | Small size-regression evaluation set |
| Chulalongkorn Polyp Size 2 | Private clinical still images | Some size labels and boxes; no depth or verified calibration | Candidate only, after a label/provenance audit |

## Rules before loading data

- A colonoscope is not a calibrated RGB camera by default. It has a wide-angle lens, moving illumination, specular highlights, liquid/debris, and often rolling clinical workflow changes. Pixel area, apparent lesion size, and monocular depth are therefore not a millimetre measurement.
- Treat camera intrinsics, distortion model, scale anchor, and label provenance as part of the dataset. Never silently substitute an approximate intrinsic matrix for device calibration.
- Use the prepared Hugging Face mirrors for the project’s reproducible training inputs. Their rows are decoded temporal triplets, not the original clinical archives.
- Keep public, restricted, and private data separate. Access approval does not authorize copying source media into this repository or a model checkpoint.

## Project-prepared Hugging Face datasets

### C3VDv2 SfM mirror

**Access and loading.** Public mirror: [SmartWhatt/c3vdv2-SfM](https://huggingface.co/datasets/SmartWhatt/c3vdv2-SfM). The checked 2026-07-23 revision was `4f0cfa64a79355f23443f9f7642735b652f3dea4`; Hub reported 93,896,080,690 bytes (87.45 GiB) across 951 files. Stream first when exploring:

```python
from datasets import load_dataset

train = load_dataset("SmartWhatt/c3vdv2-SfM", split="train", streaming=True)
```

For the repository training entry point:

```bash
DATA_SOURCE=hf HF_REPO_ID=SmartWhatt/c3vdv2-SfM HF_STREAMING=1 \
  bash scripts/run_finetune_c3vdv2.sh
```

| Property | Details |
| --- | --- |
| Video type | **Physical phantom** colonoscopy. C3VDv2 contains matched clean and synthetically degraded appearances (for example food residue, blood, mucus, and foam). It is not clinical patient video. |
| Source scale | 192 videos and 169,371 source frames; lesion major axes span 1.4--24.8 mm. |
| Mirror scale | 67,548 temporal-triplet rows: 31,430 train / 19,745 validation / 16,373 test; 81 / 44 / 44 sequences respectively. |
| Camera and calibration | Original frames are 1350 x 1080 omnidirectional fisheye images. The mirror undistorts them to 512 x 512 perspective images with the sequence `camera_intrinsics.txt` and the Scaramuzza camera model. It stores normalized perspective intrinsics `K`; recover pixel intrinsics only after choosing the image width and height. |
| Extra signals | RGB previous/current/next frames, 16-bit depth, occlusion, and 4 x 4 relative transforms in both directions. The project conversion uses `depth_m = uint16 / 65535 * 0.1`. |
| Appropriate claims | A controlled phantom-depth/pose benchmark and the source of Run D, the preferred `stage2_specular_pps/weights_best` baseline. It is not evidence that clinical video has recovered millimetre scale. |

**Why this dataset has useful geometry.** C3VD/C3VDv2 depth and pose come from registration to known silicone-phantom geometry, not from a clinical depth sensor. The original C3VD registration procedure reports 0.321 mm translation and 0.159 degree rotation accuracy. That makes its scale useful for a controlled gate, while remaining a different camera/scene regime from live colonoscopy.

### REAL-Colon SfM mirror

**Access and loading.** Public mirror: [SmartWhatt/real-colon-SfM](https://huggingface.co/datasets/SmartWhatt/real-colon-SfM). The checked 2026-07-23 revision was `21611b9335a71ec6c49eec38a0d4c059e28cf6b5`; Hub reported 251,219,078,529 bytes (233.97 GiB) across 1,256 files.

```python
from datasets import load_dataset

train = load_dataset("SmartWhatt/real-colon-SfM", split="train", streaming=True)
```

```bash
DATA_SOURCE=hf HF_REPO_ID=SmartWhatt/real-colon-SfM HF_STREAMING=1 \
  bash scripts/run_finetune_real_colon.sh
```

| Property | Details |
| --- | --- |
| Video type | **Clinical** full colonoscopy videos from four centres (Japan, Austria, Italy, and the USA). The native recordings use Olympus CF-HQ290L or Fujifilm endoscopes. |
| Source scale | 60 full-resolution 1920 x 1080 MP4s; 2,757,723 native frames; 351,264 annotated boxes; 132 excised polyps. The source paper reports metadata including histology, location, bowel preparation, frame rate, and endoscope brand/model. |
| Mirror scale | 250,522 lower-frame-rate temporal-triplet rows: 201,543 train / 30,525 validation / 18,454 test. `frame_stride` records the original-frame distance. These are not 250,522 independent clinical procedures. |
| Camera and calibration | Conventional wide-angle optics, approximately 170-degree field of view, rather than the C3VDv2 omnidirectional fisheye. The public source does not supply device-specific intrinsics or a distortion model. The repository loader uses a normalized approximate `K` as a pragmatic training prior (including for Fujifilm), **not** ground-truth calibration. |
| Extra signals | RGB temporal triplets and sequence metadata only. Depth, occlusion, and relative pose are null/absent in the mirror. Histopathology-derived polyp size is not a per-frame geometric scale anchor. |
| Appropriate claims | Clinical appearance adaptation and qualitative assessment only. Target-only adaptation has shown forgetting or scale drift; every depth-adapting checkpoint must still pass C3VDv2 no-median depth, median-scaled depth, pose, and a fixed REAL-Colon panel. |

**Practical implication.** REAL-Colon lets a model see clinical lighting, mucosa, specularity, motion, and instrument variation. It does not make monocular clinical video metrically calibrated. Do not report a millimetre size from these rows unless a separately validated anchor is visible and accepted by the evaluation protocol.

## Other public datasets relevant to the next project phase

### Original C3VD

| Property | Details |
| --- | --- |
| Access | [Original C3VD release](https://durrlab.github.io/C3VD/) (Google Drive from the dataset authors); use the project C3VDv2 mirror above for the active reproducible pipeline. |
| Video type | **Physical phantom** colonoscopy: four silicone colon phantoms. |
| Sample and storage scale | 22 registered sequences, 10,015 annotated frames, and about 20,058 additional simulated-screening frames; the documented download is approximately 100+ GB. |
| Camera and calibration | Registration is against known phantom meshes and includes camera poses. Inspect each release’s calibration metadata rather than assuming the C3VDv2 perspective conversion applies unchanged. |
| Extra signals | 16-bit z-axis depth (0--100 mm), normals, optical flow, occlusion, 6-DoF poses, coverage masks, and OBJ meshes. |
| Use / limitation | Useful for legacy controlled-geometry comparisons, not as clinical sizing validation. |

### PolypSense3D

| Property | Details |
| --- | --- |
| Access | [Dataset repository](https://github.com/HNUicda/PolypSense3D) and [Harvard Dataverse record](https://doi.org/10.7910/DVN/K13H89). No project-prepared Hugging Face mirror exists. Download only to an approved data location. |
| Video type | Mixed **virtual**, **physical phantom**, and **clinical** data. The clinical subset comes from 127 patients and was recorded with an Olympus EVIS EXERA III / CF-H290I system. |
| Sample scale | Paper-reported total: 43k+ frames and 114 polyp instances. The clinical component has about 11k frames, 438 polyp-containing frames, and 53 polyps; virtual data has about 32k frames / 30 polyps; physical data has 13 videos / 74 polyp frames / 13 polyps. |
| Storage scale | Not verified in this handoff. Read the Dataverse file manifest and checksums before allocating storage or copying files. |
| Camera and calibration | Calibration metadata maps sequences to camera IDs in `calibrations/video_info.xlsx`. The clinical camera is not a promise of per-frame metric scale by itself. |
| Extra signals | RGB, masks, intrinsics, and size labels. Virtual data has dense synthetic depth; physical data uses calibrated/geometry-derived information; clinical data has sparse forceps-derived metric depth points and forceps-assisted in-vivo size, not dense clinical depth. |
| Local evaluation layout | The repository sparse-depth evaluator expects `labeled_data/depth/rgb`, `labeled_data/depth/json`, `labeled_data/depth/depth_gt.xlsx`, and `calibrations/video_info.xlsx` beneath an approved root. Run `tools/evaluate_polysense3d_sparse_depth.py --data_root <APPROVED_POLYSENSE3D_ROOT> --weights_folder <CHECKPOINT_DIR>`. |
| Use / limitation | Best current public route for testing an anchor-gated clinical-sizing idea. The clinical size range is below 10 mm (reported maximum 9.98 mm), so do not generalize it to larger lesions without a new evaluation set. |

### Polyp-Size Dataset

| Property | Details |
| --- | --- |
| Access | [Figshare dataset](https://doi.org/10.6084/m9.figshare.28030115) and [project code](https://github.com/Scarlett213/Polyp-Size-Dataset). No project-prepared Hugging Face mirror exists. |
| Video type | **Clinical endoscopy video of excised polyp specimens**, not in-vivo patient lesions. |
| Sample and storage scale | 42 MP4s, one polyp per video; 12,340 raw frames at about 30 fps, or 3,858 frames after 15-fps sampling; 1000 x 1170 resolution; sizes 2.80--11.74 mm. Storage size was not recorded in the project catalogue; inspect the Figshare manifest before download. |
| Camera and calibration | Scope brand/model metadata is reported, but the project catalogue has no supplied per-video intrinsics, distortion calibration, or depth map. Treat it as uncalibrated RGB for this project. |
| Extra signals | Fresh resected-specimen size measured with a 0.01-mm digital vernier caliper by three endoscopists (the maximum measurement was selected; reported ICC 0.987). No depth, pose, boxes, or masks are supplied for SfM training. |
| Use / limitation | A small outcome-evaluation set for direct size prediction after an external detector/segmenter. Ex-vivo specimen dimensions may differ from in-vivo appearance and should not be treated as a live-procedure calibration study. |

## Private or restricted project data

### Chulalongkorn Polyp Size 2 (candidate)

| Property | Details |
| --- | --- |
| Access | Private project data. The approved CU download is [Google Drive: Chulalongkorn Polyp Size 2](https://drive.google.com/file/d/1yx827a_nNv1p0L8QoEHiGoVpw2z5XHjR/view?usp=sharing). Request access from the data owner and use the approved clinical-data environment. Do not add source paths, images, identifiers, or copied labels to this repository or handoff. |
| Data type | **Clinical still images**, not temporally ordered video. |
| Sample and storage scale | 6,386 JPG images in 13,035 total files; 2,402,009,759 bytes (about 2.24 GiB). The inventory records 2,809 images with peer six-column size labels and 2,893 six-column boxes. |
| Camera and calibration | Camera model, intrinsics, distortion calibration, and metric depth are not verified in the current inventory. Do not infer them from image dimensions. |
| Extra signals | LabelMe JSON boxes, YOLO-like text labels, CSV/ZIP material, and notebooks. Some labels include a size field; depth and pose are absent. |
| Data-quality status | Candidate only. The inventory found exact duplicates crossing train/validation, conflicting labels, 330 training labels without size, mixed normalized/pixel coordinates, and uncertain provenance. Freeze it out of quantitative claims until an immutable audit, group-aware split, label-unit check, and clinician review are complete. |

## Dataset choice and decision rules

1. **Need a controlled geometric baseline?** Train/evaluate on the C3VDv2 mirror and preserve Run D as the baseline.
2. **Need clinical appearance adaptation?** Use REAL-Colon triplets, but first compare the clinically interesting source-replay final checkpoint with its `weights_best` counterpart on the fixed panel and apply the four required geometry/qualitative gates in the playbook. Do not default to another frozen-depth control.
3. **Need to evaluate a direct sizing method?** Build short lesion clips with valid lesion/anchor/quality labels, use PolypSense3D for sparse anchor-aware checks, and use Polyp-Size only as small ex-vivo outcome evidence.
4. **No validated anchor in view?** Output an ordinal class, uncertainty interval, or abstention--not a metric millimetre claim.
5. **Considering private stills?** Complete the private-data audit before model selection, and do not mix still-image labels with video-level claims.

For experiment order, fixed panels, checkpoint terminology, and public/private boundaries, continue with the [research playbook](research_playbook.md). The [provenance record](provenance.md) defines the claim-safe boundaries for the package.
