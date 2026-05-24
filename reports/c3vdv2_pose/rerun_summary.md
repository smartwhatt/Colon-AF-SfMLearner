---
source: "asahi:~/Colon-AF-SfMLearner/reports/c3vdv2_pose"
copied: 2026-05-24
---

# C3VDv2 Pose Rerun Summary

Rerun artifacts were copied from `asahi` into this directory. Each JSONL has
32,746 pose rows: 16,373 `prev` pairs and 16,373 `next` pairs.

## Results

| Run | Rot mean | Rot median | Rot p90 | Trans L2 mean | Trans L2 median | Trans L2 p90 | GT trans norm median | Pairs below 1e-6 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_b_base` | 0.0916 deg | 0.0560 | 0.1998 | 0.2364 mm | 0.1686 | 0.5125 | 0.0 m | 32746/32746 |
| `run_d_base` | 0.0964 deg | 0.0560 | 0.1978 | 0.2291 mm | 0.1618 | 0.5048 | 0.0 m | 32746/32746 |
| `run_b_pose_transform` | 0.0928 deg | 0.0560 | 0.1897 | 0.0794 mm | 0.0526 | 0.1627 | 0.0 m | 32746/32746 |
| `run_d_pose_transform` | 0.1382 deg | 0.1047 | 0.2238 | 0.0730 mm | 0.0478 | 0.1515 | 0.0 m | 32746/32746 |

## Interpretation

All GT translation norms are below `1e-6 m`, so `Trans cosine` and
`Trans mag ratio` are invalid for model selection on this rerun. `Trans L2`
is effectively the predicted translation magnitude because the GT translation
vector is zero for every evaluated pair.

Use rotation error as the reliable pose-accuracy signal. Run B
`pose_transform` preserves rotation relative to Run B base and reduces predicted
translation magnitude. Run D `pose_transform` has the smallest predicted
translation magnitude but has a clear rotation regression.

## Files

- `run_b_base.summary.json`, `run_b_base.json`, `run_b_base.jsonl`
- `run_d_base.summary.json`, `run_d_base.json`, `run_d_base.jsonl`
- `run_b_pose_transform.summary.json`, `run_b_pose_transform.json`, `run_b_pose_transform.jsonl`
- `run_d_pose_transform.summary.json`, `run_d_pose_transform.json`, `run_d_pose_transform.jsonl`
