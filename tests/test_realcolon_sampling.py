import os
import sys
from argparse import Namespace
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from scripts.generate_splits_realcolon import (
    SequenceInfo,
    available_frame_indices,
    infer_extracted_frame_step,
    sample_available_frame_indices,
    validate_sequences,
)
from scripts.utils.download_real_colon_to_hf import (
    README_TEMPLATE,
    QualityFilterConfig,
    QualityFilterState,
    _triplet_rows_for_sequence,
    process_split,
    write_parquet,
)


def _seq(seq_id="001-001", num_frames=31, fps=30.0):
    return SequenceInfo(
        seq_id=seq_id,
        num_frames=num_frames,
        fps=fps,
        endoscope_brand="Olympus",
        bbps="3",
    )


def _write_frames(root, seq_id, indices):
    frame_dir = root / f"{seq_id}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        (frame_dir / f"{seq_id}_{index}.jpg").write_bytes(f"frame-{index}".encode())


def _write_image_frames(root, seq_id, frames):
    frame_dir = root / f"{seq_id}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, array in frames.items():
        Image.fromarray(array.astype(np.uint8), mode="RGB").save(
            frame_dir / f"{seq_id}_{index}.jpg"
        )


def _quality_state():
    return QualityFilterState(
        config=QualityFilterConfig(
            downsample_size=64,
            min_mean_luma=8.0,
            max_dark_fraction=0.98,
            max_saturated_fraction=0.98,
            min_luma_std=3.0,
            min_edge_mean=1.0,
            max_noise_score=None,
        ),
        cache={},
        stats=Counter(),
    )


def test_validate_sequences_accepts_sparse_official_extraction(tmp_path):
    seq = _seq(num_frames=32)
    _write_frames(tmp_path, seq.seq_id, [1, 7, 13, 19, 25, 31])

    valid, warnings = validate_sequences(
        str(tmp_path),
        [seq],
        verify_sample_paths=False,
        sample_stride=10,
        require_contiguous_neighbors=False,
    )

    assert valid == [seq]
    assert warnings == []


def test_validate_sequences_rejects_sparse_extraction_for_local_split_mode(tmp_path):
    seq = _seq(num_frames=32)
    _write_frames(tmp_path, seq.seq_id, [1, 7, 13, 19, 25, 31])

    valid, warnings = validate_sequences(
        str(tmp_path),
        [seq],
        verify_sample_paths=False,
        sample_stride=10,
    )

    assert valid == []
    assert warnings == ["001-001: missing expected contiguous frame(s) 2, 32"]


def test_sample_available_frame_indices_snaps_target_fps_to_existing_grid(tmp_path):
    seq = _seq()
    _write_frames(tmp_path, seq.seq_id, [1, 7, 13, 19, 25, 31])
    indices = available_frame_indices(str(tmp_path), seq.seq_id)

    sampled = sample_available_frame_indices(
        seq,
        indices,
        target_fps=3.0,
        fallback_stride=10,
    )

    assert indices == [1, 7, 13, 19, 25, 31]
    assert infer_extracted_frame_step(indices) == 6
    assert sampled == [1, 13, 25]


def test_hf_triplets_use_existing_sparse_frames(tmp_path):
    seq = _seq()
    _write_frames(tmp_path, seq.seq_id, [1, 7, 13, 19, 25, 31])

    rows = list(
        _triplet_rows_for_sequence(
            str(tmp_path),
            seq,
            target_fps=3.0,
            fallback_stride=10,
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["frame_idx_prev"] == 1
    assert row["frame_idx_curr"] == 13
    assert row["frame_idx_next"] == 25
    assert row["frame_stride"] == 12
    assert row["rgb_prev"]["bytes"] == b"frame-1"
    assert row["rgb_curr"]["bytes"] == b"frame-13"
    assert row["rgb_next"]["bytes"] == b"frame-25"


def test_realcolon_readme_declares_hf_split_data_files():
    readme = README_TEMPLATE.format(repo_id="owner/name")

    assert "configs:" in readme
    assert "- config_name: default" in readme
    assert "- split: train\n    path: data/train/**/*.parquet" in readme
    assert "- split: val\n    path: data/val/**/*.parquet" in readme
    assert "- split: test\n    path: data/test/**/*.parquet" in readme


def test_quality_filter_drops_triplet_with_black_frame(tmp_path):
    seq = _seq()
    textured = np.indices((32, 32)).sum(axis=0) % 2
    textured = np.repeat((textured * 120 + 60)[:, :, None], 3, axis=2)
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    _write_image_frames(
        tmp_path,
        seq.seq_id,
        {
            1: textured,
            7: textured,
            13: black,
            19: textured,
            25: textured,
            31: textured,
        },
    )
    state = _quality_state()

    rows = list(
        _triplet_rows_for_sequence(
            str(tmp_path),
            seq,
            target_fps=3.0,
            fallback_stride=10,
            quality_state=state,
            split_name="train",
        )
    )

    assert rows == []
    assert state.stats["quality_candidate_triplets"] == 1
    assert state.stats["quality_skipped_triplets"] == 1
    assert state.stats["quality_bad_frames"] == 1
    assert state.stats["quality_bad_frame_too_dark"] == 1


def test_process_split_stages_parquet_under_hf_data_tree(tmp_path):
    seq = _seq()
    _write_frames(tmp_path, seq.seq_id, [1, 7, 13, 19, 25, 31])
    staging_dir = tmp_path / "stage"
    args = Namespace(
        data_root=str(tmp_path),
        target_fps=3.0,
        fallback_stride=10,
        shard_size=1,
    )

    total = process_split("train", [seq], args, str(staging_dir))

    shard_path = staging_dir / "data" / "train" / "shard_00000.parquet"
    table = pq.read_table(shard_path)
    assert total == 1
    assert shard_path.is_file()
    assert table.column("frame_idx_curr").to_pylist() == [13]
    assert table.column("frame_stride").to_pylist() == [12]


def test_process_split_resumes_existing_staged_rows(tmp_path):
    seq = _seq(num_frames=43)
    _write_frames(tmp_path, seq.seq_id, [1, 7, 13, 19, 25, 31, 37, 43])
    staging_dir = tmp_path / "stage"
    split_dir = staging_dir / "data" / "train"
    split_dir.mkdir(parents=True)
    args = Namespace(
        data_root=str(tmp_path),
        target_fps=3.0,
        fallback_stride=10,
        shard_size=1,
    )
    first_row = list(
        _triplet_rows_for_sequence(
            str(tmp_path),
            seq,
            target_fps=3.0,
            fallback_stride=10,
        )
    )[0]
    write_parquet([first_row], split_dir / "shard_00000.parquet")

    total = process_split("train", [seq], args, str(staging_dir))

    shard_0 = pq.read_table(split_dir / "shard_00000.parquet")
    shard_1 = pq.read_table(split_dir / "shard_00001.parquet")
    assert total == 2
    assert shard_0.column("frame_idx_curr").to_pylist() == [13]
    assert shard_1.column("frame_idx_curr").to_pylist() == [25]
