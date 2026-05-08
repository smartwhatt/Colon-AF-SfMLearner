import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colon_datasets.hf_triplet_dataset import HFTripletMapDataset


def _rgb(value):
    return Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8), mode="RGB")


def _depth(value):
    return Image.fromarray(np.full((8, 8), value, dtype=np.uint16), mode="I;16")


def _mask(value):
    return Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L")


def _base_row():
    return {
        "dataset_name": "fixture",
        "sequence": "seq",
        "frame_idx": 2,
        "frame_idx_prev": 1,
        "frame_idx_curr": 2,
        "frame_idx_next": 3,
        "rgb_prev": _rgb(10),
        "rgb_curr": _rgb(20),
        "rgb_next": _rgb(30),
        "K": [0.82, 0.0, 0.5, 0.0, 1.02, 0.5, 0.0, 0.0, 1.0],
        "source_fps": 30.0,
        "target_fps": 3.0,
        "frame_stride": 10,
    }


def test_hf_triplet_c3vdv2_format_with_labels():
    row = _base_row()
    row.update(
        {
            "depth": _depth(32767),
            "occlusion": _mask(255),
            "pose_curr2prev": np.eye(4).reshape(-1).tolist(),
            "pose_curr2next": np.eye(4).reshape(-1).tolist(),
            "has_depth": True,
            "has_occlusion": True,
            "has_pose": True,
        }
    )
    dataset = HFTripletMapDataset(
        [row],
        target_size=(4, 6),
        is_train=False,
        output_format="c3vdv2",
        scales=(0, 1),
        seed=1,
    )

    sample = dataset[0]
    assert sample["color_curr"].shape == (3, 4, 6)
    assert sample["gt_depth"].shape == (1, 4, 6)
    assert sample["occlusion"].shape == (1, 4, 6)
    assert sample["K"].shape == (3, 3)
    assert sample["axang_curr2prev"].shape == (3,)
    assert sample["trans_curr2next"].shape == (3,)


def test_hf_triplet_trainer_format_without_labels():
    row = _base_row()
    row.update(
        {
            "depth": None,
            "occlusion": None,
            "pose_curr2prev": None,
            "pose_curr2next": None,
            "has_depth": False,
            "has_occlusion": False,
            "has_pose": False,
        }
    )
    dataset = HFTripletMapDataset(
        [row],
        target_size=(4, 6),
        is_train=False,
        output_format="trainer",
        scales=(0, 1),
        seed=1,
    )

    sample = dataset[0]
    assert sample[("color", 0, 0)].shape == (3, 4, 6)
    assert sample[("color_aug", -1, 1)].shape == (3, 2, 3)
    assert sample[("K", 0)].shape == (4, 4)
    assert sample[("inv_K", 1)].shape == (4, 4)
    assert "gt_depth" not in sample
    assert "axang_curr2prev" not in sample
