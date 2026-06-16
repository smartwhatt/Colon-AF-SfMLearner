import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.evaluate_polysense3d_sparse_depth as polysense_eval
from colon_datasets.polysense3d_sparse_dataset import (
    PolypSense3DSparseDepthDataset,
    parse_sparse_depth_json,
    read_depth_gt_xlsx,
    read_video_camera_map,
    sample_depth_at_points,
    scale_points,
)
from tools.evaluate_polysense3d_sparse_depth import (
    _metrics_from_arrays,
    _official_group_rows,
    _paper_eval_metrics,
)


def _cell_ref(col_idx: int, row_idx: int) -> str:
    col = ""
    idx = col_idx + 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        col = chr(ord("A") + rem) + col
    return f"{col}{row_idx}"


def _write_xlsx(path: Path, rows: list[list[str]]) -> None:
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = _cell_ref(col_idx, row_idx)
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def _write_labelme_json(path: Path, sample_id: str) -> None:
    payload = {
        "version": "2.5.4",
        "flags": {},
        "shapes": [
            {"label": "top", "points": [[10.0, 2.0]], "shape_type": "point"},
            {"label": "width", "points": [[11.0, 2.0]], "shape_type": "point"},
            {"label": "value", "points": [[5.0, 2.0]], "shape_type": "point"},
            {"label": "value", "points": [[10.0, 4.0]], "shape_type": "point"},
            {"label": "value", "points": [[15.0, 6.0]], "shape_type": "point"},
        ],
        "imagePath": f"{sample_id}.jpg",
        "imageHeight": 10,
        "imageWidth": 20,
    }
    path.write_text(json.dumps(payload))


def _make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "polysense"
    depth = root / "labeled_data" / "depth"
    (depth / "json").mkdir(parents=True)
    (depth / "rgb").mkdir(parents=True)
    (root / "calibrations").mkdir()

    _write_xlsx(
        depth / "depth_gt.xlsx",
        [["id", "gt(mm)"], ["0001_0001", "12.5"], ["0001_0002", "13.5"]],
    )
    _write_xlsx(
        root / "calibrations" / "video_info.xlsx",
        [["id", "video_name", "camera_id"], ["1", "sequence001", "42"]],
    )
    for sample_id in ("0001_0001", "0001_0002"):
        _write_labelme_json(depth / "json" / f"{sample_id}.json", sample_id)
        image = np.full((10, 20, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(depth / "rgb" / f"{sample_id}.jpg"), image)
    return root


def test_xlsx_parsers_read_depth_gt_and_camera_map(tmp_path):
    root = _make_dataset_root(tmp_path)

    depth_gt = read_depth_gt_xlsx(root / "labeled_data" / "depth" / "depth_gt.xlsx")
    camera_map = read_video_camera_map(root / "calibrations" / "video_info.xlsx")

    assert depth_gt == {"0001_0001": 12.5, "0001_0002": 13.5}
    assert camera_map == {"sequence001": "42"}


def test_labelme_parser_scores_value_points_only(tmp_path):
    json_path = tmp_path / "case.json"
    _write_labelme_json(json_path, "0001_0001")

    parsed = parse_sparse_depth_json(json_path)

    assert parsed["value_points"].shape == (3, 2)
    assert parsed["top_points"].shape == (1, 2)
    assert parsed["ignored_labels"] == ("width",)


def test_scale_and_sample_sparse_points():
    points = np.array([[5.0, 2.0], [10.0, 4.0]], dtype=np.float32)
    scaled = scale_points(points, source_size=(20, 10), target_size=(5, 10))

    assert scaled.tolist() == [[2.5, 1.0], [5.0, 2.0]]

    depth = np.arange(50, dtype=np.float32).reshape(5, 10)
    sampled = sample_depth_at_points(depth, scaled)
    assert sampled.tolist() == [12.0, 25.0]

    patch_sampled = sample_depth_at_points(depth, scaled[:1], radius_px=1)
    assert patch_sampled[0] == pytest.approx(np.median(depth[0:3, 1:4]))


def test_polysense_dataset_loads_rgb_points_and_metadata(tmp_path):
    root = _make_dataset_root(tmp_path)
    dataset = PolypSense3DSparseDepthDataset(root, target_size=(5, 10))

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["color_curr"].shape == (3, 5, 10)
    assert sample["sample_id"] == "0001_0001"
    assert sample["group_id"] == "0001"
    assert sample["camera_id"] == "42"
    assert sample["gt_depth_mm"] == 12.5
    assert sample["value_points"].shape == (3, 2)
    assert sample["top_points"].shape == (1, 2)
    assert sample["ignored_labels"] == ("width",)


def test_sparse_metrics_and_official_grouping():
    metrics = _metrics_from_arrays([10.0, 20.0], [10.0, 10.0])
    assert metrics["count"] == 2
    assert metrics["mae_mm"] == pytest.approx(5.0)
    assert metrics["rmse_mm"] > metrics["mae_mm"]

    frame_rows = [
        {
            "sample_id": "0001_0001",
            "group_id": "0001",
            "paper_pred_mean_mm": 10.0,
            "gt_depth_mm": 10.0,
            "paper_mae_mm": 0.0,
        },
        {
            "sample_id": "0001_0002",
            "group_id": "0001",
            "paper_pred_mean_mm": 12.0,
            "gt_depth_mm": 10.0,
            "paper_mae_mm": 2.0,
        },
        {
            "sample_id": "0001_0003",
            "group_id": "0001",
            "paper_pred_mean_mm": 9.5,
            "gt_depth_mm": 10.0,
            "paper_mae_mm": 10.0,
        },
        {
            "sample_id": "0002_0001",
            "group_id": "0002",
            "paper_pred_mean_mm": 8.0,
            "gt_depth_mm": 10.0,
            "paper_mae_mm": 2.0,
        },
    ]

    rows = _official_group_rows(frame_rows)

    assert len(rows) == 2
    assert rows[0]["selected_sample_ids"] == "0001_0001,0001_0002"
    assert rows[0]["pred_mm"] == pytest.approx(11.0)

    paper_metrics = _paper_eval_metrics(rows)
    assert paper_metrics["mae"] == pytest.approx(1.5)
    assert paper_metrics["a1_percent"] == pytest.approx(50.0)


def test_evaluator_writes_outputs_with_mocked_model(tmp_path, monkeypatch):
    root = _make_dataset_root(tmp_path)
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "encoder.pth").write_bytes(b"stub")
    (weights / "depth.pth").write_bytes(b"stub")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(polysense_eval, "_load_model", lambda weights_folder, device: object())
    monkeypatch.setattr(
        polysense_eval,
        "_predict_depth_m",
        lambda model, color, min_depth, max_depth: np.full((5, 10), 0.0125, dtype=np.float32),
    )

    args = type(
        "Args",
        (),
        {
            "data_root": str(root),
            "weights_folder": str(weights),
            "out_dir": str(out_dir),
            "run_id": "fixture",
            "height": 5,
            "width": 10,
            "min_depth": 0.001,
            "max_depth": 0.1,
            "sample_radius_px": 0,
            "paper_num_points": 3,
            "median_scaling": True,
            "topk": 1,
            "depth_vis_min": 0.001,
            "depth_vis_max": 0.1,
            "allow_manifest_errors": False,
            "cpu": True,
        },
    )()

    polysense_eval.evaluate(args)

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["dataset"]["frames"] == 2
    assert summary["dataset"]["points"] == 6
    assert summary["metrics"]["all_points"]["mae_mm"] == pytest.approx(0.5)
    assert summary["polysense3d_paper_style"]["raw_metrics"]["mae"] == pytest.approx(0.5)
    assert "median_scaled_metrics" in summary["polysense3d_paper_style"]
    assert (out_dir / "per_point_metrics.csv").is_file()
    assert (out_dir / "official_group_summary.csv").is_file()
    assert (out_dir / "paper_depth_all.csv").is_file()
    assert (out_dir / "paper_depth_summary.csv").is_file()
    assert (out_dir / "paper_depth_eval_summary.json").is_file()
    assert (out_dir / "paper_depth_summary_median_scaled.csv").is_file()
    assert (out_dir / "paper_depth_eval_median_scaled_summary.json").is_file()
    assert (out_dir / "topk" / "topk_index.json").is_file()
