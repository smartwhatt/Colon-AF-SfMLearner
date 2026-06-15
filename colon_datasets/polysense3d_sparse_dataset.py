"""PolypSense3D clinical sparse-depth dataset helpers.

The clinical sparse-depth subset stores one RGB frame and one LabelMe JSON per
case under ``labeled_data/depth``. The public PolypSense3D evaluation code
scores only points labelled ``value`` and uses ``depth_gt.xlsx`` as the
forceps-derived metric depth target in millimetres.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
import torch


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_CELL_RE = re.compile(r"([A-Z]+)([0-9]+)")


@dataclass(frozen=True)
class SparseDepthCase:
    """Metadata for one PolypSense3D sparse-depth frame."""

    sample_id: str
    group_id: str
    rgb_path: Path
    json_path: Path
    gt_depth_mm: float
    image_width: int
    image_height: int
    image_path: str
    value_points: np.ndarray
    top_points: np.ndarray
    ignored_labels: tuple[str, ...]


def _column_index(cell_ref: str) -> int:
    match = _CELL_RE.fullmatch(cell_ref)
    if match is None:
        raise ValueError(f"Bad XLSX cell reference: {cell_ref}")
    col = match.group(1)
    idx = 0
    for char in col:
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def read_xlsx_rows(path: str | Path) -> list[list[str]]:
    """Read the first worksheet of a simple XLSX file into string rows.

    This intentionally avoids adding an ``openpyxl`` dependency. It supports
    shared strings and inline strings, which is enough for the PolypSense3D
    metadata spreadsheets used by the clinical benchmark.
    """
    path = Path(path)
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_XLSX_NS}si"):
                text = "".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t"))
                shared_strings.append(text)

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row_node in sheet.findall(f".//{_XLSX_NS}row"):
            values: dict[int, str] = {}
            max_idx = -1
            for cell in row_node.findall(f"{_XLSX_NS}c"):
                ref = cell.get("r")
                if ref is None:
                    continue
                idx = _column_index(ref)
                max_idx = max(max_idx, idx)
                cell_type = cell.get("t")
                value_node = cell.find(f"{_XLSX_NS}v")
                inline_node = cell.find(f"{_XLSX_NS}is")

                value = ""
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or "0")]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(
                        node.text or "" for node in inline_node.iter(f"{_XLSX_NS}t")
                    )
                elif value_node is not None:
                    value = value_node.text or ""
                values[idx] = value

            if max_idx >= 0:
                rows.append([values.get(i, "") for i in range(max_idx + 1)])
        return rows


def read_depth_gt_xlsx(path: str | Path) -> dict[str, float]:
    """Return ``{sample_id: gt_depth_mm}`` from ``depth_gt.xlsx``."""
    rows = read_xlsx_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    header = [cell.strip() for cell in rows[0]]
    try:
        id_col = header.index("id")
        gt_col = header.index("gt(mm)")
    except ValueError as exc:
        raise ValueError(f"Expected columns 'id' and 'gt(mm)' in {path}") from exc

    out: dict[str, float] = {}
    for row in rows[1:]:
        if len(row) <= max(id_col, gt_col):
            continue
        sample_id = row[id_col].strip()
        gt_text = row[gt_col].strip()
        if not sample_id or not gt_text:
            continue
        out[sample_id] = float(gt_text)
    return out


def read_video_camera_map(path: str | Path) -> dict[str, str]:
    """Return ``{sequence_name: camera_id}`` from ``video_info.xlsx``."""
    rows = read_xlsx_rows(path)
    if not rows:
        return {}
    header = [cell.strip() for cell in rows[0]]
    try:
        video_col = header.index("video_name")
        camera_col = header.index("camera_id")
    except ValueError:
        return {}

    mapping: dict[str, str] = {}
    for row in rows[1:]:
        if len(row) <= max(video_col, camera_col):
            continue
        video_name = row[video_col].strip()
        camera_id = row[camera_col].strip()
        if video_name and camera_id:
            mapping[video_name] = camera_id
    return mapping


def parse_sparse_depth_json(path: str | Path) -> dict[str, Any]:
    """Parse a LabelMe sparse-depth annotation file."""
    path = Path(path)
    data = json.loads(path.read_text())
    image_width = int(data["imageWidth"])
    image_height = int(data["imageHeight"])
    image_path = str(data.get("imagePath", ""))

    value_points: list[list[float]] = []
    top_points: list[list[float]] = []
    ignored_labels: list[str] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        points = shape.get("points") or []
        if len(points) != 1:
            ignored_labels.append(label)
            continue
        point = [float(points[0][0]), float(points[0][1])]
        if label == "value":
            value_points.append(point)
        elif label == "top":
            top_points.append(point)
        else:
            ignored_labels.append(label)

    return {
        "image_width": image_width,
        "image_height": image_height,
        "image_path": image_path,
        "value_points": np.asarray(value_points, dtype=np.float32).reshape(-1, 2),
        "top_points": np.asarray(top_points, dtype=np.float32).reshape(-1, 2),
        "ignored_labels": tuple(ignored_labels),
    }


def scale_points(
    points_xy: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale ``(x, y)`` points from source ``(width, height)`` to target ``(H, W)``."""
    if points_xy.size == 0:
        return points_xy.astype(np.float32).reshape(-1, 2)
    source_w, source_h = source_size
    target_h, target_w = target_size
    scaled = points_xy.astype(np.float32).copy()
    scaled[:, 0] *= float(target_w) / float(source_w)
    scaled[:, 1] *= float(target_h) / float(source_h)
    return scaled


def sample_depth_at_points(
    depth_hw: np.ndarray,
    points_xy: np.ndarray,
    *,
    radius_px: int = 0,
) -> np.ndarray:
    """Sample a depth map at sparse ``(x, y)`` points using nearest or patch median."""
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    depth = np.asarray(depth_hw)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")

    h, w = depth.shape
    values: list[float] = []
    for x_f, y_f in np.asarray(points_xy, dtype=np.float32):
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        if radius_px == 0:
            values.append(float(depth[y, x]))
            continue
        x0 = max(0, x - radius_px)
        x1 = min(w, x + radius_px + 1)
        y0 = max(0, y - radius_px)
        y1 = min(h, y + radius_px + 1)
        values.append(float(np.median(depth[y0:y1, x0:x1])))
    return np.asarray(values, dtype=np.float32)


class PolypSense3DSparseDepthDataset(torch.utils.data.Dataset):
    """Map-style dataset for PolypSense3D clinical sparse-depth evaluation."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        target_size: tuple[int, int] = (256, 320),
        strict: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.depth_root = self.data_root / "labeled_data" / "depth"
        self.rgb_dir = self.depth_root / "rgb"
        self.json_dir = self.depth_root / "json"
        self.gt_path = self.depth_root / "depth_gt.xlsx"
        self.video_info_path = self.data_root / "calibrations" / "video_info.xlsx"

        self.camera_by_sequence = (
            read_video_camera_map(self.video_info_path)
            if self.video_info_path.is_file()
            else {}
        )
        gt_by_id = read_depth_gt_xlsx(self.gt_path)
        self.cases: list[SparseDepthCase] = []
        errors: list[str] = []

        for sample_id in sorted(gt_by_id):
            rgb_path = self.rgb_dir / f"{sample_id}.jpg"
            json_path = self.json_dir / f"{sample_id}.json"
            if not rgb_path.is_file():
                errors.append(f"missing RGB for {sample_id}: {rgb_path}")
                continue
            if not json_path.is_file():
                errors.append(f"missing JSON for {sample_id}: {json_path}")
                continue
            parsed = parse_sparse_depth_json(json_path)
            value_points = parsed["value_points"]
            if len(value_points) == 0:
                errors.append(f"no value points in {json_path}")
                continue

            group_id = sample_id.split("_", 1)[0]
            self.cases.append(
                SparseDepthCase(
                    sample_id=sample_id,
                    group_id=group_id,
                    rgb_path=rgb_path,
                    json_path=json_path,
                    gt_depth_mm=float(gt_by_id[sample_id]),
                    image_width=int(parsed["image_width"]),
                    image_height=int(parsed["image_height"]),
                    image_path=str(parsed["image_path"]),
                    value_points=value_points,
                    top_points=parsed["top_points"],
                    ignored_labels=parsed["ignored_labels"],
                )
            )

        if strict and errors:
            preview = "\n".join(errors[:20])
            raise FileNotFoundError(
                f"PolypSense3D sparse-depth manifest has {len(errors)} error(s):\n"
                f"{preview}"
            )
        self.errors = errors

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index]
        image_bgr = cv2.imread(str(case.rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read RGB image: {case.rgb_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        color = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
        value_points_model = scale_points(
            case.value_points,
            source_size=(case.image_width, case.image_height),
            target_size=self.target_size,
        )
        top_points_model = scale_points(
            case.top_points,
            source_size=(case.image_width, case.image_height),
            target_size=self.target_size,
        )
        sequence_name = f"sequence{int(case.group_id):03d}"
        return {
            "color_curr": color,
            "sample_id": case.sample_id,
            "group_id": case.group_id,
            "sequence": sequence_name,
            "camera_id": self.camera_by_sequence.get(sequence_name, ""),
            "rgb_path": str(case.rgb_path),
            "json_path": str(case.json_path),
            "image_path": case.image_path,
            "native_width": case.image_width,
            "native_height": case.image_height,
            "gt_depth_mm": float(case.gt_depth_mm),
            "gt_depth_m": float(case.gt_depth_mm / 1000.0),
            "value_points": torch.from_numpy(value_points_model).float(),
            "value_points_native": torch.from_numpy(case.value_points).float(),
            "top_points": torch.from_numpy(top_points_model).float(),
            "top_points_native": torch.from_numpy(case.top_points).float(),
            "ignored_labels": case.ignored_labels,
        }
