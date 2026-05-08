from __future__ import annotations

import io
import random
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from .c3vdv2_dataset import pose_to_axisangle_translation


C3VD_DEPTH_MAX_M = 0.1
REAL_COLON_K_NORM = np.array(
    [[0.82, 0.0, 0.5], [0.0, 1.02, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
)


def load_hf_split(
    repo_id: str,
    split: str,
    streaming: bool,
    cache_dir: str | None = None,
):
    """Import Hugging Face datasets lazily so local-only training has no import cost."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires `datasets[vision]`. "
            "Install project dependencies from pyproject.toml before using --data_source hf."
        ) from exc

    return load_dataset(repo_id, split=split, streaming=streaming, cache_dir=cache_dir)


def build_hf_triplet_dataset(
    repo_id: str,
    split: str,
    streaming: bool,
    cache_dir: str | None,
    target_size: tuple[int, int],
    is_train: bool,
    output_format: str,
    scales: Iterable[int],
    shuffle_buffer: int,
    seed: int = 42,
):
    hf_dataset = load_hf_split(repo_id, split, streaming, cache_dir)
    common_kwargs = dict(
        target_size=target_size,
        is_train=is_train,
        output_format=output_format,
        scales=tuple(scales),
        seed=seed,
    )
    if streaming:
        return HFTripletIterableDataset(
            hf_dataset,
            shuffle_buffer=shuffle_buffer if is_train else 0,
            **common_kwargs,
        )
    return HFTripletMapDataset(hf_dataset, **common_kwargs)


class _HFTripletMixin:
    def __init__(
        self,
        *,
        target_size: tuple[int, int],
        is_train: bool,
        output_format: str,
        scales: tuple[int, ...],
        seed: int,
    ) -> None:
        if output_format not in {"c3vdv2", "trainer"}:
            raise ValueError("output_format must be 'c3vdv2' or 'trainer'")
        self.target_size = target_size
        self.is_train = is_train
        self.output_format = output_format
        self.scales = scales
        self.seed = seed
        self.epoch = 0
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _convert_row(self, row: dict[str, Any]) -> dict:
        decoded = self._decode_row(row)
        if self.output_format == "trainer":
            return self._format_trainer(decoded)
        return self._format_c3vdv2(decoded)

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        rgb_prev = self._load_rgb(row["rgb_prev"])
        rgb_curr = self._load_rgb(row["rgb_curr"])
        rgb_next = self._load_rgb(row["rgb_next"])

        aug_prev, aug_curr, aug_next = rgb_prev, rgb_curr, rgb_next
        if self.is_train:
            aug_prev, aug_curr, aug_next = self._augment_triplet(
                [rgb_prev, rgb_curr, rgb_next]
            )

        decoded = {
            "rgb_prev": rgb_prev,
            "rgb_curr": rgb_curr,
            "rgb_next": rgb_next,
            "rgb_prev_aug": aug_prev,
            "rgb_curr_aug": aug_curr,
            "rgb_next_aug": aug_next,
            "K": self._load_K(row),
        }

        if self._row_bool(row, "has_depth", row.get("depth") is not None):
            depth = self._load_depth(row.get("depth"))
            if depth is not None:
                decoded["gt_depth"] = depth

        if self._row_bool(row, "has_occlusion", row.get("occlusion") is not None):
            occlusion = self._load_occlusion(row.get("occlusion"))
            if occlusion is not None:
                decoded["occlusion"] = occlusion

        if self._row_bool(row, "has_pose", row.get("pose_curr2prev") is not None):
            self._add_pose(decoded, row, "prev", "pose_curr2prev")
            self._add_pose(decoded, row, "next", "pose_curr2next")

        return decoded

    def _format_c3vdv2(self, decoded: dict[str, Any]) -> dict:
        out = {
            "color_prev": decoded["rgb_prev_aug"],
            "color_curr": decoded["rgb_curr_aug"],
            "color_next": decoded["rgb_next_aug"],
            "K": decoded["K"],
        }
        for key in (
            "gt_depth",
            "occlusion",
            "axang_curr2prev",
            "trans_curr2prev",
            "axang_curr2next",
            "trans_curr2next",
        ):
            if key in decoded:
                out[key] = decoded[key]
        return out

    def _format_trainer(self, decoded: dict[str, Any]) -> dict:
        frames = {
            -1: ("rgb_prev", "rgb_prev_aug"),
            0: ("rgb_curr", "rgb_curr_aug"),
            1: ("rgb_next", "rgb_next_aug"),
        }
        out = {}
        for frame_id, (raw_key, aug_key) in frames.items():
            out[("color", frame_id, 0)] = decoded[raw_key]
            out[("color_aug", frame_id, 0)] = decoded[aug_key]

        for scale in self.scales:
            if scale == 0:
                continue
            h = self.target_size[0] // (2 ** scale)
            w = self.target_size[1] // (2 ** scale)
            for frame_id, (raw_key, aug_key) in frames.items():
                out[("color", frame_id, scale)] = self._resize_tensor(
                    decoded[raw_key], (h, w), mode="bilinear"
                )
                out[("color_aug", frame_id, scale)] = self._resize_tensor(
                    decoded[aug_key], (h, w), mode="bilinear"
                )

        for scale in self.scales:
            h = self.target_size[0] // (2 ** scale)
            w = self.target_size[1] // (2 ** scale)
            K = torch.zeros(4, 4, dtype=torch.float32)
            K[:3, :3] = decoded["K"]
            K[0, :] *= w
            K[1, :] *= h
            K[3, 3] = 1.0
            out[("K", scale)] = K
            out[("inv_K", scale)] = torch.linalg.pinv(K)

        for key in (
            "gt_depth",
            "occlusion",
            "axang_curr2prev",
            "trans_curr2prev",
            "axang_curr2next",
            "trans_curr2next",
        ):
            if key in decoded:
                out[key] = decoded[key]
        return out

    def _load_rgb(self, value: Any) -> torch.Tensor:
        image = self._as_pil(value).convert("RGB")
        image = image.resize(
            (self.target_size[1], self.target_size[0]), self._pil_lanczos()
        )
        return self.to_tensor(image)

    def _load_depth(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        raw = np.array(self._as_pil(value))
        depth = raw.astype(np.float32) / 65535.0 * C3VD_DEPTH_MAX_M
        depth = cv2.resize(
            depth,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return torch.from_numpy(depth).unsqueeze(0).float()

    def _load_occlusion(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        raw = np.array(self._as_pil(value).convert("L"))
        mask = (raw > 0).astype(np.float32)
        mask = cv2.resize(
            mask,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return torch.from_numpy(mask).unsqueeze(0).float()

    def _load_K(self, row: dict[str, Any]) -> torch.Tensor:
        vals = row.get("K")
        if vals is None:
            return torch.from_numpy(REAL_COLON_K_NORM.copy())
        K = np.asarray(vals, dtype=np.float32)
        if K.size == 16:
            K = K.reshape(4, 4)[:3, :3]
        elif K.size == 9:
            K = K.reshape(3, 3)
        else:
            raise ValueError(f"Expected K with 9 or 16 values, got {K.size}")
        return torch.from_numpy(K.astype(np.float32))

    def _add_pose(
        self,
        decoded: dict[str, Any],
        row: dict[str, Any],
        suffix: str,
        field_name: str,
    ) -> None:
        vals = row.get(field_name)
        if vals is None:
            return
        T = np.asarray(vals, dtype=np.float64).reshape(4, 4)
        axang, trans = pose_to_axisangle_translation(T)
        decoded[f"axang_curr2{suffix}"] = torch.from_numpy(axang)
        decoded[f"trans_curr2{suffix}"] = torch.from_numpy(trans)

    def _augment_triplet(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        pil_images = [transforms.ToPILImage()(image) for image in images]
        params = transforms.ColorJitter.get_params(
            self.color_jitter.brightness,
            self.color_jitter.contrast,
            self.color_jitter.saturation,
            self.color_jitter.hue,
        )
        if callable(params):
            aug = params
        else:
            _, brightness, contrast, saturation, hue = params
            aug = transforms.ColorJitter(
                brightness=(brightness, brightness),
                contrast=(contrast, contrast),
                saturation=(saturation, saturation),
                hue=(hue, hue),
            )
        return [self.to_tensor(aug(image)) for image in pil_images]

    @staticmethod
    def _as_pil(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                image = Image.open(io.BytesIO(value["bytes"]))
                image.load()
                return image
            if value.get("path") is not None:
                image = Image.open(value["path"])
                image.load()
                return image
        if isinstance(value, (bytes, bytearray)):
            image = Image.open(io.BytesIO(value))
            image.load()
            return image
        if isinstance(value, np.ndarray):
            return Image.fromarray(value)
        raise TypeError(f"Unsupported image value type: {type(value)!r}")

    @staticmethod
    def _row_bool(row: dict[str, Any], key: str, fallback: bool) -> bool:
        value = row.get(key, fallback)
        if value is None:
            return False
        return bool(value)

    @staticmethod
    def _resize_tensor(
        tensor: torch.Tensor, size: tuple[int, int], mode: str
    ) -> torch.Tensor:
        kwargs = {"mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        return F.interpolate(tensor.unsqueeze(0), size=size, **kwargs).squeeze(0)

    @staticmethod
    def _pil_lanczos():
        if hasattr(Image, "Resampling"):
            return Image.Resampling.LANCZOS
        return Image.LANCZOS


class HFTripletMapDataset(_HFTripletMixin, data.Dataset):
    def __init__(self, hf_dataset, **kwargs) -> None:
        self.hf_dataset = hf_dataset
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict:
        return self._convert_row(self.hf_dataset[index])


class HFTripletIterableDataset(_HFTripletMixin, data.IterableDataset):
    def __init__(self, hf_dataset, shuffle_buffer: int, **kwargs) -> None:
        self.hf_dataset = hf_dataset
        self.shuffle_buffer = shuffle_buffer
        super().__init__(**kwargs)

    def __iter__(self):
        source = self.hf_dataset
        if self.shuffle_buffer > 0:
            source = source.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=self.seed + self.epoch,
            )
        if hasattr(source, "set_epoch"):
            source.set_epoch(self.epoch)
        for row in source:
            yield self._convert_row(row)
