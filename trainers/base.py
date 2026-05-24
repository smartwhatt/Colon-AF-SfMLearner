"""
trainers/base.py
----------------
BaseTrainer holds all infrastructure shared across every training mode:
  - model setup / optimizer / data loading / TensorBoard
  - train / run_epoch / val loop
  - shared forward methods (predict_poses, generate_images_pred)
  - shared loss helpers (compute_reprojection_loss, compute_losses, compute_losses_val)
  - logging, saving, loading

Subclasses override:
  - build_models()          → populate self.models and self.parameters_to_train
  - set_train() / set_eval()
  - process_batch()
  - compute_losses()
  - log()                   (optional – default logs stage-2 style outputs)
  - build_dataloaders()     (optional – default uses SCARED splits/*.txt layout)
  - load_model()            (optional – default does NOT freeze loaded params)
"""

from __future__ import annotations

import os
import time
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import colon_datasets
import networks
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

if TYPE_CHECKING:
    from options import TrainingConfig

from layers import (
    SSIM,
    SpatialTransformer,
    get_occu_mask_backward,
    get_occu_mask_bidirection,
    BackprojectDepth,
    Project3D,
    optical_flow,
    disp_to_depth,
    transformation_from_parameters,
    get_smooth_loss,
    get_smooth_bright,
    ncc_loss,
)
from utils import readlines, sec_to_hm_str, normalize_image
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


# Registry: maps --dataset argument to dataset class
DATASET_REGISTRY = {
    "endovis": colon_datasets.SCAREDRAWDataset,
    "real_colon": colon_datasets.REALColonDataset,
}

DATASET_IMAGE_EXT = {
    "endovis": ".png",
    "real_colon": ".jpg",
}


class BaseTrainer(ABC):
    """Abstract base trainer. Subclasses must implement build_models, set_train,
    set_eval, process_batch, and compute_losses."""

    opt: "TrainingConfig"

    def __init__(self, options: "TrainingConfig") -> None:
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = (
            2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        )
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ── Models ────────────────────────────────────────────────────────────
        self.models = {}
        self.parameters_to_train = []
        self.build_models()

        # if there are multiple gpus and model is not on cpu, use DataParallel
        if torch.cuda.device_count() > 1 and not self.opt.no_cuda:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            for model in self.models.values():
                model = torch.nn.DataParallel(model)

        # ── Optimizers ────────────────────────────────────────────────────────
        self.build_optimizers()

        # ── Load pretrained weights ───────────────────────────────────────────
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.best_model_metric = None
        self.best_model_metadata = None

        # ── Data loaders ──────────────────────────────────────────────────────
        self.build_dataloaders()

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # ── Geometry layers ───────────────────────────────────────────────────
        self._setup_geometry()

        self.depth_metric_names = [
            "de/abs_rel",
            "de/sq_rel",
            "de/rms",
            "de/log_rms",
            "da/a1",
            "da/a2",
            "da/a3",
        ]
        self.save_opts()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_models(self) -> None: ...

    @abstractmethod
    def set_train(self) -> None: ...

    @abstractmethod
    def set_eval(self) -> None: ...

    @abstractmethod
    def process_batch(self, inputs: dict) -> tuple[dict, dict]: ...

    @abstractmethod
    def compute_losses(self, inputs: dict, outputs: dict) -> dict: ...

    # ── Optional overrides ────────────────────────────────────────────────────

    def build_optimizers(self):
        """Single Adam optimizer over self.parameters_to_train."""
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )

    def build_dataloaders(self):
        """Default: SCARED-style splits/{split}/{train,val}_files.txt."""
        if self.opt.data_source == "hf":
            return self.build_hf_dataloaders()

        dataset_cls = DATASET_REGISTRY[self.opt.dataset]
        fpath = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "splits",
            self.opt.split,
            "{}_files.txt",
        )
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = DATASET_IMAGE_EXT.get(self.opt.dataset, ".png")

        train_dataset = dataset_cls(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=True,
            img_ext=img_ext,
        )
        val_dataset = dataset_cls(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=img_ext,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = (
            len(train_dataset) // self.opt.batch_size * self.opt.num_epochs
        )
        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)
            )
        )

    def build_hf_dataloaders(self):
        """Build loaders for the unified Hugging Face triplet schema."""
        if not self.opt.hf_repo_id:
            raise ValueError("--hf_repo_id is required when --data_source hf")

        from colon_datasets.hf_triplet_dataset import build_hf_triplet_dataset

        train_dataset = build_hf_triplet_dataset(
            repo_id=self.opt.hf_repo_id,
            split="train",
            streaming=self.opt.hf_streaming,
            cache_dir=self.opt.hf_cache_dir,
            target_size=(self.opt.height, self.opt.width),
            is_train=True,
            output_format="trainer",
            scales=self.opt.scales,
            shuffle_buffer=self.opt.hf_shuffle_buffer,
        )
        val_dataset = build_hf_triplet_dataset(
            repo_id=self.opt.hf_repo_id,
            split="val",
            streaming=self.opt.hf_streaming,
            cache_dir=self.opt.hf_cache_dir,
            target_size=(self.opt.height, self.opt.width),
            is_train=False,
            output_format="trainer",
            scales=self.opt.scales,
            shuffle_buffer=0,
        )

        # Avoid duplicated samples from wrapping a HF IterableDataset in multiple
        # PyTorch workers. Cached HF datasets still use normal worker loading.
        train_workers = 0 if self.opt.hf_streaming else self.opt.num_workers
        val_workers = 0 if self.opt.hf_streaming else 1
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            shuffle=not self.opt.hf_streaming,
            num_workers=train_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = self._num_total_steps(train_dataset)
        print(f"Using HF dataset: {self.opt.hf_repo_id}")
        print(
            "  mode={} train={} val={}\n".format(
                "streaming" if self.opt.hf_streaming else "cached",
                self._dataset_len(train_dataset),
                self._dataset_len(val_dataset),
            )
        )

    def _num_total_steps(self, train_dataset) -> int:
        if self.opt.steps_per_epoch is not None:
            return self.opt.steps_per_epoch * self.opt.num_epochs
        try:
            return len(train_dataset) // self.opt.batch_size * self.opt.num_epochs
        except TypeError:
            return 0

    @staticmethod
    def _dataset_len(dataset) -> str:
        try:
            return f"{len(dataset):,}"
        except TypeError:
            return "unknown"

    def load_model(self):
        """Load weights into self.models. Default: no freezing."""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(
            self.opt.load_weights_folder
        ), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if n not in self.models:
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        print("Adam is randomly initialized")

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        try:
            for self.epoch in range(self.opt.num_epochs):
                self.run_epoch()
                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()
        finally:
            self.close_writers()

    def run_epoch(self):
        """Default: single-optimizer step per batch."""
        print("Training")
        self.set_train()
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            if (
                self.opt.steps_per_epoch is not None
                and batch_idx + 1 >= self.opt.steps_per_epoch
            ):
                break

        self.model_lr_scheduler.step()

    # ── Validation ────────────────────────────────────────────────────────────

    def val(self):
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses)
            self._maybe_save_best_model("val/loss", losses["loss"], mode="min")
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        """Default val forward pass (stage-2 / end-to-end style)."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids]
            )
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {
                k: [f[i] for f in all_features]
                for i, k in enumerate(self.opt.frame_ids)
            }
            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)
        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """NCC-based validation loss (identical across all trainers)."""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            target = inputs[("color", 0, 0)]
            registration_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(
                        outputs[("registration", scale, frame_id)].mean(1, True),
                        target.mean(1, True),
                    )
                )
            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, _ = torch.min(registration_losses, dim=1)
            loss = -registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    # ── Shared forward methods ────────────────────────────────────────────────

    def predict_poses(self, inputs, features, disps):
        """Full forward: position → registration → occlusion → transform → pose.
        Shared between StageTwoTrainer and EndToEndTrainer (identical logic)."""
        outputs = {}
        if self.num_pose_frames != 2:
            return outputs

        if self.opt.pose_model_type == "shared":
            pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
        else:
            pose_feats = {
                f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids
            }

        for f_i in self.opt.frame_ids[1:]:
            if f_i == "s":
                continue

            inputs_all = [pose_feats[f_i], pose_feats[0]]
            inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

            # Optical flow (position)
            position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
            position_inputs_reverse = self.models["position_encoder"](
                torch.cat(inputs_all_reverse, 1)
            )
            outputs_0 = self.models["position"](position_inputs)
            outputs_1 = self.models["position"](position_inputs_reverse)

            for scale in self.opt.scales:
                outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                outputs[("position", "high", scale, f_i)] = F.interpolate(
                    outputs[("position", scale, f_i)],
                    [self.opt.height, self.opt.width],
                    mode="bilinear",
                    align_corners=False,
                )
                outputs[("registration", scale, f_i)] = self.spatial_transform(
                    inputs[("color", f_i, 0)],
                    outputs[("position", "high", scale, f_i)],
                )
                outputs[("position_reverse", scale, f_i)] = outputs_1[
                    ("position", scale)
                ]
                outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                    outputs[("position_reverse", scale, f_i)],
                    [self.opt.height, self.opt.width],
                    mode="bilinear",
                    align_corners=False,
                )
                (
                    outputs[("occu_mask_backward", scale, f_i)],
                    outputs[("occu_map_backward", scale, f_i)],
                ) = self.get_occu_mask_backward(
                    outputs[("position_reverse", "high", scale, f_i)]
                )
                outputs[("occu_map_bidirection", scale, f_i)] = (
                    self.get_occu_mask_bidirection(
                        outputs[("position", "high", scale, f_i)],
                        outputs[("position_reverse", "high", scale, f_i)],
                    )
                )

            # Appearance transform
            transform_input = [
                outputs[("registration", 0, f_i)],
                inputs[("color", 0, 0)],
            ]
            transform_inputs = self.models["transform_encoder"](
                torch.cat(transform_input, 1)
            )
            outputs_2 = self.models["transform"](transform_inputs)

            for scale in self.opt.scales:
                outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                outputs[("transform", "high", scale, f_i)] = F.interpolate(
                    outputs[("transform", scale, f_i)],
                    [self.opt.height, self.opt.width],
                    mode="bilinear",
                    align_corners=False,
                )
                outputs[("refined", scale, f_i)] = torch.clamp(
                    outputs[("transform", "high", scale, f_i)]
                    * outputs[("occu_mask_backward", 0, f_i)].detach()
                    + inputs[("color", 0, 0)],
                    min=0.0,
                    max=1.0,
                )

            # Pose
            pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0]
            )

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Depth → 3D → reproject: fill outputs with warped color frames."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if not self.opt.v1_multiscale:
                disp = F.interpolate(
                    disp,
                    [self.opt.height, self.opt.width],
                    mode="bilinear",
                    align_corners=False,
                )
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0] * mean_inv_depth[:, 0],
                        frame_id < 0,
                    )

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )
                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border",
                )
                outputs[("position_depth", scale, frame_id)] = self.position_depth[
                    source_scale
                ](cam_points, inputs[("K", source_scale)], T)

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, mode, inputs, outputs, losses):
        """Default: log stage-2 style outputs (brightness, registration, refined, disp)."""
        writer = self.writers[mode]
        self.log_scalars(mode, losses)

        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    writer.add_image(
                        "brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data,
                        self.step,
                    )
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data,
                        self.step,
                    )
                    writer.add_image(
                        "refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data,
                        self.step,
                    )
                    if s == 0:
                        writer.add_image(
                            "occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id)][j].data,
                            self.step,
                        )
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]),
                    self.step,
                )

    def log_scalars(self, mode, losses, prefix=None):
        writer = self.writers[mode]
        for tag, value in losses.items():
            if prefix is not None:
                tag = "{}/{}".format(prefix, tag)
            writer.add_scalar(tag, self._scalar_for_logging(value), self.step)

    @staticmethod
    def _scalar_for_logging(value):
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                return value.cpu().item()
            return value.cpu()
        return value

    def close_writers(self):
        for writer in getattr(self, "writers", {}).values():
            writer.flush()
            writer.close()

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps / self.step - 1.0) * time_sofar
            if self.step > 0 and self.num_total_steps > 0
            else 0
        )
        print(
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}"
            " | loss: {:.5f} | time elapsed: {} | time left: {}".format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
                sec_to_hm_str(time_sofar),
                sec_to_hm_str(training_time_left),
            )
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "opt.json"), "w") as f:
            json.dump(self.opt.__dict__.copy(), f, indent=2)

    def save_model(self, folder_name=None):
        if folder_name is None:
            folder_name = "weights_{}".format(self.epoch)
        save_folder = os.path.join(self.log_path, "models", folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            if model_name.startswith("teacher_"):
                continue
            to_save = model.state_dict()
            if model_name == "encoder":
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
                to_save["use_stereo"] = self.opt.use_stereo
            torch.save(to_save, os.path.join(save_folder, "{}.pth".format(model_name)))

        torch.save(
            self.model_optimizer.state_dict(),
            os.path.join(save_folder, "adam.pth"),
        )
        return save_folder

    def _maybe_save_best_model(self, metric_name, metric_value, mode="min"):
        value = self._scalar_for_logging(metric_value)
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        value = float(value)
        if not np.isfinite(value):
            return False

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        current_best = self.best_model_metric
        improved = (
            current_best is None
            or (mode == "min" and value < current_best)
            or (mode == "max" and value > current_best)
        )
        if not improved:
            return False

        self.best_model_metric = value
        self.best_model_metadata = {
            "metric": metric_name,
            "value": value,
            "mode": mode,
            "epoch": int(getattr(self, "epoch", -1)),
            "step": int(getattr(self, "step", -1)),
            "folder": "weights_best",
        }
        save_folder = self.save_model(folder_name="weights_best")
        metadata_path = os.path.join(save_folder, "best.json")
        with open(metadata_path, "w") as f:
            json.dump(self.best_model_metadata, f, indent=2)

        print(
            "Saved new best checkpoint: {} = {:.6f} at epoch {} step {} -> {}".format(
                metric_name,
                value,
                self.best_model_metadata["epoch"],
                self.best_model_metadata["step"],
                save_folder,
            )
        )
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _setup_geometry(self):
        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.spatial_transform = SpatialTransformer(
            (self.opt.height, self.opt.width)
        ).to(self.device)
        self.get_occu_mask_backward = get_occu_mask_backward(
            (self.opt.height, self.opt.width)
        ).to(self.device)
        self.get_occu_mask_bidirection = get_occu_mask_bidirection(
            (self.opt.height, self.opt.width)
        ).to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2**scale)
            w = self.opt.width // (2**scale)
            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w
            ).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(
                self.device
            )
            self.position_depth[scale] = optical_flow(
                (h, w), self.opt.batch_size, h, w
            ).to(self.device)
