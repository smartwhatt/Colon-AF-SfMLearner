"""
trainers/c3vdv2.py
------------------
C3VDv2 fine-tuning variants of Stage 1 and Stage 2 trainers.

Both trainers:
  - Replace the SCARED dataloaders with C3VDv2Dataset
  - Fix Adam β2 to paper value (0.99)
  - Adapt C3VDv2Dataset key format to the tuple keys expected by trainers

C3VDv2Stage2Trainer additionally:
  - Adds supervised depth loss (scale-invariant log)
  - Adds supervised pose loss (axis-angle + translation regression)
  - Ramps supervised loss weights from high → low over training epochs
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from colon_datasets.c3vdv2_dataset import C3VDv2Dataset
from colon_datasets.c3vdv2_metrics import (
    C3VD_MAX_DEPTH,
    C3VD_MIN_DEPTH,
    build_c3vdv2_valid_mask_torch,
    depth_metric_sums_torch,
    finalize_depth_metric_sums,
    merge_depth_metric_sums,
)
from .c3vdv2_losses import (
    scale_invariant_log_loss,
    compute_specular_pseudo_labels,
    specular_bce_loss,
    pps_supervised_loss,
)
from .c3vdv2_replay import adapt_c3vdv2_batch, supervised_pose_loss_from_batch
from .base import BaseTrainer
from .stage_one import StageOneTrainer
from .stage_two import StageTwoTrainer

# Paper constants (Section 4.2)
PAPER_BETA1 = 0.9
PAPER_BETA2 = 0.99


# ── Shared mixin ─────────────────────────────────────────────────────────────

class _C3VDv2DataMixin(BaseTrainer):
    """
    Abstract mixin that replaces build_dataloaders() and provides _adapt_keys().

    Inherits from BaseTrainer so that:
      - type checkers resolve self.opt, super().process_batch(), etc.
      - Python's cooperative MRO chains correctly to the concrete trainer
        (e.g. C3VDv2Stage1Trainer → _C3VDv2DataMixin → StageOneTrainer → BaseTrainer)

    Never instantiate directly — always use via multiple inheritance with a
    concrete trainer: class C3VDv2Stage1Trainer(_C3VDv2DataMixin, StageOneTrainer).
    """

    # Satisfy ABC — these delegate to the concrete trainer via cooperative MRO.
    def build_models(self) -> None: super().build_models()  # type: ignore[abstract]
    def set_train(self) -> None: super().set_train()        # type: ignore[abstract]
    def set_eval(self) -> None: super().set_eval()          # type: ignore[abstract]
    def compute_losses(self, inputs: dict, outputs: dict) -> dict:
        return super().compute_losses(inputs, outputs)      # type: ignore[abstract]

    def build_dataloaders(self):
        self.dev_loader = None
        if self.opt.data_source == "hf":
            return self._build_hf_c3vdv2_dataloaders()

        undistort_maps = getattr(self.opt, "undistort_maps", "undistort_maps/")
        dev_split_file = self._local_dev_split_file()

        train_dataset = C3VDv2Dataset(
            data_root=self.opt.data_path,
            split_file=os.path.join("splits", self.opt.split, "train.txt"),
            undistort_maps=undistort_maps,
            target_size=(self.opt.height, self.opt.width),
            augment=True,
        )
        val_dataset = C3VDv2Dataset(
            data_root=self.opt.data_path,
            split_file=os.path.join("splits", self.opt.split, "val.txt"),
            undistort_maps=undistort_maps,
            target_size=(self.opt.height, self.opt.width),
            augment=False,
        )
        dev_dataset = None
        if dev_split_file is not None:
            dev_dataset = C3VDv2Dataset(
                data_root=self.opt.data_path,
                split_file=dev_split_file,
                undistort_maps=undistort_maps,
                target_size=(self.opt.height, self.opt.width),
                augment=False,
            )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            shuffle=True, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size,
            shuffle=False, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True,
        )
        if dev_dataset is not None:
            self.dev_loader = DataLoader(
                dev_dataset, self.opt.batch_size,
                shuffle=False, num_workers=self.opt.num_workers,
                pin_memory=True, drop_last=True,
            )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = (
            len(train_dataset) // self.opt.batch_size * self.opt.num_epochs
        )
        dev_count = 0 if dev_dataset is None else len(dev_dataset)
        print(
            f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}"
            f"  Dev: {dev_count:,}"
        )

    def _local_dev_split_file(self) -> str | None:
        explicit = getattr(self.opt, "c3vdv2_dev_split_file", None)
        if explicit:
            return os.path.expanduser(explicit)
        candidate = os.path.join("splits", self.opt.split, "dev.txt")
        return candidate if os.path.exists(candidate) else None

    def _build_hf_c3vdv2_dataloaders(self):
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
            output_format="c3vdv2",
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
            output_format="c3vdv2",
            scales=self.opt.scales,
            shuffle_buffer=0,
        )
        dev_dataset = None
        if self.opt.c3vdv2_dev_hf_split:
            dev_dataset = build_hf_triplet_dataset(
                repo_id=self.opt.hf_repo_id,
                split=self.opt.c3vdv2_dev_hf_split,
                streaming=self.opt.hf_streaming,
                cache_dir=self.opt.hf_cache_dir,
                target_size=(self.opt.height, self.opt.width),
                is_train=False,
                output_format="c3vdv2",
                scales=self.opt.scales,
                shuffle_buffer=0,
            )

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
            drop_last=True,
        )
        if dev_dataset is not None:
            self.dev_loader = DataLoader(
                dev_dataset,
                self.opt.batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                drop_last=True,
            )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = self._num_total_steps(train_dataset)
        print(f"Using HF C3VDv2 dataset: {self.opt.hf_repo_id}")
        print(
            "  mode={} train={} val={} dev={}".format(
                "streaming" if self.opt.hf_streaming else "cached",
                self._dataset_len(train_dataset),
                self._dataset_len(val_dataset),
                self._dataset_len(dev_dataset) if dev_dataset is not None else "disabled",
            )
        )

    def process_batch(self, inputs: dict) -> tuple:  # type: ignore[override]
        # super() resolves to the concrete trainer (StageOneTrainer / StageTwoTrainer)
        # via cooperative MRO — not to BaseTrainer's abstract method.
        return super().process_batch(self._adapt_keys(inputs))  # type: ignore[abstract]

    def process_batch_val(self, inputs: dict) -> tuple:
        return super().process_batch_val(self._adapt_keys(inputs))  # type: ignore[abstract]

    def _adapt_keys(self, batch):
        return adapt_c3vdv2_batch(batch, self.opt)


# ── Stage 1 fine-tune ────────────────────────────────────────────────────────

class C3VDv2Stage1Trainer(_C3VDv2DataMixin, StageOneTrainer):
    """Fine-tune the registration net (Stage 1) on C3VDv2 colonoscopy data.
    Self-supervised only — no GT depth or pose used."""

    def __init__(self, options):
        super().__init__(options)

        # Fix Adam β2 to paper value (0.99 vs PyTorch default 0.999)
        for pg in self.model_optimizer.param_groups:
            pg["betas"] = (PAPER_BETA1, PAPER_BETA2)

        print(f"\n{'='*60}")
        print("  Stage 1 Fine-Tuning  |  C3VDv2 Colonoscopy")
        print(f"{'='*60}")
        print(f"  Training:  position_encoder + position  (only)")
        print(f"  Loss:      photometric + smoothness  (no GT used)")
        print(f"  Resolution: {self.opt.width}x{self.opt.height}  Batch: {self.opt.batch_size}")
        print(f"  LR: {self.opt.learning_rate}  Epochs: {self.opt.num_epochs}")
        print(f"{'='*60}\n")

    def val(self):
        """Log fixed-batch C3VDv2 registration metrics for Stage 1.

        Stage 1 training uses photometric+smoothness loss. Log that same
        objective as val/loss for train-vs-val comparison, and log NCC
        registration quality under split-specific diagnostic tags.
        """
        if getattr(self.opt, "c3vdv2_val_batches", 0) <= 0:
            return super().val()

        self.set_eval()
        summaries = {}
        with torch.no_grad():
            if getattr(self, "dev_loader", None) is not None:
                summary = self._log_c3vdv2_registration_metrics(
                    "dev", self.dev_loader
                )
                if summary is not None:
                    summaries["dev"] = summary
            summary = self._log_c3vdv2_registration_metrics("ood_t4", self.val_loader)
            if summary is not None:
                summaries["ood_t4"] = summary

            primary_summary = summaries.get("dev") or summaries.get("ood_t4")
            if primary_summary is not None and "training_objective" in primary_summary:
                self.writers["val"].add_scalar(
                    "loss",
                    primary_summary["training_objective"],
                    self.step,
                )

            best_metric = self._best_metric_from_stage1_summaries(summaries)
            if best_metric is not None:
                metric_name, metric_value, mode = best_metric
                self._maybe_save_best_model(metric_name, metric_value, mode=mode)

        self.set_train()
        if not summaries:
            return super().val()

    @staticmethod
    def _best_metric_from_stage1_summaries(summaries):
        """Prefer in-domain dev NCC, then trajectory-OOD NCC."""
        for prefix in ("dev", "ood_t4"):
            summary = summaries.get(prefix)
            if summary and "ncc_registration" in summary:
                return f"{prefix}/ncc_registration", summary["ncc_registration"], "max"
        return None

    def _log_c3vdv2_registration_metrics(self, prefix: str, loader) -> dict | None:
        max_batches = max(int(getattr(self.opt, "c3vdv2_val_batches", 0)), 0)
        if max_batches == 0:
            return None

        ncc_total = 0.0
        objective_total = 0.0
        batches = 0
        for batch_idx, raw_inputs in enumerate(loader):
            if batch_idx >= max_batches:
                break

            inputs = self._adapt_keys(raw_inputs)
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            outputs = StageOneTrainer._predict_positions(self, inputs)
            objective_losses = StageOneTrainer.compute_losses(self, inputs, outputs)
            ncc_losses = StageOneTrainer.compute_losses_val(self, inputs, outputs)

            ncc_total += float(ncc_losses["loss"].detach().cpu())
            objective_total += float(objective_losses["loss"].detach().cpu())
            batches += 1
            del raw_inputs, inputs, outputs, objective_losses, ncc_losses

        if batches == 0:
            return None

        summary = {
            "ncc_registration": ncc_total / batches,
            "training_objective": objective_total / batches,
            "batches": batches,
        }
        writer = self.writers["val"]
        writer.add_scalar(
            f"{prefix}/training_objective",
            summary["training_objective"],
            self.step,
        )
        writer.add_scalar(
            f"{prefix}/ncc_registration",
            summary["ncc_registration"],
            self.step,
        )
        writer.add_scalar(f"{prefix}/batches", batches, self.step)
        return summary


# ── Stage 2 fine-tune ────────────────────────────────────────────────────────

class C3VDv2Stage2Trainer(_C3VDv2DataMixin, StageTwoTrainer):
    """
    Fine-tune Stage 2 on C3VDv2 with supervised depth + pose losses.

    Loss weights ramp from high GT supervision → lower as the self-supervised
    signal becomes reliable over training epochs.
    """

    def __init__(self, options):
        self.w_depth_start = getattr(options, "w_depth_start", 1.0)
        self.w_depth_end = getattr(options, "w_depth_end", 0.3)
        self.w_pose_start = getattr(options, "w_pose_start", 0.5)
        self.w_pose_end = getattr(options, "w_pose_end", 0.1)
        self.ramp_epochs = getattr(options, "ramp_epochs", 8)
        self.w_depth = self.w_depth_start
        self.w_pose = self.w_pose_start

        # Force correct depth range for C3VDv2 (1–100 mm).
        # The default options use outdoor values (0.1–100 m) which would make
        # disp_to_depth produce depths 1000× too large for this dataset.
        options.min_depth = C3VD_MIN_DEPTH
        options.max_depth = C3VD_MAX_DEPTH

        super().__init__(options)

        for pg in self.model_optimizer.param_groups:
            pg["betas"] = (PAPER_BETA1, PAPER_BETA2)

        print(f"\n{'='*60}")
        print("  Stage 2 Fine-Tuning  |  C3VDv2 Colonoscopy")
        print(f"{'='*60}")
        print(f"  Training:  encoder+depth  transform  pose_encoder+pose")
        print(f"  Frozen:    position_encoder + position  (Stage 1 weights)")
        print(f"  Resolution: {self.opt.width}x{self.opt.height}  Batch: {self.opt.batch_size}")
        print(f"  LR: {self.opt.learning_rate}  Epochs: {self.opt.num_epochs}")
        print(
            f"  w_depth: {self.w_depth_start} → {self.w_depth_end}  "
            f"w_pose: {self.w_pose_start} → {self.w_pose_end}  "
            f"over {self.ramp_epochs} epochs"
        )
        print(f"{'='*60}\n")

    def _update_loss_weights(self):
        if self.epoch < self.ramp_epochs:
            self.w_depth = self.w_depth_start
            self.w_pose = self.w_pose_start
        else:
            total_ramp = max(self.opt.num_epochs - self.ramp_epochs, 1)
            progress = min((self.epoch - self.ramp_epochs) / total_ramp, 1.0)
            self.w_depth = self.w_depth_start + progress * (
                self.w_depth_end - self.w_depth_start
            )
            self.w_pose = self.w_pose_start + progress * (
                self.w_pose_end - self.w_pose_start
            )
        if hasattr(self, "writers"):
            self.writers["train"].add_scalar("schedule/w_depth", self.w_depth, self.epoch)
            self.writers["train"].add_scalar("schedule/w_pose", self.w_pose, self.epoch)

    def run_epoch(self):
        self._update_loss_weights()
        print(f"  Epoch {self.epoch}  |  w_depth={self.w_depth:.3f}  w_pose={self.w_pose:.3f}")
        super().run_epoch()

    def val(self):
        """Log fixed-batch C3VDv2 validation metrics.

        The Stage 2 training objective is logged as val/loss for comparison
        against train/loss. Depth metrics and NCC registration diagnostics stay
        under split-specific tags so they are not confused with the objective.
        """
        if getattr(self.opt, "c3vdv2_val_batches", 0) <= 0:
            return super().val()

        self.set_eval()
        summaries = {}
        with torch.no_grad():
            if getattr(self, "dev_loader", None) is not None:
                summary = self._log_c3vdv2_loader_metrics("dev", self.dev_loader)
                if summary is not None:
                    summaries["dev"] = summary
            summary = self._log_c3vdv2_loader_metrics("ood_t4", self.val_loader)
            if summary is not None:
                summaries["ood_t4"] = summary

            primary_summary = summaries.get("dev") or summaries.get("ood_t4")
            if primary_summary is not None and "training_objective" in primary_summary:
                self.writers["val"].add_scalar(
                    "loss",
                    primary_summary["training_objective"],
                    self.step,
                )

            best_metric = self._best_metric_from_c3vdv2_summaries(
                summaries,
                getattr(self.opt, "c3vdv2_best_metric", "auto"),
            )
            if best_metric is not None:
                metric_name, metric_value, mode = best_metric
                self._maybe_save_best_model(metric_name, metric_value, mode=mode)

        self.set_train()
        if not summaries:
            return super().val()

    @staticmethod
    def _best_metric_from_c3vdv2_summaries(summaries, best_metric="auto"):
        """Prefer in-domain dev depth, then trajectory-OOD depth, then fallbacks."""
        best_metric = best_metric or "auto"
        if best_metric != "auto":
            return C3VDv2Stage2Trainer._explicit_best_metric_from_summaries(
                summaries,
                best_metric,
            )

        for prefix in ("dev", "ood_t4"):
            summary = summaries.get(prefix)
            if not summary:
                continue
            for key in ("rmse_mm", "depth_supervised"):
                if key in summary:
                    return f"{prefix}/{key}", summary[key], "min"
            if "ncc_registration" in summary:
                return f"{prefix}/ncc_registration", summary["ncc_registration"], "max"
        return None

    @staticmethod
    def _explicit_best_metric_from_summaries(summaries, best_metric):
        metric_map = {
            "rmse_mm": ("rmse_mm", "rmse_mm", "min", False),
            "median_rmse_mm": ("median_scaled_rmse_mm", "rmse_mm", "min", True),
            "abs_rel": ("abs_rel", "abs_rel", "min", False),
            "median_abs_rel": ("median_scaled_abs_rel", "abs_rel", "min", True),
            "a1": ("a1", "a1", "max", False),
            "median_a1": ("median_scaled_a1", "a1", "max", True),
            "ncc_registration": ("ncc_registration", "ncc_registration", "max", False),
            "training_objective": ("training_objective", "training_objective", "min", False),
            "depth_supervised": ("depth_supervised", "depth_supervised", "min", False),
        }
        if best_metric not in metric_map:
            valid = ", ".join(["auto"] + sorted(metric_map))
            raise ValueError(
                f"Unsupported c3vdv2_best_metric={best_metric!r}; valid: {valid}"
            )

        summary_key, tag_key, mode, median_scaled = metric_map[best_metric]
        for prefix in ("dev", "ood_t4"):
            summary = summaries.get(prefix)
            if summary and summary_key in summary:
                tag_prefix = f"{prefix}_median" if median_scaled else prefix
                return f"{tag_prefix}/{tag_key}", summary[summary_key], mode
        return None

    def _log_c3vdv2_loader_metrics(self, prefix: str, loader) -> dict | None:
        max_batches = max(int(getattr(self.opt, "c3vdv2_val_batches", 0)), 0)
        if max_batches == 0:
            return None

        metric_totals = {}
        metric_totals_median = {}
        objective_total = 0.0
        ncc_total = 0.0
        depth_loss_total = 0.0
        depth_loss_batches = 0
        pose_loss_total = 0.0
        pose_loss_batches = 0
        batches = 0

        for batch_idx, raw_inputs in enumerate(loader):
            if batch_idx >= max_batches:
                break

            inputs = self._adapt_keys(raw_inputs)
            outputs, objective_losses = StageTwoTrainer.process_batch(self, inputs)
            ncc_losses = BaseTrainer.compute_losses_val(self, inputs, outputs)
            batches += 1
            objective_total += float(objective_losses["loss"].detach().cpu())
            ncc_total += float(ncc_losses["loss"].detach().cpu())

            if "gt_depth" in inputs and ("depth", 0, 0) in outputs:
                gt_depth = inputs["gt_depth"]
                pred_depth = outputs[("depth", 0, 0)]
                depth_mask = build_c3vdv2_valid_mask_torch(
                    gt_depth, inputs.get("occlusion")
                )
                merge_depth_metric_sums(
                    metric_totals,
                    depth_metric_sums_torch(gt_depth, pred_depth, depth_mask),
                )
                merge_depth_metric_sums(
                    metric_totals_median,
                    depth_metric_sums_torch(
                        gt_depth,
                        pred_depth,
                        depth_mask,
                        median_scaling=True,
                    ),
                )
                l_depth = scale_invariant_log_loss(
                    pred_depth, gt_depth, depth_mask.float()
                )
                depth_loss_total += float(l_depth.detach().cpu())
                depth_loss_batches += 1

            l_pose = self._supervised_pose_loss(inputs, outputs)
            if l_pose is not None:
                pose_loss_total += float(l_pose.detach().cpu())
                pose_loss_batches += 1

            del raw_inputs, inputs, outputs, objective_losses, ncc_losses

        if batches == 0:
            return None

        writer = self.writers["val"]
        summary = {
            "training_objective": objective_total / batches,
            "ncc_registration": ncc_total / batches,
            "batches": batches,
        }
        writer.add_scalar(
            f"{prefix}/training_objective",
            summary["training_objective"],
            self.step,
        )
        writer.add_scalar(
            f"{prefix}/ncc_registration",
            summary["ncc_registration"],
            self.step,
        )
        writer.add_scalar(f"{prefix}/batches", batches, self.step)

        metrics = finalize_depth_metric_sums(metric_totals)
        if metrics["valid_pixels"] > 0:
            metric_keys = (
                "abs_rel", "sq_rel", "rmse", "rmse_mm",
                "rmse_log", "a1", "a2", "a3",
            )
            for key in metric_keys:
                writer.add_scalar(f"{prefix}/{key}", metrics[key], self.step)
            writer.add_scalar(
                f"{prefix}/valid_pixels",
                metrics["valid_pixels"],
                self.step,
            )
            summary.update(metrics)

        median_metrics = finalize_depth_metric_sums(metric_totals_median)
        if median_metrics["valid_pixels"] > 0:
            for key in (
                "abs_rel", "sq_rel", "rmse", "rmse_mm",
                "rmse_log", "a1", "a2", "a3",
            ):
                writer.add_scalar(
                    f"{prefix}_median/{key}",
                    median_metrics[key],
                    self.step,
                )
            writer.add_scalar(
                f"{prefix}_median/valid_pixels",
                median_metrics["valid_pixels"],
                self.step,
            )
            summary.update(
                {f"median_scaled_{key}": value for key, value in median_metrics.items()}
            )

        if depth_loss_batches > 0:
            summary["depth_supervised"] = depth_loss_total / depth_loss_batches
            writer.add_scalar(
                f"{prefix}/depth_supervised",
                summary["depth_supervised"],
                self.step,
            )
        if pose_loss_batches > 0:
            summary["pose_supervised"] = pose_loss_total / pose_loss_batches
            writer.add_scalar(
                f"{prefix}/pose_supervised",
                summary["pose_supervised"],
                self.step,
            )
        return summary

    def _supervised_pose_loss(self, inputs, outputs):
        pb = self.models.get("pose_balance")
        balance = pb.balance if pb is not None else None
        return supervised_pose_loss_from_batch(
            inputs, outputs, self.opt.frame_ids, balance=balance
        )

    def compute_losses(self, inputs, outputs):
        losses = super().compute_losses(inputs, outputs)

        # Supervised depth
        if "gt_depth" in inputs:
            gt_depth = inputs["gt_depth"]
            pred_depth = outputs[("depth", 0, 0)]
            depth_mask = build_c3vdv2_valid_mask_torch(
                gt_depth, inputs.get("occlusion")
            ).float()
            l_depth = scale_invariant_log_loss(pred_depth, gt_depth, depth_mask)
            losses["loss/depth_supervised"] = l_depth
            weighted_depth = self.w_depth * l_depth
            losses["loss_weighted/depth_supervised"] = weighted_depth
            losses["loss"] = losses["loss"] + weighted_depth

            w_pps = getattr(self.opt, "pps_sup_weight", 0.0)
            if w_pps > 0:
                pps_mask = depth_mask
                if getattr(self.opt, "use_specular_net", False) and (
                    "specular", 0
                ) in outputs:
                    specular_valid = 1.0 - outputs[("specular", 0)].detach()
                    if specular_valid.shape[2:] != pps_mask.shape[2:]:
                        specular_valid = F.interpolate(
                            specular_valid,
                            size=pps_mask.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    pps_mask = pps_mask * specular_valid.clamp(0.0, 1.0)
                baseline_m = getattr(self.opt, "pps_light_baseline_mm", 3.0) / 1000.0
                light_position = pred_depth.new_tensor([baseline_m / 2.0, 0.0, 0.0])
                l_pps = pps_supervised_loss(
                    pred_depth,
                    gt_depth,
                    inputs[("inv_K", 0)],
                    pps_mask,
                    light_position,
                )
                losses["loss/pps_sup"] = l_pps
                weighted_pps = w_pps * l_pps
                losses["loss_weighted/pps_sup"] = weighted_pps
                losses["loss"] = losses["loss"] + weighted_pps

        # Specular BCE weak supervision
        if getattr(self.opt, "use_specular_net", False) and ("specular", 0) in outputs:
            color_0 = inputs[("color", 0, 0)]
            pseudo = compute_specular_pseudo_labels(
                color_0, threshold=getattr(self.opt, "specular_threshold", 0.98)
            )
            pred_s = outputs[("specular", 0)]
            # Resize pseudo-label if training resolution differs from full image
            if pseudo.shape != pred_s.shape:
                pseudo = F.interpolate(
                    pseudo, size=pred_s.shape[2:], mode="nearest"
                )
            l_specular = specular_bce_loss(pred_s, pseudo)
            losses["loss/specular_bce"] = l_specular
            w_spec = getattr(self.opt, "specular_loss_weight", 0.05)
            weighted_specular = w_spec * l_specular
            losses["loss_weighted/specular_bce"] = weighted_specular
            losses["loss"] = losses["loss"] + weighted_specular

        # Supervised pose
        l_pose = self._supervised_pose_loss(inputs, outputs)
        if l_pose is not None:
            losses["loss/pose_supervised"] = l_pose
            weighted_pose = self.w_pose * l_pose
            losses["loss_weighted/pose_supervised"] = weighted_pose
            losses["loss"] = losses["loss"] + weighted_pose

        # Log balance scalar when enabled
        pb = getattr(self, "models", {}).get("pose_balance")
        if pb is not None:
            losses["pose_balance/alpha"] = torch.sigmoid(pb.balance).detach()

        return losses
