"""
trainers/end_to_end.py
----------------------
Trains all networks simultaneously with two separate optimizers:
  - optimizer_0: position_encoder + position  (optical flow / registration)
  - optimizer:   encoder + depth + transform + pose

Within each batch, the position net is updated first (optimizer_0), then the
remaining networks are updated with the position net frozen (optimizer).
"""

from __future__ import absolute_import, division, print_function

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import networks

from layers import get_smooth_loss, get_smooth_bright
from .base import BaseTrainer


class EndToEndTrainer(BaseTrainer):

    def build_models(self):
        # Depth
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained"
        ).to(self.device)
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales
        ).to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        # Position (separate optimizer group)
        self.parameters_to_train_0 = []
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            num_input_images=2,
        ).to(self.device)
        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales
        ).to(self.device)
        self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())
        self.parameters_to_train_0 += list(self.models["position"].parameters())

        # Appearance transform
        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            num_input_images=2,
        ).to(self.device)
        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales
        ).to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())
        self.parameters_to_train += list(self.models["transform"].parameters())

        # Pose
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,
                ).to(self.device)
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2,
                )
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                )
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking"
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1),
            ).to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

    def build_optimizers(self):
        """Two Adam optimizers: one for position net, one for everything else."""
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )
        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.1
        )

    # ── Train / eval modes ────────────────────────────────────────────────────

    def set_train_0(self):
        """Activate position net; freeze everything else."""
        for name in ("position_encoder", "position"):
            for param in self.models[name].parameters():
                param.requires_grad = True
            self.models[name].train()
        for name in ("encoder", "depth", "pose_encoder", "pose",
                     "transform_encoder", "transform"):
            if name in self.models:
                for param in self.models[name].parameters():
                    param.requires_grad = False
                self.models[name].eval()

    def set_train(self):
        """Freeze position net; activate everything else."""
        for name in ("position_encoder", "position"):
            for param in self.models[name].parameters():
                param.requires_grad = False
            self.models[name].eval()
        for name in ("encoder", "depth", "pose_encoder", "pose",
                     "transform_encoder", "transform"):
            if name in self.models:
                for param in self.models[name].parameters():
                    param.requires_grad = True
                self.models[name].train()

    def set_eval(self):
        for name in ("encoder", "depth", "position_encoder", "position",
                     "transform_encoder", "transform", "pose_encoder", "pose"):
            if name in self.models:
                self.models[name].eval()

    # ── Training loop ─────────────────────────────────────────────────────────

    def run_epoch(self):
        print("Training")
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # Step 1: update position net
            self.set_train_0()
            _, losses_0 = self.process_batch_0(inputs)
            self.model_optimizer_0.zero_grad()
            losses_0["loss"].backward()
            self.model_optimizer_0.step()

            # Step 2: update depth / pose / transform (position frozen)
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log_scalars("train", losses_0, prefix="position_step")
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            if (
                self.opt.steps_per_epoch is not None
                and batch_idx + 1 >= self.opt.steps_per_epoch
            ):
                break

        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    # ── Forward passes ────────────────────────────────────────────────────────

    def process_batch_0(self, inputs):
        """Position-net update step: flow + transform (no depth/pose)."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self._predict_position_transform(inputs)
        losses = self._compute_losses_0(inputs, outputs)
        return outputs, losses

    def process_batch(self, inputs):
        """Main update step (position net frozen): depth + pose + transform."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids]
            )
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {k: [f[i] for f in all_features]
                        for i, k in enumerate(self.opt.frame_ids)}
            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def _predict_position_transform(self, inputs):
        """Position-only step: flow + registration + occlusion + transform."""
        outputs = {}
        if self.num_pose_frames != 2:
            return outputs

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i == "s":
                continue

            inputs_all = [pose_feats[f_i], pose_feats[0]]
            inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

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
                    mode="bilinear", align_corners=False,
                )
                outputs[("registration", scale, f_i)] = self.spatial_transform(
                    inputs[("color", f_i, 0)],
                    outputs[("position", "high", scale, f_i)],
                )
                outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                    outputs[("position_reverse", scale, f_i)],
                    [self.opt.height, self.opt.width],
                    mode="bilinear", align_corners=False,
                )
                outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                    outputs[("position_reverse", "high", scale, f_i)]
                )
                outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                    outputs[("position", "high", scale, f_i)],
                    outputs[("position_reverse", "high", scale, f_i)],
                )

            # Transform
            transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
            transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
            outputs_2 = self.models["transform"](transform_inputs)

            for scale in self.opt.scales:
                outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                outputs[("transform", "high", scale, f_i)] = F.interpolate(
                    outputs[("transform", scale, f_i)],
                    [self.opt.height, self.opt.width],
                    mode="bilinear", align_corners=False,
                )
                outputs[("refined", scale, f_i)] = torch.clamp(
                    outputs[("transform", "high", scale, f_i)]
                    * outputs[("occu_mask_backward", 0, f_i)].detach()
                    + inputs[("color", 0, 0)],
                    min=0.0, max=1.0,
                )

        return outputs

    # ── Losses ────────────────────────────────────────────────────────────────

    def _compute_losses_0(self, inputs, outputs):
        """Registration + transform smoothness loss for position-net step."""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += get_smooth_loss(
                    outputs[("position", scale, frame_id)], color
                )
                loss_registration += (
                    self.compute_reprojection_loss(
                        outputs[("registration", scale, frame_id)],
                        outputs[("refined", scale, frame_id)].detach(),
                    ) * occu_mask
                ).sum() / occu_mask.sum()

            loss_photo = loss_registration / 2.0
            loss_smooth = self.opt.position_smoothness * (
                loss_smooth_registration / 2.0
            ) / (2 ** scale)

            loss += loss_photo
            loss += loss_smooth

            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses["loss_weighted/registration_photo/{}".format(scale)] = loss_photo
            losses["loss_weighted/registration_smooth/{}".format(scale)] = loss_smooth

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask = outputs[("occu_mask_backward", 0, frame_id)].detach()

                loss_reprojection += (
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)],
                        outputs[("refined", scale, frame_id)],
                    ) * occu_mask
                ).sum() / occu_mask.sum()

                loss_transform += (
                    torch.abs(
                        outputs[("refined", scale, frame_id)]
                        - outputs[("registration", 0, frame_id)].detach()
                    ).mean(1, True) * occu_mask
                ).sum() / occu_mask.sum()

                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)],
                    inputs[("color", 0, 0)],
                    outputs[("registration", scale, frame_id)].detach(),
                    occu_mask,
                )

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss_photo = loss_reprojection / 2.0
            loss_transform_weighted = self.opt.transform_constraint * (
                loss_transform / 2.0
            )
            loss_cvt_weighted = self.opt.transform_smoothness * (loss_cvt / 2.0)
            loss_disp_smooth = self.opt.disparity_smoothness * smooth_loss / (
                2 ** scale
            )

            loss += loss_photo
            loss += loss_transform_weighted
            loss += loss_cvt_weighted
            loss += loss_disp_smooth

            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses["loss_weighted/reprojection/{}".format(scale)] = loss_photo
            losses["loss_weighted/transform_constraint/{}".format(scale)] = (
                loss_transform_weighted
            )
            losses["loss_weighted/transform_smoothness/{}".format(scale)] = (
                loss_cvt_weighted
            )
            losses["loss_weighted/disparity_smoothness/{}".format(scale)] = (
                loss_disp_smooth
            )

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
