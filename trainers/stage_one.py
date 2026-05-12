"""
trainers/stage_one.py
---------------------
Trains only the position/registration network (position_encoder + position).
No depth, pose, or appearance transform — purely self-supervised photometric
alignment.

This is the prerequisite for Stage 2: the learned flow is frozen and used as
the geometric anchor for the occlusion mask and brightness-calibrated target.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import networks

from layers import get_smooth_registration, ncc_loss
from .base import BaseTrainer


class StageOneTrainer(BaseTrainer):

    def build_models(self):
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=2,
        ).to(self.device)
        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales
        ).to(self.device)

        self.parameters_to_train += list(self.models["position_encoder"].parameters())
        self.parameters_to_train += list(self.models["position"].parameters())

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self._predict_positions(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def process_batch_val(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self._predict_positions(inputs)
        losses = self.compute_losses_val(inputs, outputs)
        return outputs, losses

    def _predict_positions(self, inputs):
        """Position-only forward: flow + warp. No pose, transform, or depth."""
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            position_inputs = self.models["position_encoder"](
                torch.cat([pose_feats[f_i], pose_feats[0]], 1)
            )
            outputs_0 = self.models["position"](position_inputs)

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
        return outputs

    # ── Losses ────────────────────────────────────────────────────────────────

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_registration = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                loss_registration += get_smooth_registration(
                    outputs[("position", scale, frame_id)]
                )
                registration_losses.append(
                    self.compute_reprojection_loss(
                        outputs[("registration", scale, frame_id)], target
                    )
                )

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, _ = torch.min(registration_losses, dim=1)

            loss_photo = registration_losses.mean()
            loss_smooth = self.opt.position_smoothness * (
                loss_registration / 2.0
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

    def compute_losses_val(self, inputs, outputs):
        """NCC-based val loss (registration quality)."""
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

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        self.log_scalars(mode, losses)

        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step,
                    )
