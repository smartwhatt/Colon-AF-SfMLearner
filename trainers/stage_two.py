"""
trainers/stage_two.py
---------------------
Trains encoder + depth + transform_encoder + transform + pose_encoder + pose.
Loads and freezes position_encoder + position from a Stage 1 checkpoint.

The frozen position net provides reliable registrations which serve as:
  1. Input to the brightness-calibration (transform) network
  2. Occlusion masks
  3. The `refined` supervision target for depth/pose training
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import networks

import torch.nn.functional as F


class PoseBalanceScalar(nn.Module):
    """Single learnable logit that balances rotation vs. translation in pose loss.

    α = σ(balance) interpolates between rot (α → 1) and trans (α → 0).
    Initialised at 0 → α = 0.5. Sigmoid-bounded, one parameter, no instability.
    """

    def __init__(self):
        super().__init__()
        self.balance = nn.Parameter(torch.zeros(1))
from layers import get_smooth_loss, get_smooth_bright
from .base import BaseTrainer


class StageTwoTrainer(BaseTrainer):
    _DEFAULT_TRAIN_MODELS = (
        "encoder",
        "depth",
        "transform_encoder",
        "transform",
        "pose_encoder",
        "pose",
        "specular",
    )

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

        # Position (frozen after load_model — not added to parameters_to_train)
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            num_input_images=2,
        ).to(self.device)
        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales
        ).to(self.device)

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

        if getattr(self.opt, "use_specular_net", False):
            self.models["specular"] = networks.SpecularDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales
            ).to(self.device)
            self.parameters_to_train += list(self.models["specular"].parameters())

        if getattr(self.opt, "use_uncertainty_pose", False):
            self.models["pose_balance"] = PoseBalanceScalar().to(self.device)
            self.parameters_to_train += list(self.models["pose_balance"].parameters())

        self._apply_trainable_module_allowlist()

    def _trainable_module_allowlist(self):
        modules = getattr(self.opt, "trainable_modules", []) or []
        unique_modules = []
        for name in modules:
            if name not in unique_modules:
                unique_modules.append(name)
        return unique_modules

    def _uses_trainable_module_allowlist(self):
        return len(self._trainable_module_allowlist()) > 0

    def _validate_trainable_module_allowlist(self, trainable_modules):
        missing = [name for name in trainable_modules if name not in self.models]
        if missing:
            available = ", ".join(sorted(self.models.keys()))
            requested = ", ".join(missing)
            raise ValueError(
                "Unknown trainable_modules: {}. Available modules: {}".format(
                    requested, available
                )
            )

    def _module_should_train(self, name):
        trainable_modules = self._trainable_module_allowlist()
        if not trainable_modules:
            return True
        return name in trainable_modules

    def _apply_trainable_module_allowlist(self):
        trainable_modules = self._trainable_module_allowlist()
        if not trainable_modules:
            return

        self._validate_trainable_module_allowlist(trainable_modules)
        trainable_set = set(trainable_modules)

        for name, model in self.models.items():
            should_train = name in trainable_set
            for param in model.parameters():
                param.requires_grad = should_train
            if not should_train:
                model.eval()

        self.parameters_to_train = []
        for name in trainable_modules:
            self.parameters_to_train += list(self.models[name].parameters())

    def load_model(self):
        """Load frozen position weights and optional trainable Stage 2 weights."""
        import os
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if n not in self.models:
                continue
            print("Loading {} weights...".format(n))
            self._load_model_weights(self.opt.load_weights_folder, n)
            self.models[n].eval()
            for param in self.models[n].parameters():
                param.requires_grad = False

        if getattr(self.opt, "load_trainable_weights_folder", None):
            trainable_folder = os.path.expanduser(self.opt.load_trainable_weights_folder)
            assert os.path.isdir(trainable_folder), \
                "Cannot find folder {}".format(trainable_folder)
            print("loading trainable model weights from folder {}".format(trainable_folder))
            for n in self.opt.trainable_models_to_load:
                if n not in self.models or n in self.opt.models_to_load:
                    continue
                path = os.path.join(trainable_folder, "{}.pth".format(n))
                if not os.path.isfile(path):
                    print("Skipping {} weights (not found)".format(n))
                    continue
                print("Loading {} weights without freezing...".format(n))
                self._load_model_weights(trainable_folder, n)
                for param in self.models[n].parameters():
                    param.requires_grad = self._module_should_train(n)

        self._apply_trainable_module_allowlist()

        print("Adam is randomly initialized")

    def _load_model_weights(self, weights_folder, model_name):
        import os
        path = os.path.join(weights_folder, "{}.pth".format(model_name))
        model_dict = self.models[model_name].state_dict()
        pretrained_dict = torch.load(path, map_location=self.device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.models[model_name].load_state_dict(model_dict)

    def set_train(self):
        if self._uses_trainable_module_allowlist():
            trainable_set = set(self._trainable_module_allowlist())
            for name, model in self.models.items():
                if name in trainable_set:
                    model.train()
                else:
                    model.eval()
            return

        for name in self._DEFAULT_TRAIN_MODELS:
            if name in self.models:
                self.models[name].train()

    def set_eval(self):
        for name in self._DEFAULT_TRAIN_MODELS:
            if name in self.models:
                self.models[name].eval()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def process_batch(self, inputs):
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

        if "specular" in self.models:
            outputs.update(self.models["specular"](features))

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    # ── Losses ────────────────────────────────────────────────────────────────

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

            # Specular gating mask: high-confidence specular pixels are down-weighted.
            # Detach so S is only trained through its own BCE objective.
            if ("specular", scale) in outputs:
                specular_conf = outputs[("specular", scale)].detach()  # (B,1,H_s,W_s)
            else:
                specular_conf = None

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask = outputs[("occu_mask_backward", 0, frame_id)].detach()

                if specular_conf is not None:
                    sc = specular_conf
                    if sc.shape[2:] != occu_mask.shape[2:]:
                        sc = F.interpolate(sc, size=occu_mask.shape[2:],
                                           mode="bilinear", align_corners=False)
                    valid = occu_mask * (1.0 - sc)
                else:
                    valid = occu_mask

                loss_reprojection += (
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)],
                        outputs[("refined", scale, frame_id)],
                    ) * valid
                ).sum() / valid.sum().clamp(min=1)

                loss_transform += (
                    torch.abs(
                        outputs[("refined", scale, frame_id)]
                        - outputs[("registration", 0, frame_id)].detach()
                    ).mean(1, True) * valid
                ).sum() / valid.sum().clamp(min=1)

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
