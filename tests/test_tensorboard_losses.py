import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trainers.base as base_module
import trainers.c3vdv2 as c3vdv2_module
import trainers.stage_one as stage_one_module
import trainers.stage_two as stage_two_module
from trainers.base import BaseTrainer
from trainers.c3vdv2 import C3VDv2Stage2Trainer
from trainers.stage_one import StageOneTrainer
from trainers.stage_two import StageTwoTrainer


class _ConcreteBaseTrainer(BaseTrainer):
    def build_models(self) -> None:
        raise NotImplementedError

    def set_train(self) -> None:
        raise NotImplementedError

    def set_eval(self) -> None:
        raise NotImplementedError

    def process_batch(self, inputs: dict):
        raise NotImplementedError

    def compute_losses(self, inputs: dict, outputs: dict) -> dict:
        raise NotImplementedError


def _trainer_stub(cls, scales=(0,), frame_ids=(0, -1, 1)):
    trainer = object.__new__(cls)
    trainer.opt = SimpleNamespace(
        scales=list(scales),
        frame_ids=list(frame_ids),
        position_smoothness=0.5,
        transform_constraint=0.01,
        transform_smoothness=0.02,
        disparity_smoothness=0.1,
    )
    trainer.num_scales = len(scales)
    return trainer


def _val_inputs_outputs(scales=(0, 1), frame_ids=(0, -1, 1)):
    inputs = {("color", 0, 0): torch.zeros(2, 3, 4, 5)}
    outputs = {}
    for scale in scales:
        for frame_id in frame_ids[1:]:
            outputs[("registration", scale, frame_id)] = torch.zeros(2, 3, 4, 5)
    return inputs, outputs


def _negative_ncc(pred, target):
    return torch.full_like(pred, -0.25)


@pytest.mark.parametrize(
    ("module", "cls"),
    [(base_module, _ConcreteBaseTrainer), (stage_one_module, StageOneTrainer)],
)
def test_validation_loss_tags_are_positive(monkeypatch, module, cls):
    monkeypatch.setattr(module, "ncc_loss", _negative_ncc)
    trainer = _trainer_stub(cls, scales=(0, 1))
    inputs, outputs = _val_inputs_outputs(scales=(0, 1))

    losses = cls.compute_losses_val(trainer, inputs, outputs)

    assert losses["loss"].item() == pytest.approx(0.25)
    assert losses["loss/0"].item() == pytest.approx(0.25)
    assert losses["loss/1"].item() == pytest.approx(0.25)


def _stage_two_tensors():
    inputs = {("color", 0, 0): torch.zeros(1, 3, 2, 2)}
    outputs = {("disp", 0): torch.ones(1, 1, 2, 2)}

    for frame_id in (-1, 1):
        inputs[("color", frame_id, 0)] = torch.zeros(1, 3, 2, 2)
        outputs[("color", frame_id, 0)] = torch.zeros(1, 3, 2, 2)
        outputs[("refined", 0, frame_id)] = torch.full((1, 3, 2, 2), 3.0)
        outputs[("registration", 0, frame_id)] = torch.ones(1, 3, 2, 2)
        outputs[("occu_mask_backward", 0, frame_id)] = torch.ones(1, 1, 2, 2)
        outputs[("transform", "high", 0, frame_id)] = torch.zeros(1, 3, 2, 2)

    return inputs, outputs


def test_stage_two_loss_dict_includes_weighted_components(monkeypatch):
    monkeypatch.setattr(
        stage_two_module,
        "get_smooth_loss",
        lambda disp, color: torch.tensor(3.0),
    )
    monkeypatch.setattr(
        stage_two_module,
        "get_smooth_bright",
        lambda transform, target, pred, occu_mask: torch.tensor(5.0),
    )
    trainer = _trainer_stub(StageTwoTrainer)
    trainer.compute_reprojection_loss = lambda pred, target: torch.full(
        (pred.shape[0], 1, pred.shape[2], pred.shape[3]), 2.0
    )
    inputs, outputs = _stage_two_tensors()

    losses = StageTwoTrainer.compute_losses(trainer, inputs, outputs)

    assert losses["loss_weighted/reprojection/0"].item() == pytest.approx(2.0)
    assert losses["loss_weighted/transform_constraint/0"].item() == pytest.approx(0.02)
    assert losses["loss_weighted/transform_smoothness/0"].item() == pytest.approx(0.1)
    assert losses["loss_weighted/disparity_smoothness/0"].item() == pytest.approx(0.3)
    assert losses["loss/0"].item() == pytest.approx(2.42)
    assert losses["loss"].item() == pytest.approx(2.42)


def test_c3vdv2_supervised_losses_include_weighted_terms(monkeypatch):
    monkeypatch.setattr(
        stage_two_module,
        "get_smooth_loss",
        lambda disp, color: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        stage_two_module,
        "get_smooth_bright",
        lambda transform, target, pred, occu_mask: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        c3vdv2_module,
        "build_c3vdv2_valid_mask_torch",
        lambda gt_depth, occlusion: torch.ones_like(gt_depth, dtype=torch.bool),
    )
    monkeypatch.setattr(
        c3vdv2_module,
        "scale_invariant_log_loss",
        lambda pred_depth, gt_depth, mask: torch.tensor(4.0),
    )

    trainer = _trainer_stub(C3VDv2Stage2Trainer)
    trainer.w_depth = 0.5
    trainer.w_pose = 0.25
    trainer._supervised_pose_loss = lambda inputs, outputs: torch.tensor(6.0)
    trainer.compute_reprojection_loss = lambda pred, target: torch.zeros(
        pred.shape[0], 1, pred.shape[2], pred.shape[3]
    )
    inputs, outputs = _stage_two_tensors()
    inputs["gt_depth"] = torch.ones(1, 1, 2, 2)
    outputs[("depth", 0, 0)] = torch.ones(1, 1, 2, 2)

    losses = C3VDv2Stage2Trainer.compute_losses(trainer, inputs, outputs)

    assert losses["loss/depth_supervised"].item() == pytest.approx(4.0)
    assert losses["loss_weighted/depth_supervised"].item() == pytest.approx(2.0)
    assert losses["loss/pose_supervised"].item() == pytest.approx(6.0)
    assert losses["loss_weighted/pose_supervised"].item() == pytest.approx(1.5)
