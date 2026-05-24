import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import disp_to_depth
from options import TrainingConfig
from trainers.stage_two import StageTwoTrainer


class _DummyTeacherEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class _DummyTeacherDepth(torch.nn.Module):
    def __init__(self, disp_value=0.5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.disp_value = disp_value

    def forward(self, features):
        batch, _channels, height, width = features.shape
        disp = torch.full(
            (batch, 1, height, width),
            self.disp_value,
            dtype=features.dtype,
            device=features.device,
        )
        return {("disp", 0): disp}


def test_teacher_consistency_cli_options_parse():
    cfg = TrainingConfig.from_args(
        [
            "--teacher_weights_folder",
            "/tmp/teacher",
            "--teacher_consistency_weight",
            "0.5",
            "--teacher_consistency_mode",
            "relative_log_median",
        ]
    )

    assert cfg.teacher_weights_folder == "/tmp/teacher"
    assert cfg.teacher_consistency_weight == 0.5
    assert cfg.teacher_consistency_mode == "relative_log_median"


def test_relative_log_median_removes_global_depth_scale():
    depth = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])

    rel_a = StageTwoTrainer.relative_log_median(depth)
    rel_b = StageTwoTrainer.relative_log_median(depth * 7.0)

    assert torch.allclose(rel_a, rel_b, atol=1e-6)


def test_teacher_consistency_loss_accepts_scaled_student_depth():
    trainer = object.__new__(StageTwoTrainer)
    trainer.device = torch.device("cpu")
    trainer.opt = SimpleNamespace(
        teacher_weights_folder="/tmp/teacher",
        teacher_consistency_weight=0.5,
        teacher_consistency_mode="relative_log_median",
        height=4,
        width=4,
        min_depth=0.001,
        max_depth=0.1,
    )
    trainer.models = {
        "teacher_encoder": _DummyTeacherEncoder(),
        "teacher_depth": _DummyTeacherDepth(disp_value=0.5),
    }
    StageTwoTrainer._freeze_teacher_models(trainer)

    color = torch.zeros(1, 3, 4, 4)
    teacher_disp = torch.full((1, 1, 4, 4), 0.5)
    _, teacher_depth = disp_to_depth(
        teacher_disp, trainer.opt.min_depth, trainer.opt.max_depth
    )
    outputs = {("depth", 0, 0): teacher_depth * 3.0}
    inputs = {("color_aug", 0, 0): color}

    loss = StageTwoTrainer._compute_teacher_consistency_loss(trainer, inputs, outputs)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert all(
        not param.requires_grad
        for model in trainer.models.values()
        for param in model.parameters()
    )
