import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.stage_two import StageTwoTrainer
from options import TrainingConfig


MODEL_NAMES = (
    "encoder",
    "depth",
    "position_encoder",
    "position",
    "transform_encoder",
    "transform",
    "pose_encoder",
    "pose",
    "specular",
)

POSE_TRANSFORM_MODULES = [
    "pose_encoder",
    "pose",
    "transform_encoder",
    "transform",
]


def _stage_two_stub(trainable_modules):
    trainer = object.__new__(StageTwoTrainer)
    trainer.device = torch.device("cpu")
    trainer.models = {name: torch.nn.Linear(2, 2) for name in MODEL_NAMES}
    trainer.parameters_to_train = [
        param
        for name in StageTwoTrainer._DEFAULT_TRAIN_MODELS
        if name in trainer.models
        for param in trainer.models[name].parameters()
    ]
    trainer.opt = SimpleNamespace(trainable_modules=list(trainable_modules))
    return trainer


def _param_ids(params):
    return [id(param) for param in params]


def _all_require_grad(model):
    return all(param.requires_grad for param in model.parameters())


def _none_require_grad(model):
    return all(not param.requires_grad for param in model.parameters())


def test_empty_trainable_modules_preserves_default_trainable_parameters():
    trainer = _stage_two_stub(trainable_modules=[])
    original_param_ids = _param_ids(trainer.parameters_to_train)

    StageTwoTrainer._apply_trainable_module_allowlist(trainer)

    assert _param_ids(trainer.parameters_to_train) == original_param_ids
    assert all(_all_require_grad(model) for model in trainer.models.values())


def test_trainable_modules_cli_list_parses():
    cfg = TrainingConfig.from_args(
        ["--trainable_modules", "pose_encoder", "pose", "transform_encoder", "transform"]
    )

    assert cfg.trainable_modules == POSE_TRANSFORM_MODULES


def test_pose_transform_allowlist_freezes_depth_and_specular_modules():
    trainer = _stage_two_stub(trainable_modules=POSE_TRANSFORM_MODULES)

    StageTwoTrainer._apply_trainable_module_allowlist(trainer)

    for name in POSE_TRANSFORM_MODULES:
        assert _all_require_grad(trainer.models[name])

    for name in ("encoder", "depth", "specular", "position_encoder", "position"):
        assert _none_require_grad(trainer.models[name])
        assert not trainer.models[name].training

    expected_param_ids = [
        id(param)
        for name in POSE_TRANSFORM_MODULES
        for param in trainer.models[name].parameters()
    ]
    assert _param_ids(trainer.parameters_to_train) == expected_param_ids

    StageTwoTrainer.set_train(trainer)

    for name in POSE_TRANSFORM_MODULES:
        assert trainer.models[name].training
    for name in ("encoder", "depth", "specular", "position_encoder", "position"):
        assert not trainer.models[name].training


def test_warm_start_loading_does_not_reenable_frozen_modules(tmp_path):
    trainer = _stage_two_stub(trainable_modules=POSE_TRANSFORM_MODULES)
    trainer.opt.load_weights_folder = str(tmp_path)
    trainer.opt.load_trainable_weights_folder = str(tmp_path)
    trainer.opt.models_to_load = ["position_encoder", "position"]
    trainer.opt.trainable_models_to_load = [
        "encoder",
        "depth",
        "transform_encoder",
        "transform",
        "pose_encoder",
        "pose",
        "specular",
    ]

    for name, model in trainer.models.items():
        torch.save(model.state_dict(), tmp_path / "{}.pth".format(name))

    StageTwoTrainer.load_model(trainer)

    for name in POSE_TRANSFORM_MODULES:
        assert _all_require_grad(trainer.models[name])

    for name in ("encoder", "depth", "specular", "position_encoder", "position"):
        assert _none_require_grad(trainer.models[name])
        assert not trainer.models[name].training


def test_invalid_trainable_module_name_raises_clear_error():
    trainer = _stage_two_stub(trainable_modules=["encoder", "bogus"])

    with pytest.raises(ValueError, match="Unknown trainable_modules: bogus"):
        StageTwoTrainer._apply_trainable_module_allowlist(trainer)


def test_frozen_depth_anything_pair_stays_frozen_through_allowlist_and_set_train():
    """DA-v2 frozen under the depth_anything_frozen preset (pose+transform allowlist).

    The allowlist already freezes encoder/depth (they're not in the allowlist);
    ``_freeze_depth_anything`` is the belt-and-suspenders re-assertion that
    must keep them frozen even after ``set_train`` toggles.
    """
    from networks.depth_anything_adapter import build_depth_anything_pair

    trainer = object.__new__(StageTwoTrainer)
    trainer.device = torch.device("cpu")
    encoder, depth = build_depth_anything_pair(
        model=torch.nn.Conv2d(3, 1, 1),
        min_depth=0.001,
        max_depth=0.1,
        frozen=True,
    )
    trainer.models = {name: torch.nn.Linear(2, 2) for name in MODEL_NAMES}
    trainer.models["encoder"] = encoder
    trainer.models["depth"] = depth
    trainer.parameters_to_train = []
    trainer.opt = SimpleNamespace(
        trainable_modules=list(POSE_TRANSFORM_MODULES),
        depth_backend="depth_anything",
        depth_anything_frozen=True,
    )

    StageTwoTrainer._apply_trainable_module_allowlist(trainer)
    StageTwoTrainer._freeze_depth_anything(trainer)
    StageTwoTrainer.set_train(trainer)

    for name in ("encoder", "depth"):
        assert _none_require_grad(trainer.models[name])
        assert not trainer.models[name].dav2.training
    for name in POSE_TRANSFORM_MODULES:
        assert trainer.models[name].training
