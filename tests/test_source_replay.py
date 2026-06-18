import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colon_datasets.hf_triplet_dataset as hf_triplet_dataset
import trainers.stage_two as stage_two_module
from options import TrainingConfig
from trainers.c3vdv2_replay import raw_log_depth_smooth_l1_loss
from trainers.stage_two import StageTwoTrainer


def test_source_replay_cli_options_parse():
    cfg = TrainingConfig.from_args(
        [
            "--source_replay_weight",
            "1.0",
            "--source_replay_hf_repo_id",
            "SmartWhatt/c3vdv2-SfM",
            "--source_replay_train_split",
            "train",
            "--source_replay_val_split",
            "val",
            "--source_replay_batch_size",
            "8",
            "--source_replay_depth_weight",
            "2.0",
            "--source_replay_pose_weight",
            "0.25",
            "--source_replay_depth_loss",
            "smooth_l1_log",
            "--source_replay_val_batches",
            "3",
        ]
    )

    assert cfg.source_replay_weight == 1.0
    assert cfg.source_replay_hf_repo_id == "SmartWhatt/c3vdv2-SfM"
    assert cfg.source_replay_batch_size == 8
    assert cfg.source_replay_depth_weight == 2.0
    assert cfg.source_replay_pose_weight == 0.25
    assert cfg.source_replay_depth_loss == "smooth_l1_log"
    assert cfg.source_replay_val_batches == 3


def test_source_replay_raw_log_depth_loss_penalizes_global_scale():
    gt_depth = torch.tensor([[[[0.010, 0.020], [0.030, 0.040]]]])
    mask = torch.ones_like(gt_depth)

    perfect = raw_log_depth_smooth_l1_loss(gt_depth, gt_depth, mask)
    scaled = raw_log_depth_smooth_l1_loss(gt_depth * 2.0, gt_depth, mask)

    assert perfect.item() == pytest.approx(0.0)
    assert scaled.item() > 0.0


def test_source_replay_loss_combines_weighted_depth_and_pose(monkeypatch):
    trainer = object.__new__(StageTwoTrainer)
    trainer.opt = SimpleNamespace(
        source_replay_weight=3.0,
        source_replay_depth_weight=2.0,
        source_replay_pose_weight=0.5,
        source_replay_depth_loss="smooth_l1_log",
        frame_ids=[0, -1, 1],
    )
    trainer.models = {}
    monkeypatch.setattr(
        stage_two_module,
        "source_replay_depth_loss",
        lambda pred, gt, occ, loss_type: torch.tensor(4.0),
    )
    trainer._source_replay_pose_loss = lambda inputs, outputs: torch.tensor(6.0)
    inputs = {"gt_depth": torch.ones(1, 1, 2, 2)}
    outputs = {("depth", 0, 0): torch.ones(1, 1, 2, 2)}

    losses = StageTwoTrainer._compute_source_replay_losses(trainer, inputs, outputs)

    assert losses["source_replay/depth"].item() == pytest.approx(4.0)
    assert losses["source_replay_weighted/depth"].item() == pytest.approx(8.0)
    assert losses["source_replay/pose"].item() == pytest.approx(6.0)
    assert losses["source_replay_weighted/pose"].item() == pytest.approx(3.0)
    assert losses["source_replay/loss"].item() == pytest.approx(11.0)
    assert losses["loss_weighted/source_replay"].item() == pytest.approx(33.0)


def test_source_replay_dataloader_uses_c3vdv2_hf_format(monkeypatch):
    calls = []

    class FakeDataset:
        def __len__(self):
            return 12

    class FakeLoader:
        def __init__(self, dataset, batch_size, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.kwargs = kwargs

        def __iter__(self):
            return iter(())

    def fake_build_hf_triplet_dataset(**kwargs):
        calls.append(kwargs)
        return FakeDataset()

    monkeypatch.setattr(
        hf_triplet_dataset,
        "build_hf_triplet_dataset",
        fake_build_hf_triplet_dataset,
    )
    monkeypatch.setattr(stage_two_module, "DataLoader", FakeLoader)

    trainer = object.__new__(StageTwoTrainer)
    trainer.opt = SimpleNamespace(
        source_replay_hf_repo_id="SmartWhatt/c3vdv2-SfM",
        source_replay_train_split="train",
        source_replay_val_split="val",
        source_replay_batch_size=8,
        batch_size=12,
        hf_streaming=False,
        hf_cache_dir="/tmp/hf",
        height=256,
        width=320,
        scales=[0, 1, 2, 3],
        hf_shuffle_buffer=100,
        num_workers=2,
    )

    StageTwoTrainer._build_source_replay_dataloaders(trainer)

    assert [call["split"] for call in calls] == ["train", "val"]
    assert all(call["repo_id"] == "SmartWhatt/c3vdv2-SfM" for call in calls)
    assert all(call["output_format"] == "c3vdv2" for call in calls)
    assert calls[0]["is_train"] is True
    assert calls[1]["is_train"] is False
    assert trainer.source_replay_train_loader.batch_size == 8
    assert trainer.source_replay_val_loader.batch_size == 8


def test_source_replay_val_logs_median_scaled_metrics(monkeypatch):
    class FakeWriter:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = (float(value), step)

    trainer = object.__new__(StageTwoTrainer)
    trainer.opt = SimpleNamespace(source_replay_val_batches=1)
    trainer.device = torch.device("cpu")
    trainer.step = 7
    trainer.writers = {"val": FakeWriter()}
    trainer.source_replay_val_loader = [
        {
            "gt_depth": torch.tensor(
                [[[[0.010, 0.020], [0.030, 0.040]]]], dtype=torch.float32
            )
        }
    ]
    trainer._forward_stage2 = lambda inputs: {
        ("depth", 0, 0): inputs["gt_depth"] * 2.0
    }
    trainer._compute_source_replay_losses = lambda inputs, outputs: {
        "source_replay/loss": torch.tensor(1.5)
    }
    monkeypatch.setattr(stage_two_module, "adapt_c3vdv2_batch", lambda batch, opt, device: batch)

    summary = StageTwoTrainer._log_source_replay_val_metrics(trainer)

    scalars = trainer.writers["val"].scalars
    assert "source_replay/rmse_mm" in scalars
    assert "source_replay_median/rmse_mm" in scalars
    assert "source_replay_median/a1" in scalars
    assert scalars["source_replay/rmse_mm"][0] > 0.0
    assert scalars["source_replay_median/rmse_mm"][0] == pytest.approx(0.0)
    assert summary["median_scaled_rmse_mm"] == pytest.approx(0.0)
