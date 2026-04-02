"""
trainers/__init__.py
--------------------
Trainer registry. Maps --mode argument to trainer class.

Usage:
    from trainers import TRAINERS
    trainer = TRAINERS["stage2"](opts)
    trainer.train()
"""

from .stage_one import StageOneTrainer
from .stage_two import StageTwoTrainer
from .end_to_end import EndToEndTrainer
from .c3vdv2 import C3VDv2Stage1Trainer, C3VDv2Stage2Trainer

TRAINERS = {
    "stage1":          StageOneTrainer,
    "stage2":          StageTwoTrainer,
    "end_to_end":      EndToEndTrainer,
    "c3vdv2_stage1":   C3VDv2Stage1Trainer,
    "c3vdv2_stage2":   C3VDv2Stage2Trainer,
}

__all__ = [
    "StageOneTrainer",
    "StageTwoTrainer",
    "EndToEndTrainer",
    "C3VDv2Stage1Trainer",
    "C3VDv2Stage2Trainer",
    "TRAINERS",
]
