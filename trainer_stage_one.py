# Thin shim — kept for backward compatibility with train_stage_one.py
# New code should use: from trainers import StageOneTrainer
from trainers.stage_one import StageOneTrainer as Trainer

__all__ = ["Trainer"]
