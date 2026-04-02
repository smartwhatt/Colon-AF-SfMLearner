# Thin shim — kept for backward compatibility with train_stage_two.py
# New code should use: from trainers import StageTwoTrainer
from trainers.stage_two import StageTwoTrainer as Trainer

__all__ = ["Trainer"]
