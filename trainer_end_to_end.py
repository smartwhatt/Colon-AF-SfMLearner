# Thin shim — kept for backward compatibility with train_end_to_end.py
# New code should use: from trainers import EndToEndTrainer
from trainers.end_to_end import EndToEndTrainer as Trainer

__all__ = ["Trainer"]
