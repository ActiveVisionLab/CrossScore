"""Backwards compatibility: re-export from crossscore package."""
from crossscore.task.core import CrossScoreLightningModule, CrossScoreNet

__all__ = ["CrossScoreLightningModule", "CrossScoreNet"]
