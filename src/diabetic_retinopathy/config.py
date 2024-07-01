"""Central configuration objects for local training and inference.

The original notebook used absolute Google Drive paths throughout the code. This
module keeps those locations configurable so the same workflow can run from a
local checkout, Colab, or another machine by changing command-line arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationPaths:
    """Directory layout for the IDRiD-style lesion segmentation dataset."""

    train_images: Path
    train_microaneurysms: Path
    train_hemorrhages: Path
    train_hard_exudates: Path
    test_images: Path
    test_microaneurysms: Path
    test_hemorrhages: Path
    test_hard_exudates: Path


@dataclass(frozen=True)
class GradingPaths:
    """Directory layout for disease grading images and labels."""

    train_images: Path
    test_images: Path
    train_labels_csv: Path
    test_labels_csv: Path | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Training values shared by the scripts.

    These defaults match the notebook closely while still making the costly
    choices explicit at the command line.
    """

    batch_size: int = 2
    epochs: int = 30
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    patch_size: int = 512
    patch_step: int = 512
    random_seed: int = 42
