"""Evaluation helpers for classification experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Container for the metrics reported by the original notebook."""

    precision: float
    recall: float
    accuracy: float


def compute_classification_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> ClassificationMetrics:
    """Compute macro precision, macro recall, and accuracy."""

    predictions_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return ClassificationMetrics(
        precision=precision_score(labels_np, predictions_np, average="macro", zero_division=0),
        recall=recall_score(labels_np, predictions_np, average="macro", zero_division=0),
        accuracy=accuracy_score(labels_np, predictions_np),
    )


def collect_predictions(model: torch.nn.Module, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a model over a DataLoader and collect predicted and true labels."""

    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_predictions), torch.cat(all_labels)
