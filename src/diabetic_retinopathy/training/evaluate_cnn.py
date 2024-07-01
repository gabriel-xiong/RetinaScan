"""Evaluate a saved retinopathy grading CNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diabetic_retinopathy.data.grading import (
    RetinopathyGradingDataset,
    grading_transform,
    load_label_map,
)
from diabetic_retinopathy.evaluation import collect_predictions, compute_classification_metrics
from diabetic_retinopathy.models.rog_cnn import RogCNN


def evaluate_cnn(args: argparse.Namespace) -> None:
    """Load a checkpoint, run predictions, and print summary metrics."""

    label_map = load_label_map(args.labels_csv)
    dataset = RetinopathyGradingDataset(
        image_dir=args.image_dir,
        label_map=label_map,
        transform=grading_transform(args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = RogCNN(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    predictions, labels = collect_predictions(model, loader, device)
    metrics = compute_classification_metrics(predictions, labels)
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"Accuracy: {metrics.accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Evaluation image directory.")
    parser.add_argument("--labels-csv", type=Path, required=True, help="CSV with Image name and grade columns.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Saved CNN checkpoint.")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_cnn(parse_args())
