"""Train the CNN retinopathy disease-grading model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from diabetic_retinopathy.data.grading import (
    RetinopathyGradingDataset,
    grading_transform,
    load_label_map,
)
from diabetic_retinopathy.models.rog_cnn import RogCNN


def train_cnn(args: argparse.Namespace) -> None:
    """Train and save the retinopathy grading CNN."""

    label_map = load_label_map(args.labels_csv)
    dataset = RetinopathyGradingDataset(
        image_dir=args.image_dir,
        label_map=label_map,
        transform=grading_transform(args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = RogCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: mean loss = {running_loss / max(len(loader), 1):.4f}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved CNN checkpoint to {args.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Training image directory.")
    parser.add_argument("--labels-csv", type=Path, required=True, help="CSV with Image name and grade columns.")
    parser.add_argument("--output-path", type=Path, required=True, help="Checkpoint path to write.")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    train_cnn(parse_args())
