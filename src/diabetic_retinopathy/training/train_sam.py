"""Fine-tune SAM's mask decoder for one diabetic retinopathy lesion type."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import monai
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from diabetic_retinopathy.data.segmentation import build_segmentation_dataset
from diabetic_retinopathy.models.sam_dataset import SAMDataset


def freeze_sam_encoders(model: SamModel) -> None:
    """Train only the mask decoder, matching the original experiment."""

    for name, parameter in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            parameter.requires_grad_(False)


def train_sam(args: argparse.Namespace) -> None:
    """Run the SAM fine-tuning loop and save a checkpoint."""

    dataset = build_segmentation_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        patch_size=args.patch_size,
        patch_step=args.patch_step,
        filter_empty=True,
    )
    processor = SamProcessor.from_pretrained(args.base_model)
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = SamModel.from_pretrained(args.base_model).to(device)
    freeze_sam_encoders(model)

    optimizer = Adam(model.mask_decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    segmentation_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    model.train()
    for epoch in range(args.epochs):
        epoch_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False,
            )

            predicted_masks = outputs.pred_masks.squeeze(1)
            predicted_masks = nn.functional.interpolate(
                predicted_masks,
                size=(args.patch_size, args.patch_size),
                mode="bilinear",
                align_corners=False,
            )
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = segmentation_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"Epoch {epoch + 1}: mean loss = {mean(epoch_losses):.4f}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved SAM checkpoint to {args.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Training fundus image directory.")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Training mask directory for one lesion.")
    parser.add_argument("--output-path", type=Path, required=True, help="Checkpoint path to write.")
    parser.add_argument("--base-model", default="facebook/sam-vit-base", help="Hugging Face SAM model id.")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patch-step", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    train_sam(parse_args())
