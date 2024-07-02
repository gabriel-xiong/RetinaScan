"""Run a trained SAM lesion segmentation checkpoint on images with mask prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SamConfig, SamModel, SamProcessor

from diabetic_retinopathy.data.segmentation import build_full_size_segmentation_dataset
from diabetic_retinopathy.models.sam_dataset import get_bounding_box


def load_sam_checkpoint(checkpoint_path: Path, base_model: str, device: str) -> SamModel:
    """Create a SAM model and load local fine-tuned weights."""

    model_config = SamConfig.from_pretrained(base_model)
    model = SamModel(config=model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_mask(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    prompt_mask: Image.Image,
    device: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a binary segmentation mask and return it with probabilities."""

    prompt = get_bounding_box(np.asarray(prompt_mask))
    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    resized_masks = F.interpolate(
        outputs.pred_masks.squeeze(1),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    probabilities = torch.sigmoid(resized_masks).cpu().numpy().squeeze()
    binary_mask = (probabilities > threshold).astype(np.uint8)
    return binary_mask, probabilities


def run_inference(args: argparse.Namespace) -> None:
    """Generate masks for every image/mask pair in the provided directories."""

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    processor = SamProcessor.from_pretrained(args.base_model)
    model = load_sam_checkpoint(args.checkpoint_path, args.base_model, device)
    dataset = build_full_size_segmentation_dataset(args.image_dir, args.prompt_mask_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(dataset):
        try:
            binary_mask, probabilities = predict_mask(
                model=model,
                processor=processor,
                image=item["image"],
                prompt_mask=item["label"],
                device=device,
                threshold=args.threshold,
            )
        except ValueError as error:
            print(f"Skipping image {index}: {error}")
            continue

        Image.fromarray(binary_mask * 255).save(args.output_dir / f"mask_{index:04d}.png")
        if args.save_probabilities:
            np.save(args.output_dir / f"probability_{index:04d}.npy", probabilities)

    print(f"Saved {len(dataset)} predicted masks to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Images to segment.")
    parser.add_argument(
        "--prompt-mask-dir",
        type=Path,
        required=True,
        help="Ground-truth or seed masks used only to create SAM box prompts.",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Trained SAM checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for predicted masks.")
    parser.add_argument("--base-model", default="facebook/sam-vit-base")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
