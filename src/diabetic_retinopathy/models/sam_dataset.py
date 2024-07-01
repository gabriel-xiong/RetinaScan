"""Dataset adapter for Hugging Face's Segment Anything implementation."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


def get_bounding_box(mask: np.ndarray, max_perturbation: int = 20) -> list[int]:
    """Create a slightly randomized box prompt around a binary mask."""

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    height, width = mask.shape

    x_min = max(0, x_min - np.random.randint(0, max_perturbation))
    x_max = min(width, x_max + np.random.randint(0, max_perturbation))
    y_min = max(0, y_min - np.random.randint(0, max_perturbation))
    y_max = min(height, y_max + np.random.randint(0, max_perturbation))
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


class SAMDataset(Dataset):
    """Serve SAM-ready tensors generated from images, masks, and box prompts."""

    def __init__(self, dataset, processor) -> None:
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        image = item["image"]
        ground_truth_mask = np.asarray(item["label"])
        prompt = get_bounding_box(ground_truth_mask)

        # The processor returns a batch dimension; squeeze it because the
        # DataLoader will create the real batch dimension later.
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs
