"""Plot helpers for checking images, masks, and SAM predictions."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def show_image_and_mask(image: Image.Image, mask: Image.Image, title: str = "Mask") -> None:
    """Display an image beside its mask for quick dataset verification."""

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.asarray(image))
    axes[0].set_title("Image")
    axes[1].imshow(np.asarray(mask), cmap="gray")
    axes[1].set_title(title)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.show()


def show_sam_prediction(image: Image.Image, binary_mask: np.ndarray, probability_map: np.ndarray) -> None:
    """Display the fundus image, predicted mask, and probability map."""

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.asarray(image))
    axes[0].set_title("Image")
    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[2].imshow(probability_map)
    axes[2].set_title("Probability Map")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.show()
