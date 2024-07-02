# RetinaScan

RetinaScan is a diabetic retinopathy analysis pipeline built to help turn retinal fundus images into more useful clinical signals. The project focuses on two connected problems: finding lesion evidence in the image and using that evidence to support disease-grade classification.

The system is organized around three practical goals:

- Identify lesion regions associated with diabetic retinopathy, including microaneurysms, hemorrhages, and hard exudates.
- Generate SAM-based segmentation masks that make those lesion areas easier to inspect and reuse.
- Train a PyTorch CNN to classify retinopathy severity from retinal images.

This is not a clinical diagnostic device. It is a machine-learning project for building, testing, and evaluating a computer-vision approach to diabetic retinopathy analysis.

## How It Works

RetinaScan breaks the problem into two stages.

First, it fine-tunes Meta's Segment Anything Model on retinal lesion masks. Each lesion type is trained separately so the model can focus on the visual structure of one abnormality at a time. The segmentation stage produces lesion masks that can be reviewed directly or used as part of downstream modeling.

Second, it trains a compact convolutional neural network for disease grading. The grading model learns from labeled retinal images and can be evaluated with precision, recall, and accuracy.

## Project Layout

```text
.
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- scripts/
|   |-- train_microaneurysms.ps1
|   |-- train_hemorrhages.ps1
|   `-- train_hard_exudates.ps1
`-- src/diabetic_retinopathy/
    |-- config.py
    |-- data/
    |   |-- grading.py
    |   `-- segmentation.py
    |-- evaluation.py
    |-- inference/
    |   `-- sam_inference.py
    |-- models/
    |   |-- rog_cnn.py
    |   `-- sam_dataset.py
    |-- training/
    |   |-- evaluate_cnn.py
    |   |-- train_cnn.py
    |   `-- train_sam.py
    `-- utils/
        `-- visualization.py
```

## Setup

Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installer before running long jobs.

## Dataset Layout

RetinaScan expects an IDRiD-style dataset layout:

```text
Dataset2/
|-- A. Segmentation/
|   |-- 1. Original Images/
|   |   |-- a. Training Set/
|   |   `-- b. Testing Set/
|   `-- 2. All Segmentation Groundtruths/
|       |-- a. Training Set/
|       |   |-- 1. Microaneurysms/
|       |   |-- 2. Haemorrhages/
|       |   `-- 3. Hard Exudates/
|       `-- b. Testing Set/
|           |-- 1. Microaneurysms/
|           |-- 2. Haemorrhages/
|           `-- 3. Hard Exudates/
`-- B. Disease Grading/
    |-- 1. Original Images/
    |   |-- a. Training Set/
    |   `-- b. Testing Set/
    `-- 2. Groundtruths/
```

Keep large datasets and generated checkpoints outside git. The `.gitignore` excludes `data/`, `checkpoints/`, `outputs/`, and model checkpoint files.

## Train Lesion Segmentation

Train one lesion detector at a time. The output checkpoint represents the model's learned segmentation behavior for that lesion class.

```powershell
python -m diabetic_retinopathy.training.train_sam `
  --image-dir "data\Dataset2\A. Segmentation\1. Original Images\a. Training Set" `
  --mask-dir "data\Dataset2\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\1. Microaneurysms" `
  --output-path "checkpoints\microaneurysms.pth"
```

Convenience PowerShell wrappers are included:

```powershell
.\scripts\train_microaneurysms.ps1 -DatasetRoot "data\Dataset2"
.\scripts\train_hemorrhages.ps1 -DatasetRoot "data\Dataset2"
.\scripts\train_hard_exudates.ps1 -DatasetRoot "data\Dataset2"
```

## Generate Lesion Masks

After training, use a checkpoint to generate predicted masks for retinal images. SAM requires a prompt, so this script builds bounding-box prompts from a mask directory and saves the predicted masks.

```powershell
python -m diabetic_retinopathy.inference.sam_inference `
  --image-dir "data\Dataset2\A. Segmentation\1. Original Images\b. Testing Set" `
  --prompt-mask-dir "data\Dataset2\A. Segmentation\2. All Segmentation Groundtruths\b. Testing Set\1. Microaneurysms" `
  --checkpoint-path "checkpoints\microaneurysms.pth" `
  --output-dir "outputs\microaneurysms"
```

## Train Disease Grading

Train the CNN that predicts retinopathy grade from retinal images:

```powershell
python -m diabetic_retinopathy.training.train_cnn `
  --image-dir "data\Dataset2\B. Disease Grading\1. Original Images\a. Training Set" `
  --labels-csv "data\Dataset2\B. Disease Grading\2. Groundtruths\a. Edited Training Labels - a. IDRiD_Disease Grading_Training Labels.csv" `
  --output-path "checkpoints\grading_cnn.pth"
```

## Evaluate Disease Grading

Evaluate the trained grading model:

```powershell
python -m diabetic_retinopathy.training.evaluate_cnn `
  --image-dir "data\Dataset2\B. Disease Grading\1. Original Images\b. Testing Set" `
  --labels-csv "data\Dataset2\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv" `
  --checkpoint-path "checkpoints\grading_cnn.pth"
```

## Implementation Notes

- SAM training uses `facebook/sam-vit-base`, 512-pixel patches, DiceCE loss, and fine-tunes only the mask decoder by default.
- The CNN grading model is intentionally compact so the disease-grading stage is easy to train, inspect, and modify.
- Dependencies are defined in `pyproject.toml` and `requirements.txt`.
- Validate data splits, labels, and metrics carefully before using results in any medical or clinical context.

## Attribution

The SAM fine-tuning portion of this project was adapted from the `331_fine_tune_SAM_mito.ipynb` notebook in
[bnsreenu/python_for_microscopists](https://github.com/bnsreenu/python_for_microscopists), which demonstrates fine-tuning Meta's Segment Anything Model using bounding-box prompts and mask supervision.

I adapted this workflow for diabetic retinopathy analysis using the IDRiD dataset, including preprocessing retinal scans and lesion masks, training segmentation workflows for microaneurysms, hemorrhages, and hard exudates, generating SAM-based lesion masks, and using these outputs as part of a PyTorch CNN disease-grading pipeline.
