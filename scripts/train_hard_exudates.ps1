param(
    [Parameter(Mandatory = $true)] [string] $DatasetRoot
)

python -m diabetic_retinopathy.training.train_sam `
    --image-dir "$DatasetRoot\A. Segmentation\1. Original Images\a. Training Set" `
    --mask-dir "$DatasetRoot\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\3. Hard Exudates" `
    --output-path "checkpoints\hard_exudates.pth"
