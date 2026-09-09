"""
tealeaf_dataset.py
PyTorch Dataset + transforms for teaLeafBD, using REAL dataset statistics
computed from dataset_report.json:

    pixel_mean = [0.6979, 0.6996, 0.6627]
    pixel_std  = [0.1672, 0.1541, 0.2068]

7 classes, moderate imbalance (3.07x, Green mirid bug=1282 vs Tea algal leaf spot=418).
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Real dataset statistics (from your dataset_report.json)
# ---------------------------------------------------------------------------
TEALEAF_MEAN = [0.6979, 0.6996, 0.6627]
TEALEAF_STD = [0.1672, 0.1541, 0.2068]

IMG_SIZE = 224

LABEL_LIST = [
    "algal_leaf_spot",
    "brown_blight",
    "gray_blight",
    "helopeltis",
    "red_spider",
    "green_mirid_bug",
    "healthy",
]

# Actual counts from your report (used for class-weighted loss / sampler)
CLASS_COUNTS = {
    "algal_leaf_spot": 418,
    "brown_blight": 506,
    "gray_blight": 1013,
    "helopeltis": 607,
    "red_spider": 515,
    "green_mirid_bug": 1282,
    "healthy": 935,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TeaLeafBDDataset(Dataset):
    """
    Reads a split CSV produced by 00_prepare_data.py with columns:
        filepath, label, label_idx
    """

    def __init__(self, csv_path, transform=None):
        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filepath"], int(row["label_idx"]), row["label"]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label_idx, label_name = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx

    def get_labels(self):
        """Used by WeightedRandomSampler to know the label of every sample."""
        return [s[1] for s in self.samples]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform(img_size=IMG_SIZE):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),          # leaves can appear at any orientation in-field
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # Cutout-style regularization
        T.Normalize(mean=TEALEAF_MEAN, std=TEALEAF_STD),
    ])


def get_eval_transform(img_size=IMG_SIZE):
    # deterministic, no augmentation - used for val/test
    return T.Compose([
        T.Resize(int(img_size * 1.14)),   # resize shorter side then center crop (standard practice)
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=TEALEAF_MEAN, std=TEALEAF_STD),
    ])


# ---------------------------------------------------------------------------
# Class imbalance helpers
# ---------------------------------------------------------------------------
def get_class_weights():
    """
    Inverse-frequency class weights, normalized so mean weight = 1.0.
    Use with nn.CrossEntropyLoss(weight=get_class_weights())
    Given the moderate 3.07x imbalance here, standard weighted CE
    (rather than focal loss) should be sufficient - but both are easy
    to swap in train.py.
    """
    counts = torch.tensor([CLASS_COUNTS[c] for c in LABEL_LIST], dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights


def get_sample_weights(dataset: TeaLeafBDDataset):
    """
    Per-sample weights for WeightedRandomSampler (alternative to loss weighting).
    """
    class_w = get_class_weights()
    labels = dataset.get_labels()
    return torch.tensor([class_w[label_idx] for label_idx in labels], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    split_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/splits"

    train_ds = TeaLeafBDDataset(f"{split_dir}/train.csv", transform=get_train_transform())
    val_ds = TeaLeafBDDataset(f"{split_dir}/val.csv", transform=get_eval_transform())
    test_ds = TeaLeafBDDataset(f"{split_dir}/test.csv", transform=get_eval_transform())

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    img, label = train_ds[0]
    print(f"Sample image tensor shape: {img.shape}, label: {label} ({LABEL_LIST[label]})")

    weights = get_class_weights()
    print("\nClass weights (for weighted CE loss):")
    for name, w in zip(LABEL_LIST, weights.tolist()):
        print(f"  {name:20s}: {w:.3f}")
