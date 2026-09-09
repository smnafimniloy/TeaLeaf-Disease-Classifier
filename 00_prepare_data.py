"""
00_prepare_data.py
Creates stratified train/val/test CSV splits from the real teaLeafBD folder structure.

Usage:
    python 00_prepare_data.py \
        --root "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/An Indivisual Research/teaLeafBD/teaLeafBD" \
        --out_dir ./data/splits \
        --seed 42
"""

import argparse
import random
from pathlib import Path
import csv
from collections import defaultdict

# Fixed, human-readable label mapping derived from your dataset_report.json
# (folder name -> short label used everywhere in code/paper)
CLASS_MAP = {
    "1. Tea algal leaf spot": "algal_leaf_spot",
    "2. Brown Blight": "brown_blight",
    "3. Gray Blight": "gray_blight",
    "4. Helopeltis": "helopeltis",
    "5. Red spider": "red_spider",
    "6. Green mirid bug": "green_mirid_bug",
    "7. Healthy leaf": "healthy",
}

# Fixed class index order (important: keep this order everywhere -
# model output index i always corresponds to LABEL_LIST[i])
LABEL_LIST = [
    "algal_leaf_spot",
    "brown_blight",
    "gray_blight",
    "helopeltis",
    "red_spider",
    "green_mirid_bug",
    "healthy",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_LIST)}


def stratified_split(files, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split a list of files into train/val/test, shuffled deterministically."""
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to teaLeafBD/teaLeafBD folder (contains the 7 class subfolders)")
    parser.add_argument("--out_dir", type=str, default="./data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = {"train": [], "val": [], "test": []}
    class_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for folder_name, short_label in CLASS_MAP.items():
        class_dir = root / folder_name
        if not class_dir.exists():
            print(f"[WARNING] Folder not found, skipping: {class_dir}")
            continue

        files = sorted([str(f) for f in class_dir.iterdir()
                         if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        train_files, val_files, test_files = stratified_split(
            files, args.train_ratio, args.val_ratio, seed=args.seed
        )

        label_idx = LABEL_TO_IDX[short_label]

        for f in train_files:
            all_rows["train"].append((f, short_label, label_idx))
        for f in val_files:
            all_rows["val"].append((f, short_label, label_idx))
        for f in test_files:
            all_rows["test"].append((f, short_label, label_idx))

        class_counts[short_label]["train"] = len(train_files)
        class_counts[short_label]["val"] = len(val_files)
        class_counts[short_label]["test"] = len(test_files)

        print(f"{short_label:20s} (from '{folder_name}'): "
              f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}, "
              f"total={len(files)}")

    # Shuffle each split (so classes are interleaved, not grouped) and write CSVs
    rng = random.Random(args.seed)
    for split_name, rows in all_rows.items():
        rng.shuffle(rows)
        out_path = out_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "label_idx"])
            writer.writerows(rows)
        print(f"[SAVED] {out_path} ({len(rows)} rows)")

    # Save the label mapping too, so training/eval scripts stay in sync
    label_map_path = out_dir / "label_map.csv"
    with open(label_map_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "label_idx"])
        for name, idx in LABEL_TO_IDX.items():
            writer.writerow([name, idx])
    print(f"[SAVED] {label_map_path}")

    print("\n" + "=" * 60)
    print("FINAL SPLIT SUMMARY")
    print("=" * 60)
    print(f"{'Class':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'Total':>8s}")
    for label in LABEL_LIST:
        c = class_counts[label]
        total = c["train"] + c["val"] + c["test"]
        print(f"{label:20s} {c['train']:8d} {c['val']:8d} {c['test']:8d} {total:8d}")

    total_train = sum(c["train"] for c in class_counts.values())
    total_val = sum(c["val"] for c in class_counts.values())
    total_test = sum(c["test"] for c in class_counts.values())
    print("-" * 60)
    print(f"{'TOTAL':20s} {total_train:8d} {total_val:8d} {total_test:8d} "
          f"{total_train + total_val + total_test:8d}")


if __name__ == "__main__":
    main()
