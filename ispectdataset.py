"""
inspect_dataset.py
Run this on your machine (Ubuntu, Jupyter) after downloading teaLeafBD.
It will print out everything needed to understand the dataset structure:
- folder tree
- class names & counts
- image formats, sizes, corrupt file check
- sample image dimensions/aspect ratios
- basic pixel statistics (mean/std) for normalization

Usage:
    python inspect_dataset.py --root /path/to/teaLeafBD
"""

import os
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def find_dataset_root(root):
    """Auto-detect the actual class-folder root if there's a nested structure."""
    root = Path(root)
    candidates = [root]
    # walk one or two levels deep looking for a directory with several subfolders
    for depth in range(3):
        new_candidates = []
        for c in candidates:
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if len(subdirs) >= 3:  # looks like class folders
                return c
            new_candidates.extend(subdirs)
        candidates = new_candidates
    return root


def scan_classes(root):
    root = Path(root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    stats = {}
    for cdir in class_dirs:
        files = [f for f in cdir.iterdir() if f.suffix.lower() in
                 ['.jpg', '.jpeg', '.png', '.bmp']]
        stats[cdir.name] = files
    return stats


def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # reopen after verify (verify closes it)
            w, h = img.size
            mode = img.mode
        return {"ok": True, "width": w, "height": h, "mode": mode}
    except (UnidentifiedImageError, OSError) as e:
        return {"ok": False, "error": str(e)}


def compute_pixel_stats(file_list, sample_size=200):
    """Estimate per-channel mean/std from a random sample (for normalization)."""
    sample = np.random.choice(file_list, size=min(sample_size, len(file_list)), replace=False)
    means, stds = [], []
    for f in sample:
        try:
            img = Image.open(f).convert("RGB").resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            means.append(arr.mean(axis=(0, 1)))
            stds.append(arr.std(axis=(0, 1)))
        except Exception:
            continue
    if not means:
        return None, None
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to the downloaded teaLeafBD folder")
    parser.add_argument("--sample_for_stats", type=int, default=200,
                         help="How many images per class to sample for pixel stats")
    parser.add_argument("--check_corrupt", action="store_true",
                         help="Verify every image file (slower, but thorough)")
    parser.add_argument("--out", type=str, default="dataset_report.json",
                         help="Where to save the JSON report")
    args = parser.parse_args()

    detected_root = find_dataset_root(args.root)
    print(f"\n[INFO] Using dataset root: {detected_root}\n")

    class_files = scan_classes(detected_root)

    if not class_files:
        print("[WARNING] No class subfolders found directly under root. "
              "Printing full tree instead:\n")
        for p in Path(args.root).rglob("*"):
            print(p)
        return

    report = {"root": str(detected_root), "classes": {}}

    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    total = 0
    for cname, files in class_files.items():
        count = len(files)
        total += count
        print(f"{cname:30s} : {count:5d} images")
        report["classes"][cname] = {"count": count}
    print("-" * 60)
    print(f"{'TOTAL':30s} : {total:5d} images")
    print("=" * 60)

    # Imbalance ratio
    counts = {c: len(f) for c, f in class_files.items()}
    max_c, min_c = max(counts, key=counts.get), min(counts, key=counts.get)
    print(f"\n[INFO] Largest class : {max_c} ({counts[max_c]})")
    print(f"[INFO] Smallest class: {min_c} ({counts[min_c]})")
    print(f"[INFO] Imbalance ratio (max/min): {counts[max_c] / counts[min_c]:.2f}x\n")

    # Image format / dimension check (sample-based, fast)
    print("=" * 60)
    print("IMAGE FORMAT / DIMENSION CHECK (sampled)")
    print("=" * 60)
    dims = defaultdict(int)
    formats = defaultdict(int)
    corrupt_files = []

    for cname, files in class_files.items():
        sample = files if args.check_corrupt else files[:20]  # fast check by default
        for f in sample:
            info = check_image(f)
            if not info["ok"]:
                corrupt_files.append(str(f))
                continue
            dims[(info["width"], info["height"])] += 1
            formats[f.suffix.lower()] += 1

    print("\nTop 10 image dimensions found:")
    for dim, cnt in sorted(dims.items(), key=lambda x: -x[1])[:10]:
        print(f"  {dim[0]}x{dim[1]:5d} : {cnt} images")

    print("\nFile formats found:")
    for fmt, cnt in formats.items():
        print(f"  {fmt}: {cnt}")

    if corrupt_files:
        print(f"\n[WARNING] {len(corrupt_files)} corrupt/unreadable files found:")
        for f in corrupt_files[:10]:
            print(f"  {f}")
        if len(corrupt_files) > 10:
            print(f"  ... and {len(corrupt_files) - 10} more")
    else:
        print("\n[OK] No corrupt files detected in sample.")

    report["corrupt_files"] = corrupt_files
    report["dimension_sample"] = {f"{k[0]}x{k[1]}": v for k, v in dims.items()}
    report["format_sample"] = dict(formats)

    # Pixel statistics (for normalization in transforms)
    print("\n" + "=" * 60)
    print("PIXEL STATISTICS (sampled, for Normalize() transform)")
    print("=" * 60)
    all_files = [f for files in class_files.values() for f in files]
    mean, std = compute_pixel_stats(all_files, sample_size=args.sample_for_stats)
    if mean is not None:
        print(f"Estimated mean (RGB): {mean.tolist()}")
        print(f"Estimated std  (RGB): {std.tolist()}")
        report["pixel_mean"] = mean.tolist()
        report["pixel_std"] = std.tolist()
    else:
        print("[WARNING] Could not compute pixel stats.")

    # Save report
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Full report saved to: {args.out}")
    print("\nShare this JSON (or the printed output above) so the architecture "
          "and data pipeline can be tailored exactly to your dataset.")


if __name__ == "__main__":
    main()
