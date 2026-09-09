# TeaBD-HybridNet

A maximum-pretrained CNN–ViT hybrid architecture for tea leaf disease and pest classification on the teaLeafBD dataset, benchmarked against ten strong baselines spanning CNN, Vision Transformer, existing hybrid, and state-space model families.

## Overview

This repository contains a single Jupyter notebook that implements, trains, and evaluates **TeaBD-HybridNet**, a dual-branch architecture combining a fully pretrained ConvNeXt-Tiny (CNN) backbone with a fully pretrained DeiT-Small (ViT) backbone — all 12 transformer blocks transplanted, not a truncated subset — fused via simple channel concatenation and a lightweight classification head.

The core design principle: **maximize transferred pretrained knowledge, minimize randomly initialized parameters.** On a modestly sized dataset (5,276 images), large randomly initialized fusion modules tend to undertrain and underperform relative to fully pretrained single-architecture models. TeaBD-HybridNet avoids this by keeping virtually everything pretrained except a small classification head.

**Result:** 95.11% accuracy / 94.17% macro-F1 on the test set, outperforming all ten baselines including Swin-Tiny (93.52% F1), MobileViT-S (92.01% F1), and MambaVision-T (91.89% F1).

## Dataset

teaLeafBD — 5,276 RGB images across 7 classes:
- 6 disease/pest classes: algal leaf spot, brown blight, gray blight, helopeltis, red spider, green mirid bug
- 1 healthy class

Class-imbalanced (418–1,282 images per class). Split 70/15/15 (train/val/test) via stratified sampling, seed 42.

## What's in the notebook

The notebook (`TeaBD-HybridNet.ipynb`) runs end-to-end, top to bottom, and is organized into these sections:

| Section | What it does |
|---|---|
| 1–5 | Setup, config, stratified data split, dataset/transform classes, sample visualization |
| 6 | `TeaBD-HybridNet` architecture definition (CNN branch, ViT branch, fusion, classifier) |
| 7 | Fail-safe training loop: atomic checksummed checkpoints, auto-resume, EMA, mixup/cutmix, focal loss, warmup+cosine LR |
| 7b | Stochastic Weight Averaging (SWA) and Test-Time Augmentation (TTA) utilities |
| 8 | Trains TeaBD-HybridNet (single-stage, layer-wise LR, 150 epochs, patience 30) |
| 9–11 | Training curves, test-set evaluation, ROC/PR curves, calibration analysis |
| 12–14 | Baseline comparison chart, FLOPs/parameter count, checkpoint health check |
| 15 | Trains 10 baseline models (ResNet-50, EfficientNet-B3, MobileNetV3-Large, DenseNet-201, ViT-Small, DeiT-Small, Swin-Tiny, MobileViT-S, ConvNeXt-Tiny, MambaVision-T) under an identical protocol |
| 16–17 | Explainability suite (Grad-CAM, LIME, SHAP, t-SNE) and robustness evaluation (FGSM/PGD-20/CW-L2 adversarial attacks, 5-type image corruption sweep, PGD adversarial training defense) |
| 18 | Compiles all results into master comparison tables, per-class metrics, and a results dashboard |
| 19 | Repeats the explainability suite on the best-performing baseline for side-by-side comparison |

## Requirements

pip install torch torchvision timm scikit-learn pandas matplotlib seaborn thop pytorch-grad-cam lime shap torchattacks mambavision --break-system-packages

MambaVision additionally requires `mamba-ssm` and `causal-conv1d`, which need CUDA build tools (`nvcc`, matching gcc version) to compile from source. See notes below if you hit build errors.

**Hardware used:** single NVIDIA RTX 3050 (8GB VRAM), batch size 16, mixed precision (AMP).

## Running it

1. Update `DATASET_ROOT` in the Configuration cell to point to your local copy of teaLeafBD.
2. Run all cells top to bottom. Training is fully resumable — if interrupted, re-running the same cell picks up from the last valid checkpoint.
3. Outputs are written to `./results/` (figures, tables, robustness/XAI outputs) and `./checkpoints/` (model weights, checksum-verified).

## Output structure

    results/
    ├── figures/        training curves, confusion matrices, comparison charts
    ├── tables/         CSV files (baseline_results, master_model_comparison, per_class_performance, ...)
    ├── history/        per-model training history (JSON)
    ├── xai/            Grad-CAM, LIME, SHAP, t-SNE visualizations
    └── robustness/     adversarial + corruption robustness results and plots
    checkpoints/        model weights (.pt) with .sha256 checksums

## Results summary

| Rank | Model | Params (M) | FLOPs (G) | Test Acc | Test F1 |
|---|---|---|---|---|---|
| 1 | **TeaBD-HybridNet (ours)** | 49.78 | 8.70 | **0.9511** | **0.9417** |
| 2 | Swin-Tiny | 27.52 | 4.37 | 0.9461 | 0.9352 |
| 3 | MobileViT-S | 4.94 | 1.42 | 0.9261 | 0.9201 |
| 4 | MambaVision-T | 31.16 | 4.47 | 0.9298 | 0.9189 |
| ... | (7 more baselines) | | | | |

Full results, per-class breakdown, and robustness evaluation are in `results/tables/master_model_comparison.csv` and `results/tables/per_class_performance.csv`.

## Notes on setup issues

If building `mamba-ssm`/`causal-conv1d` from source fails with `cuda_runtime.h not found` or `unsupported GNU version`, you likely need:

    conda install -c nvidia/label/cuda-12.1.0 cuda-cudart-dev cuda-nvcc -y
    conda install -c conda-forge gcc=12 gxx=12 -y

then build with `--no-build-isolation --no-deps` and explicit `CC=`/`CXX=` pointing at the conda-installed gcc-12.

## License

MIT License

Copyright (c) 2026 S M Nafim Niloy

## Citation

[paper reference will be added once published]
