# RPS Model Handoff

## What This Repository Now Covers

This repository now supports the full data-preparation flow for the Rock-Paper-Scissors dataset:

1. clean free-text human comments into canonical labels
2. assign image-level labels using majority vote and a configurable threshold
3. preprocess images into a consistent square format for model training
4. export a manifest and config so the ML team can trace and reproduce the dataset

## Main Scripts

- `process_rps_comments.py`
  Comment normalization and image-level label assignment.

- `prepare_rps_ml_dataset.py`
  End-to-end handoff script that reruns label cleaning and then exports processed images plus manifests.

- `tests_simulated_comments.py`
  Simulated comment test coverage for normalization edge cases.

## Model-Ready Export

Running `prepare_rps_ml_dataset.py` creates:

- `manifest.csv`
  One row per exported image with source path, processed path, label, and vote statistics.

- `review_manifest.csv`
  Images that still require manual review instead of automatic training export.

- `dataset_config.json`
  The exact preprocessing parameters used for that dataset build.

- `images/rock/`, `images/paper/`, `images/scissors/`
  Preprocessed training images grouped by class.

- `splits/train/`, `splits/val/`, `splits/test/`
  Optional split-specific directories for the ML team when `--create-splits` is used.

- `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`
  Optional split-specific manifests when `--create-splits` is used.

- `skipped_images.csv`
  Only created if some files could not be opened or processed.

## Default Image Preprocessing

The current integrated image preprocessing does the following:

- opens images with Pillow
- applies EXIF orientation correction
- converts to RGB
- preserves aspect ratio
- resizes to a configurable square size
- pads the remaining area with a configurable background color
- exports to `jpg` or `png`

Default settings:

- label threshold: `0.6`
- image size: `128`
- pad color: `0,0,0`
- output format: `jpg`
- resample filter: `lanczos`
- train/val/test splits: optional, stratified, reproducible with a seed

## Usage

### Comment Processing Only

```bash
python process_rps_comments.py ./rps-data-for-comments --threshold 0.6 --output-dir output
```

### Full Model Handoff Build

```bash
python prepare_rps_ml_dataset.py ./rps-data-for-comments --label-threshold 0.6 --image-size 128 --dataset-output-dir prepared_dataset --create-splits --zip-output
```

### Common Parameter Changes

```bash
python prepare_rps_ml_dataset.py ./rps-data-for-comments --image-size 224
python prepare_rps_ml_dataset.py ./rps-data-for-comments --pad-color 255,255,255
python prepare_rps_ml_dataset.py ./rps-data-for-comments --output-format png
python prepare_rps_ml_dataset.py ./rps-data-for-comments --resample bicubic
python prepare_rps_ml_dataset.py ./rps-data-for-comments --create-splits --split-ratios 0.7,0.15,0.15 --split-seed 7
python prepare_rps_ml_dataset.py ./rps-data-for-comments --include-review
python prepare_rps_ml_dataset.py ./rps-data-for-comments --zip-output
```

## Why This Is Easier For The ML Team

The model team gets:

- cleaned labels instead of raw comments
- a training export that excludes unresolved review images by default
- consistent image dimensions and RGB color mode
- optional stratified train/val/test splits ready for training workflows
- a manifest that links every processed file back to its source
- a config file that makes preprocessing choices explicit and repeatable
- an optional zip bundle for easy handoff
