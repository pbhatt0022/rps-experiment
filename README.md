# Rock Paper Scissors Data Preparation Pipeline

## Overview

This repository prepares a Rock-Paper-Scissors image dataset for downstream machine learning.

It covers the full data-preparation flow:

1. read raw image metadata and human comment annotations
2. clean free-text comments into canonical labels
3. assign one final label per image using vote aggregation and review logic
4. preprocess images into a consistent model-ready format
5. optionally create train/val/test splits
6. optionally package the final dataset as a zip file for handoff

The goal is to give the model team a clean, reproducible dataset instead of raw Google Photos exports and noisy human comments.

## Repository Structure

- `process_rps_comments.py`
  Main comment-cleaning and image-label assignment pipeline.

- `prepare_rps_ml_dataset.py`
  End-to-end handoff script that reruns label cleaning, preprocesses images, optionally creates splits, and optionally creates a zip bundle.

- `tests_simulated_comments.py`
  Small test suite for normalization edge cases.

- `MODEL_HANDOFF.md`
  Shorter handoff-oriented notes for the ML team.

- `DATA_STAGE_REPORT.md`
  Detailed academic-style write-up of the data collection and preprocessing stage.

- `Takeout/Google Photos/rps-data-for-comments/`
  Canonical full dataset exported through Google Takeout and used for official runs.

- `output/`
  Current comment-processing outputs for the chosen threshold `0.6`.

- `output_t05/`, `output_t06/`, `output_t075/`
  Threshold-comparison outputs used during analysis.

## Source Data

The canonical source dataset folder is:

- `./Takeout/Google Photos/rps-data-for-comments`

The full Takeout dataset currently contains:

- `4841` metadata files matched by the processing pipeline
- `4683` source media files
- `4346` `.jpg` files
- `232` `.heic` files
- `59` `.dng` files
- `4` `.png` files
- `42` `.mp4` files

The pipeline treats the shared album comments as raw human annotations.

## Comment-Cleaning Pipeline

### 1. Metadata Extraction

The script reads all `.supplemental-metadata.json` files and extracts:

- image filename
- image metadata
- comment text
- comment owner
- comment timestamp

### 2. Basic Normalization

Basic normalization:

- strips surrounding whitespace
- lowercases text
- collapses repeated internal whitespace

Example:

- `" Rock "` -> `"rock"`

### 3. Advanced Normalization

Advanced normalization:

- applies Unicode normalization
- removes punctuation, emojis, and decorative symbols
- preserves alphabetic text and spaces
- collapses whitespace again

Examples:

- `"rock 🪨"` -> `"rock"`
- `"Paper 📄"` -> `"paper"`

### 4. Canonicalization

Normalized comments are mapped into:

- `rock`
- `paper`
- `scissors`
- `noise`
- `ambiguous`

The mapping uses:

- exact matches
- synonym mapping
- token-based matching
- conservative typo correction using Levenshtein distance
- optional Hamming and Soundex-style fallbacks

Examples:

- `"stone"` -> `rock`
- `"papet"` -> `paper`
- `"sissors"` -> `scissors`
- `"rock paper"` -> `ambiguous`
- `"phone"` -> `noise`

### 5. Image-Level Label Assignment

Each image can have multiple comments. The script:

- counts valid class votes
- finds the majority class
- computes a majority ratio
- sends uncertain images to review

Current review logic:

- if there are no valid comments, mark the image as `review`
- if the majority ratio is above threshold and there are no ambiguous comments, assign the majority label
- otherwise, mark the image as `review`

## Threshold Analysis

Thresholds compared:

- `0.5`
- `0.6`
- `0.75`

Chosen threshold:

- `0.6`

Reason:

- `0.5` and `0.6` gave the same automatic coverage
- `0.75` pushed an extra image into review
- `0.6` gave the best balance between coverage and conservatism

On the full dataset rerun, the review-queue sizes were:

- threshold `0.5`: `488`
- threshold `0.6`: `495`
- threshold `0.75`: `500`

## Existing Comment Outputs

The comment-processing pipeline produces:

- `comment_level_audit.csv`
  Full trace from raw comment to cleaned label.

- `raw_comment_counts.csv`
  Counts of raw comment variants.

- `basic_normalized_counts.csv`
  Counts after basic normalization.

- `advanced_normalized_counts.csv`
  Counts after advanced normalization.

- `canonical_label_counts.csv`
  Final canonical-label counts.

- `image_label_summary.csv`
  One row per image with vote statistics and final decision.

- `review_queue.csv`
  Images that still need manual review.

- `normalization_stats.csv`
  Summary statistics for the normalization pipeline.

- histogram PNG files
  Visual summaries of comment and label distributions.

## Full-Dataset Results

The current official full-dataset run at threshold `0.6` produced:

- total comments processed: `5565`
- unique raw comment variants: `117`
- unique variants after basic normalization: `78`
- unique variants after advanced normalization: `61`

Final canonical comment counts:

- `rock`: `1530`
- `paper`: `1526`
- `scissors`: `1561`
- `noise`: `947`
- `ambiguous`: `1`

Image-level outcomes:

- `2925` image-level entries assessed
- `850` final `scissors`
- `792` final `rock`
- `788` final `paper`
- `495` `review`

Review reasons:

- `485` `no_valid_comments`
- `9` `low_majority_ratio`
- `1` `ambiguous_comments_present`

## Image Preprocessing Pipeline

The integrated image-preprocessing flow is implemented in `prepare_rps_ml_dataset.py`.

By default it:

- reruns the comment-cleaning pipeline
- filters to final labeled images
- excludes `review` images from the training export
- opens source images with Pillow
- applies EXIF orientation correction
- converts images to RGB
- resizes while preserving aspect ratio
- pads to a square canvas
- saves processed outputs grouped by class
- writes manifests and a config file for reproducibility
- optionally exports train-only horizontal-flip augmentation after split creation

### Default Settings

- label threshold: `0.6`
- image size: `128`
- pad color: `0,0,0`
- output format: `jpg`
- JPEG quality: `95`
- resample filter: `lanczos`

### Adjustable Parameters

You can change:

- label threshold
- output directories
- image size
- pad color
- output format
- JPEG quality
- resample filter
- whether review images are included
- whether train/val/test splits are created
- split ratios
- split random seed
- whether a zip bundle is created
- whether train-only horizontal-flip augmentation is created

## Model-Handoff Outputs

A dataset build can produce:

- `images/rock/`, `images/paper/`, `images/scissors/`
  Preprocessed images grouped by label.

- `manifest.csv`
  One row per exported image with source path, processed path, label, original size, processed size, and vote statistics.

- `review_manifest.csv`
  Images that still require manual review.

- `skipped_images.csv`
  Entries that could not be exported into the final image package because the source image was missing or unreadable.

- `dataset_config.json`
  Exact preprocessing settings used for that dataset build.

- `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`
  Created when train/val/test splitting is enabled.

- `splits/train/...`, `splits/val/...`, `splits/test/...`
  Split-specific directories created when splitting is enabled.

- `<dataset-output-dir>.zip`
  Zip archive created when `--zip-output` is used.

For the latest full-dataset export at threshold `0.6`:

- `1985` labeled images were successfully exported
- `445` entries were skipped during image export
- `495` review images were listed separately

Base split counts:

- train: `1588`
- val: `198`
- test: `199`

With train-only horizontal-flip augmentation enabled:

- train rows increase from `1588` to `3176`
- validation and test remain unchanged

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Comment Cleaning Only

```bash
python process_rps_comments.py "./Takeout/Google Photos/rps-data-for-comments" --threshold 0.6 --output-dir output
```

### 2. Run Normalization Tests

```bash
python tests_simulated_comments.py
```

### 3. Build a Model-Ready Dataset

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --label-threshold 0.6 --image-size 128 --dataset-output-dir prepared_dataset
```

### 4. Build a Model-Ready Dataset With Splits And A Zip Bundle

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --label-threshold 0.6 --image-size 128 --dataset-output-dir prepared_dataset --create-splits --augment-train-horizontal-flip --zip-output
```

### 5. Example Parameter Variants

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --image-size 224
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --pad-color 255,255,255
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --output-format png
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --resample bicubic
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --include-review
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --create-splits --split-ratios 0.7,0.15,0.15 --split-seed 7
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --create-splits --augment-train-horizontal-flip
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --zip-output
```

## How To Generate The Email-Ready Zip File

If you want a zip file locally that you can directly email to the model team, run:

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --label-threshold 0.6 --image-size 128 --dataset-output-dir prepared_dataset_handoff --create-splits --augment-train-horizontal-flip --zip-output
```

This will create:

- a dataset folder: `prepared_dataset_handoff/`
- a zip archive: `prepared_dataset_handoff.zip`

The zip file will be created in the repository root beside the script files, so you can attach it directly to an email or upload it to shared storage.

The current full-dataset handoff bundle created in this repo is:

- `prepared_dataset_handoff_full.zip`

## Why This Repository Is Useful For The ML Team

The model team receives:

- cleaned labels instead of raw comments
- a filtered training export that excludes unresolved review images by default
- consistent image dimensions and RGB color mode
- optional train/val/test splits
- optional train-only horizontal-flip augmentation
- a manifest linking every processed image back to its source
- a config file that records the exact preprocessing choices
- an optional zip file for easy handoff

This makes the dataset easier to train on, audit, and regenerate when preprocessing settings change.
