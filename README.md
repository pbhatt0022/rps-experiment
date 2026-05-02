# RPS Experiment: Data Preparation for an MLOps Project

This repository contains the data-preparation stage of a semester-long university MLOps project built around a deliberately simple task: classify hand-sign images as `rock`, `paper`, or `scissors`.

The simplicity is the point. Rock-Paper-Scissors is easy to understand, which makes it a good vehicle for studying the parts of machine learning systems that are usually harder to see: messy annotation, inconsistent source data, review queues, preprocessing decisions, reproducibility, and handoff between teams or pipeline stages.

This work was developed in the context of an `MLOps & Model Deployment` course, but the repository is written to be understandable to readers outside that course as well. If you are visiting from GitHub, the best way to read this project is as a small, concrete case study in how MLOps begins at the data stage, not only at deployment time.

## What The RPS Experiment Is

- A collaborative image collection project using student-contributed Rock-Paper-Scissors hand-sign photos.
- A human-labeling workflow that used Google Photos comments as raw annotations.
- A Python pipeline that cleans those comments, aggregates votes, and prepares a model-ready dataset.
- A teaching exercise in the practical gap between "we have data" and "we have a reliable dataset."

## The Ethos Behind The Project

This repository is driven by a simple MLOps idea: keep the prediction task small so the pipeline problems become impossible to ignore.

Part of the point of spending serious time on a small problem is to understand the fundamentals well. Bigger projects often hide those fundamentals behind scale, tooling, or model complexity. This one tries to make them visible.

The main lessons behind the experiment are:

- Human labels are not ground truth. They are raw signals that need cleaning, normalization, aggregation, and auditing.
- Convenience during data collection often creates complexity later in preprocessing and quality control.
- Review queues are a feature, not a failure. Low-confidence items should be surfaced instead of being forced into the training set.
- Reproducibility starts before model training. The data-preparation path, thresholds, manifests, and outputs all matter.
- Even a classroom-scale project can expose real MLOps concerns such as traceability, handoff quality, and source-of-truth drift.

## Why Rock-Paper-Scissors?

On paper, this is a very simple computer vision problem. That is exactly why it is useful.

Because the labels are familiar and the classes are easy to explain, the project can focus on deeper operational questions:

- What happens when annotators use typos, emojis, slang, or irrelevant comments?
- How conservative should an automatic label-assignment rule be?
- What should happen to images with weak agreement or no valid labels?
- What happens when metadata exists but the corresponding source image is missing or unreadable?
- How do we package the final dataset so another team can actually use and reproduce it?

The goal is not to claim that Rock-Paper-Scissors is a difficult modeling benchmark. The goal is to use a small problem to study foundational MLOps behavior in a controlled, visible way.

## What This Repository Covers

This repository focuses on the data-preparation side of the MLOps lifecycle:

1. Read Google Takeout metadata and shared-album comments.
2. Normalize messy free-text annotations.
3. Map comments into canonical labels.
4. Aggregate multiple comments into one image-level decision.
5. Route uncertain or unresolved items into a manual review queue.
6. Preprocess accepted images into a consistent model-ready format.
7. Optionally create stratified splits, train-only augmentation, and a zip bundle for handoff.

This is intentionally a data-stage repository. It does not try to present a full production deployment system in one place. Instead, it makes the upstream dataset work explicit and inspectable.

## Current Dataset Snapshot

The canonical raw dataset used by the pipeline is:

- `Takeout/Google Photos/rps-data-for-comments/`

Current full-dataset figures from the latest repository outputs:

- `4,820` supplemental metadata files processed
- `4,683` source media files
- `4,346` `.jpg` files
- `232` `.heic` files
- `59` `.dng` files
- `4` `.png` files
- `42` `.mp4` files
- `5,565` comments processed

Current image-level outcomes at threshold `0.6`:

- `850` final `scissors`
- `792` final `rock`
- `788` final `paper`
- `495` `review`

That means `2,430` images were automatically assigned a final class label, while `495` were held back for manual review.

## Why Some Images Are Reviewed Or Skipped

The pipeline is conservative by design.

- If an image has no valid class comments, it is marked `review`.
- If the majority vote is too weak, it is marked `review`.
- If ambiguous comments are present, it is marked `review`.
- If a labeled item cannot be matched to a usable source image during export, it is written to `skipped_images.csv` instead of silently disappearing.

This is one of the core cautionary lessons of the project: uncertainty should be recorded and surfaced, not hidden behind forced labels or incomplete exports.

## What The Pipeline Produces

By default, new local runs are written under `runs/`, which keeps generated files out of the repo root and out of Git.

The comment-processing stage writes outputs such as:

- `runs/comment_pipeline/current/comment_level_audit.csv`
  Full trace from raw comment to cleaned label.
- `runs/comment_pipeline/current/raw_comment_counts.csv`
  Frequency of raw comment variants.
- `runs/comment_pipeline/current/basic_normalized_counts.csv`
  Frequency after basic normalization.
- `runs/comment_pipeline/current/advanced_normalized_counts.csv`
  Frequency after advanced normalization.
- `runs/comment_pipeline/current/canonical_label_counts.csv`
  Final canonical label counts.
- `runs/comment_pipeline/current/image_label_summary.csv`
  One row per image with vote statistics and final decision.
- `runs/comment_pipeline/current/review_queue.csv`
  Images that require manual review.
- `runs/comment_pipeline/current/normalization_stats.csv`
  Summary statistics for the normalization pipeline.

Curated threshold-comparison snapshots that are useful for understanding the project are kept under:

- `artifacts/comment_pipeline/current_snapshot/`
- `artifacts/comment_pipeline/threshold_050_snapshot/`
- `artifacts/comment_pipeline/threshold_060_snapshot/`
- `artifacts/comment_pipeline/threshold_075_snapshot/`

The dataset-build stage writes outputs such as:

- `runs/model_handoff/prepared_dataset/manifest.csv`
  One row per exported image with source path, processed path, label, and preprocessing metadata.
- `runs/model_handoff/prepared_dataset/review_manifest.csv`
  Review items excluded from the training export.
- `runs/model_handoff/prepared_dataset/skipped_images.csv`
  Export failures caused by missing or unreadable source images.
- `runs/model_handoff/prepared_dataset/dataset_config.json`
  Exact preprocessing settings used for the build.
- `runs/model_handoff/prepared_dataset/train_manifest.csv`
- `runs/model_handoff/prepared_dataset/val_manifest.csv`
- `runs/model_handoff/prepared_dataset/test_manifest.csv`
- `runs/model_handoff/prepared_dataset.zip`

For the latest full handoff export:

- `1,985` labeled images were successfully exported
- `445` items were skipped during image export
- `495` review images were listed separately
- train / validation / test counts were `1,588 / 198 / 199`
- train-only horizontal-flip augmentation expands train rows from `1,588` to `3,176`

## Repository Structure

- `process_rps_comments.py`
  Comment normalization, canonicalization, and image-level label assignment.
- `prepare_rps_ml_dataset.py`
  End-to-end preprocessing and handoff packaging script.
- `contributor_stats.py`
  Optional utility for comment-contributor statistics and audit outputs.
- `tests_simulated_comments.py`
  Small normalization test suite covering noisy human-label cases.
- `docs/`
  Project-facing documentation, including the model handoff note and the detailed report PDF.
- `artifacts/comment_pipeline/`
  Curated committed snapshots used for threshold analysis and repo documentation.
- `runs/`
  Git-ignored local outputs from pipeline runs and handoff builds.
- `Takeout/Google Photos/rps-data-for-comments/`
  Canonical source dataset folder used for local official runs and ignored from Git.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Process Comments Only

```bash
python process_rps_comments.py "./Takeout/Google Photos/rps-data-for-comments"
```

Default output location:

- `runs/comment_pipeline/current/`

### 2. Run Normalization Tests

```bash
python tests_simulated_comments.py
```

### 3. Build A Model-Ready Dataset

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments"
```

Default output locations:

- `runs/model_handoff/comment_pipeline/`
- `runs/model_handoff/prepared_dataset/`

### 4. Build Splits, Augmentation, And A Zip Handoff

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --label-threshold 0.6 --image-size 128 --dataset-output-dir runs/model_handoff/prepared_dataset_handoff --create-splits --augment-train-horizontal-flip --zip-output
```

### 5. Common Variants

```bash
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --image-size 224
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --pad-color 255,255,255
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --output-format png
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --resample bicubic
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --include-review
python prepare_rps_ml_dataset.py "./Takeout/Google Photos/rps-data-for-comments" --create-splits --split-ratios 0.7,0.15,0.15 --split-seed 7
```

### 6. Generate Contributor Statistics

```bash
python contributor_stats.py "./Takeout/Google Photos/rps-data-for-comments"
```

Default output location:

- `runs/contributor_stats/current/`

## How To Read This Repository

If you want the quick story:

- Read this `README` for the experiment rationale and the pipeline overview.
- Read `docs/model_handoff.md` if you care about the dataset as a downstream ML artifact.
- Read `docs/reports/mlops_data_report.pdf` if you want the detailed report-style explanation of collection, labeling, preprocessing, and observed failure modes.

If you are evaluating this repository from outside the course context, the most important takeaway is this:

The project is not valuable because Rock-Paper-Scissors is novel. It is valuable because it turns a familiar toy problem into a concrete lesson about how fragile, human, and operational the data side of MLOps really is.
