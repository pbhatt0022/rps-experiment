# Data Stage Report for the Rock-Paper-Scissors Dataset Project

## 1. Introduction

This report documents the data-stage workflow for the Rock-Paper-Scissors image classification project. The objective of this stage was to transform collaboratively collected raw images and human-generated labels into a clean, reproducible, and model-ready dataset suitable for downstream machine learning.

The data stage included:

- data sourcing from student contributors
- label collection through Google Photos comments
- export through Google Takeout
- comment cleaning and canonicalization
- image-level majority-vote labeling
- review handling for unresolved cases
- image preprocessing and packaging for model handoff

The canonical raw dataset used for official pipeline runs is:

- `C:\Users\priya\OneDrive\Desktop\rps-experiment\Takeout\Google Photos\rps-data-for-comments`

## 2. Data Collection Process

The dataset was sourced from two cohorts of the university's BTech 2023 batch. Participation was incentivized through marks awarded for both data contribution and labeling contribution.

The collection and annotation process was conducted in two Google Photos stages:

- initial upload album for raw data collection:
  [Collection album](https://photos.google.com/share/AF1QipNVLPu0ha-7MIEFx1vpxaE4u9k1HuOkRywodri5KdrBC8D4l5y_4FLGey8dlELHnw?key=YXV2RkFjR0NxT01DdEppQ0NWTzl5eWMyRjlsdDFn)
- secondary album used exclusively for comment-based labeling:
  [Commenting album](https://photos.google.com/share/AF1QipNnXlMMFz-_97AuyftpwSrfgeSYFpIMGu9Orw6g55futZi34QQnPuUI4bwYTaNhgQ?pli=1&key=M21XcndvQVlRRVhtd1dNM1VmSWw1dDhqbkk2MDN3)

The labeled album was then exported locally through Google Takeout. The Takeout export created a complete local copy of the dataset, including image files and comment metadata.

## 3. Structure of the Exported Dataset

The full Takeout dataset contains:

- `4820` supplemental metadata files used by the labeling pipeline
- `4683` image files

Current file-type counts in the canonical dataset folder:

- `.jpg`: `4346`
- `.heic`: `232`
- `.dng`: `59`
- `.png`: `4`
- `.mp4`: `42`

This indicates that the dataset is heterogeneous in capture source and file type. The inclusion of mobile-device formats such as HEIC and DNG reflects the real-world contribution process and increased the preprocessing requirements of the project.

## 4. Annotation Source and Metadata

The project uses the Google Photos comment stream as the raw annotation source. Each `.supplemental-metadata.json` file contains:

- image filename
- creation metadata
- image-level metadata
- comment text
- commenter identity
- comment timestamps

This meant the project relied on natural-language labels rather than labels created through a rigid annotation platform.

## 5. Comment Cleaning and Canonical Labeling

The comment-processing pipeline is implemented in `process_rps_comments.py`.

### 5.1 Basic normalization

Basic normalization:

- trims surrounding whitespace
- lowercases text
- collapses repeated spaces

### 5.2 Advanced normalization

Advanced normalization:

- applies Unicode normalization
- removes punctuation and decorative symbols
- strips emojis and non-text noise
- preserves alphabetic tokens and spaces

### 5.3 Canonicalization

Normalized comments are mapped into:

- `rock`
- `paper`
- `scissors`
- `noise`
- `ambiguous`

The canonicalization logic uses:

- direct exact matching
- synonym mapping
- token-based resolution
- conservative typo correction through Levenshtein distance
- limited Hamming and Soundex-style fallback handling

The official full-dataset rerun produced the following comment-level results:

- total comments processed: `5565`
- unique raw comment variants: `117`
- unique variants after basic normalization: `78`
- unique variants after advanced normalization: `61`

Final canonical counts:

- `rock`: `1530`
- `paper`: `1526`
- `scissors`: `1561`
- `noise`: `947`
- `ambiguous`: `1`

Method-level outcomes:

- whole-string match: `4551`
- unresolved noise: `944`
- Levenshtein correction: `58`
- exact or synonym token match: `6`
- empty after cleaning: `3`
- Soundex fallback: `2`
- multiple label tokens: `1`

## 6. Image-Level Label Assignment

Since multiple participants could comment on the same image, the project assigns labels at image level rather than trusting a single comment.

For each image, the pipeline computes:

- class vote counts
- noise count
- ambiguous count
- majority label
- majority vote count
- majority ratio

The final decision policy is:

- assign `review` if there are no valid comments
- assign the majority class if the majority ratio exceeds the threshold and there are no ambiguous comments
- otherwise assign `review`

The selected threshold for official runs is `0.6`, chosen after comparison against `0.5` and `0.75`.

On the full dataset, threshold comparison yielded:

- threshold `0.5`: `488` review cases
- threshold `0.6`: `495` review cases
- threshold `0.75`: `500` review cases

This confirms that `0.6` remains a reasonable compromise between coverage and conservatism on the larger dataset as well.

## 7. Full-Dataset Image-Level Results

The full rerun produced `2925` assessed image entries in the image-level summary.

Final image-level outcomes at threshold `0.6` were:

- `scissors`: `850`
- `rock`: `792`
- `paper`: `788`
- `review`: `495`

Thus, `2430` image entries were automatically assigned a final class label, while `495` were routed to review.

Review reasons were:

- `no_valid_comments`: `485`
- `low_majority_ratio`: `9`
- `ambiguous_comments_present`: `1`

The majority ratio among accepted non-review images ranged from `0.6667` to `1.0`, meaning even the weakest accepted cases still had a two-thirds class majority.

## 8. Image Preprocessing and Handoff

The model-ready export pipeline is implemented in `prepare_rps_ml_dataset.py`.

It:

- reruns comment cleaning against the canonical dataset path
- filters to automatically labeled images
- excludes `review` images by default
- corrects EXIF orientation
- converts images to RGB
- resizes while preserving aspect ratio
- pads to a square canvas
- exports manifests and config metadata

Default preprocessing settings:

- threshold: `0.6`
- image size: `128 x 128`
- output format: `jpg`
- resample filter: `lanczos`

Using the full dataset at threshold `0.6`, the base model-ready export produced:

- `1985` successfully exported labeled images
- `445` skipped images
- `495` review images listed separately

Split counts in the base export were:

- train: `1588` images
- validation: `198` images
- test: `199` images

Class-specific split counts were:

- train: `515 paper`, `517 rock`, `556 scissors`
- validation: `64 paper`, `65 rock`, `69 scissors`
- test: `64 paper`, `65 rock`, `70 scissors`

The skipped-image reasons were:

- `423` missing source-image matches
- `22` image open or decode failures

This is an important observation: although image-level labeling succeeded for 2430 entries, not all of those entries could be exported into the image-preprocessing handoff package because some Takeout metadata entries did not resolve to directly usable local image files.

## 9. Train-Only Augmentation Strategy

To better match likely inference conditions, the project now supports horizontal-flip augmentation for the training split only.

This design was chosen because:

- contributors may have captured images from front-facing cameras
- deployment may use a different camera orientation
- training-only augmentation improves robustness
- applying augmentation to validation or test data would risk evaluation leakage

The augmentation implementation:

- is optional
- applies only after train/val/test split assignment
- creates actual flipped files for the training split
- leaves validation and test images unchanged
- records augmentation metadata in manifests and config files

In the full augmented export:

- base training images: `1588`
- flipped training images added: `1588`
- final training manifest rows: `3176`

Validation and test splits remained unaugmented.

## 10. Challenges Encountered

Several challenges shaped the data stage:

### 9.1 Informal annotation medium

Google Photos comments made participation easy, but label quality was less controlled than it would have been in a dedicated annotation tool. This introduced typos, synonyms, inconsistent formatting, and irrelevant comments.

### 9.2 Incentivized participation quality variance

Although marks successfully encouraged participation, the quality of contributed comments varied. This required conservative normalization and explicit review handling.

### 9.3 Multi-annotator aggregation

The project had to reconcile multiple comments per image into a single final label. This required a defensible majority-vote policy and threshold selection procedure.

### 9.4 Mixed media formats

The dataset includes multiple image formats and even some videos. This increased preprocessing complexity and required robust image loading and normalization.

### 9.5 Source-of-truth confusion

An earlier working subset had been used in the repository, while the full dataset resided in the complete Takeout export. This created a documentation and reporting risk, since statistics from the subset could be mistaken for full-dataset statistics. The canonical dataset path has therefore been corrected to the full Takeout dataset.

### 9.6 Review-case handling

Some images inevitably lacked usable comments. The project addressed this by separating unresolved items into a review queue rather than forcing low-confidence labels into the training data.

### 9.7 Metadata-to-image mismatches

The full Takeout export introduced another practical difficulty: some metadata entries did not map cleanly to usable image files in the preprocessing stage. This resulted in a sizeable number of skipped exports, primarily due to missing source-image matches, and required explicit reporting in the model-ready handoff artifacts.

## 11. Conclusion

The data stage of this project successfully transformed a collaboratively sourced and loosely annotated collection of images into a structured machine-learning dataset pipeline. The result is not merely a folder of images, but a reproducible workflow that supports:

- annotation cleaning
- image-level label aggregation
- review handling
- image preprocessing
- split generation
- train-only augmentation
- model-team handoff packaging

This makes the dataset substantially more reliable, auditable, and reusable for downstream model development.
