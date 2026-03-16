\# Rock Paper Scissors Comment Processing Pipeline



\## Overview



This project processes free-text human comments collected for a Rock-Paper-Scissors image-labeling task and converts them into clean, canonical labels suitable for downstream machine learning workflows.



The main challenge in this dataset is that labels were written as natural comments by multiple people, which introduced inconsistency in:



\- capitalization

\- spacing

\- singular/plural forms

\- synonym usage

\- emojis and decorative symbols

\- spelling mistakes

\- irrelevant or noisy text



To address this, the project implements a staged normalization and canonicalization pipeline, along with image-level majority-vote labeling and threshold-based review logic.



---



\## Project Files



\- `process\_rps\_comments.py`  

&nbsp; Main bulk-processing pipeline for metadata JSON files and associated images.



\- `tests\_simulated\_comments.py`  

&nbsp; Simulated test cases used to validate normalization robustness on controlled edge cases.



\- `requirements.txt`  

&nbsp; Python dependencies.



\- `output/`  

&nbsp; Final chosen results using threshold `0.6`.



\- `output\_t05/`, `output\_t06/`, `output\_t075/`  

&nbsp; Threshold comparison outputs for experimental analysis.



---



\## Processing Workflow



\### 1. Metadata Extraction



The script reads all `.supplemental-metadata.json` files in a dataset folder and extracts:



\- image filename

\- image metadata

\- comment text

\- comment owner

\- comment timestamp



\### 2. Basic Normalization



Basic normalization performs:



\- leading/trailing whitespace removal

\- lowercase conversion

\- collapsing repeated spaces



\*\*Example:\*\*



\- `" Rock "` → `"rock"`



\### 3. Advanced Normalization



Advanced normalization performs:



\- Unicode normalization using NFKC

\- emoji and punctuation removal

\- non-text symbol cleanup

\- whitespace cleanup



\*\*Examples:\*\*



\- `"rock 🪨"` → `"rock"`

\- `"Paper 📄"` → `"paper"`



\### 4. Canonicalization



After normalization, comments are mapped into a final label space using:



\- exact matching

\- synonym mapping

\- token-based phrase handling

\- typo correction using Levenshtein distance

\- optional Hamming/Soundex experimentation for class-inspired extensions



\*\*Final canonical labels:\*\*



\- `rock`

\- `paper`

\- `scissors`

\- `noise`

\- `ambiguous`



\*\*Examples:\*\*



\- `"stone"` → `rock`

\- `"papet"` → `paper`

\- `"sissors"` → `scissors`

\- `"rock paper"` → `ambiguous`

\- `"phone"` → `noise`



\### 5. Image-Level Label Assignment



Each image may have multiple comments. Final image labels are assigned by:



\- counting valid canonical labels

\- selecting the majority label

\- computing a majority ratio

\- sending uncertain/noisy cases to a review queue



---



\## Threshold Logic



Three thresholds were compared for image-level labeling:



\- `0.5`

\- `0.6`

\- `0.75`



\### Chosen Threshold: `0.6`



\*\*Reason:\*\*



\- `0.5` and `0.6` produced identical automatic labeling coverage

\- `0.75` introduced one extra review case due to stricter confidence requirements

\- `0.6` therefore gave the best balance between coverage and conservatism



---



\## Results



\### Comment-Level Results



\- Total comments processed: \*\*380\*\*

\- Unique raw comment variants: \*\*34\*\*

\- Unique variants after basic normalization: \*\*21\*\*

\- Unique variants after advanced normalization: \*\*16\*\*



\### Final Canonical Counts



\- `rock`: \*\*141\*\*

\- `paper`: \*\*127\*\*

\- `scissors`: \*\*107\*\*

\- `noise`: \*\*5\*\*



This means \*\*375 out of 380 comments\*\* were successfully mapped into valid training labels.



\### Simulated Test Results



A simulated comment test suite covering:



\- capitalization

\- whitespace

\- emojis

\- typos

\- synonyms

\- ambiguous multi-label phrases

\- irrelevant noise



achieved:



\- \*\*18 / 18 tests passed\*\*



\### Threshold Comparison



\- Threshold `0.5`: 3 review cases

\- Threshold `0.6`: 3 review cases

\- Threshold `0.75`: 4 review cases



At `0.75`, one additional image with a 2/3 majority vote was pushed into review, showing that `0.75` was overly conservative for this dataset.



---



\## Review Queue Interpretation



The remaining review cases fall into two categories:



1\. \*\*No valid comments\*\* after preprocessing

2\. \*\*Low majority ratio\*\* under stricter thresholds



For the final chosen threshold `0.6`, the review queue contains only images with no valid comments, indicating that the unresolved cases are due to unusable annotations rather than failure of the normalization pipeline.



---



\## Key Output Files



\### Main Final Outputs (`output/`)



\- `comment\_level\_audit.csv`  

&nbsp; Full comment-level audit trail with raw text, normalized text, canonical label, and method used.



\- `raw\_comment\_counts.csv`  

&nbsp; Raw comment frequency counts.



\- `basic\_normalized\_counts.csv`  

&nbsp; Frequency counts after basic normalization.



\- `advanced\_normalized\_counts.csv`  

&nbsp; Frequency counts after advanced normalization.



\- `canonical\_label\_counts.csv`  

&nbsp; Final canonical label counts.



\- `image\_label\_summary.csv`  

&nbsp; Final image-level labels with vote counts and confidence ratios.



\- `review\_queue.csv`  

&nbsp; Images requiring manual inspection.



\- `normalization\_stats.csv`  

&nbsp; Quantitative summary of the normalization pipeline.



\### Plots



\- `raw\_hist.png`

\- `basic\_norm\_hist.png`

\- `advanced\_norm\_hist.png`

\- `canonical\_hist.png`

\- `majority\_ratio\_hist.png`



---



\## How to Run



```bash

\# Install dependencies

pip install -r requirements.txt



\# Run the main pipeline

python process\_rps\_comments.py C:\\Users\\priya\\OneDrive\\Desktop\\rps-experiment\\rps-data-for-comments --threshold 0.6



\# Run simulated tests

python tests\_simulated\_comments.py



\# Run threshold comparison experiments

python process\_rps\_comments.py C:\\Users\\priya\\OneDrive\\Desktop\\rps-experiment\\rps-data-for-comments --threshold 0.5

python process\_rps\_comments.py C:\\Users\\priya\\OneDrive\\Desktop\\rps-experiment\\rps-data-for-comments --threshold 0.6

python process\_rps\_comments.py C:\\Users\\priya\\OneDrive\\Desktop\\rps-experiment\\rps-data-for-comments --threshold 0.75

