import os
import re
import json
import math
import argparse
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    Image = None


VALID_LABELS = {"rock", "paper", "scissors"}
SYNONYM_MAP = {
    "rock": "rock",
    "stone": "rock",
    "paper": "paper",
    "scissor": "scissors",
    "scissors": "scissors",
}

# Optional phonetic-style fallback ideas can be added later.
# For now, keep the project conservative and explainable.


def normalize_basic(comment: str) -> str:
    """
    Basic normalization:
    - strip surrounding whitespace
    - lowercase
    - collapse repeated internal whitespace
    """
    if not comment:
        return ""
    c = comment.strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def normalize_unicode(comment: str) -> str:
    """
    Unicode normalization:
    - NFKC handles compatibility forms more robustly
    """
    if not comment:
        return ""
    return unicodedata.normalize("NFKC", comment)


def remove_non_text_noise(comment: str) -> str:
    """
    Remove emojis / punctuation / decorative symbols while preserving letters and spaces.
    Keep only alphabetic characters and spaces after Unicode normalization.
    """
    if not comment:
        return ""

    # Normalize first
    c = normalize_unicode(comment)

    # Replace punctuation/symbols with spaces
    cleaned_chars = []
    for ch in c:
        category = unicodedata.category(ch)

        # Keep letters and spaces
        if ch.isalpha() or ch.isspace():
            cleaned_chars.append(ch)
        # Everything else becomes a space so words don't accidentally join
        else:
            cleaned_chars.append(" ")

    c = "".join(cleaned_chars)
    c = c.lower()
    c = re.sub(r"\s+", " ", c).strip()
    return c


def normalize_advanced(comment: str) -> str:
    """
    Advanced normalization pipeline:
    - Unicode normalization
    - remove emojis/punctuation/rubbish symbols
    - lowercase
    - collapse spaces
    """
    return remove_non_text_noise(comment)


def hamming_distance(a: str, b: str):
    """
    Only defined for equal-length strings.
    Included because it was mentioned in class.
    """
    if len(a) != len(b):
        return None
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def levenshtein_distance(a: str, b: str) -> int:
    """
    Simple DP implementation for small vocabulary.
    Avoids adding more dependencies.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost  # replace
            )

    return dp[-1][-1]


def soundex(word: str) -> str:
    """
    Simple Soundex implementation for experimentation/reporting.
    Kept lightweight and explainable.
    """
    if not word:
        return ""

    word = word.upper()
    first_letter = word[0]

    mappings = {
        **dict.fromkeys(list("BFPV"), "1"),
        **dict.fromkeys(list("CGJKQSXZ"), "2"),
        **dict.fromkeys(list("DT"), "3"),
        **dict.fromkeys(list("L"), "4"),
        **dict.fromkeys(list("MN"), "5"),
        **dict.fromkeys(list("R"), "6"),
    }

    encoded = []
    prev = mappings.get(first_letter, "")

    for ch in word[1:]:
        code = mappings.get(ch, "")
        if code != prev:
            if code != "":
                encoded.append(code)
        prev = code

    result = first_letter + "".join(encoded)
    result = (result + "000")[:4]
    return result


def typo_match_token(token: str, use_soundex: bool = True):
    """
    Conservative typo correction:
    1. exact synonym map
    2. nearest valid/synonym token by edit distance
    3. optional Soundex fallback
    """
    if not token:
        return None, None

    if token in SYNONYM_MAP:
        return SYNONYM_MAP[token], "exact_or_synonym"

    candidates = list(SYNONYM_MAP.keys())

    # Edit-distance based nearest match
    best_token = None
    best_dist = math.inf

    for cand in candidates:
        d = levenshtein_distance(token, cand)
        if d < best_dist:
            best_dist = d
            best_token = cand

    # Conservative thresholds for tiny labels
    if best_token is not None:
        if len(token) <= 5 and best_dist <= 1:
            return SYNONYM_MAP[best_token], "levenshtein"
        if len(token) > 5 and best_dist <= 2:
            return SYNONYM_MAP[best_token], "levenshtein"

    # Hamming is limited but we include it as a tracked experiment
    same_length_candidates = [cand for cand in candidates if len(cand) == len(token)]
    best_hamming = math.inf
    best_hamming_token = None
    for cand in same_length_candidates:
        hd = hamming_distance(token, cand)
        if hd is not None and hd < best_hamming:
            best_hamming = hd
            best_hamming_token = cand

    if best_hamming_token is not None and best_hamming <= 1:
        return SYNONYM_MAP[best_hamming_token], "hamming"

    if use_soundex:
        token_sx = soundex(token)
        for cand in candidates:
            if soundex(cand) == token_sx:
                return SYNONYM_MAP[cand], "soundex"

    return None, None


def canonicalize_comment(comment: str):
    """
    Returns:
    {
        'basic': ...,
        'advanced': ...,
        'canonical_label': ...,
        'method': ...
    }
    """
    basic = normalize_basic(comment)
    advanced = normalize_advanced(comment)

    if not advanced:
        return {
            "basic": basic,
            "advanced": advanced,
            "canonical_label": "noise",
            "method": "empty_after_cleaning",
        }

    # First, direct whole-string match
    if advanced in SYNONYM_MAP:
        return {
            "basic": basic,
            "advanced": advanced,
            "canonical_label": SYNONYM_MAP[advanced],
            "method": "whole_string_match",
        }

    tokens = advanced.split()
    if not tokens:
        return {
            "basic": basic,
            "advanced": advanced,
            "canonical_label": "noise",
            "method": "no_tokens",
        }

    matched_labels = []
    matched_methods = []

    for token in tokens:
        label, method = typo_match_token(token)
        if label:
            matched_labels.append(label)
            matched_methods.append(method)

    matched_set = set(matched_labels)

    if len(matched_set) == 1:
        return {
            "basic": basic,
            "advanced": advanced,
            "canonical_label": list(matched_set)[0],
            "method": "+".join(sorted(set(matched_methods))) if matched_methods else "token_match",
        }

    if len(matched_set) > 1:
        return {
            "basic": basic,
            "advanced": advanced,
            "canonical_label": "ambiguous",
            "method": "multiple_label_tokens",
        }

    return {
        "basic": basic,
        "advanced": advanced,
        "canonical_label": "noise",
        "method": "unresolved_noise",
    }


def extract_metadata_rows(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_title = data.get("title", "")
    image_views = data.get("imageViews", "")
    created_formatted = data.get("creationTime", {}).get("formatted", "")
    comments = data.get("sharedAlbumComments", [])

    rows = []

    for entry in comments:
        raw_comment = entry.get("text", "")
        owner = entry.get("contentOwnerName", "")
        comment_time = entry.get("creationTime", {}).get("formatted", "")

        if not raw_comment or not raw_comment.strip():
            continue

        norm = canonicalize_comment(raw_comment)

        rows.append({
            "metadata_file": os.path.basename(json_path),
            "image_file": image_title,
            "image_views": image_views,
            "image_created_time": created_formatted,
            "comment_owner": owner,
            "comment_time": comment_time,
            "raw_comment": raw_comment,
            "basic_normalized": norm["basic"],
            "advanced_normalized": norm["advanced"],
            "canonical_label": norm["canonical_label"],
            "normalization_method": norm["method"],
        })

    return rows


def get_image_info(folder_path: str, image_file: str):
    image_path = os.path.join(folder_path, image_file)
    exists = os.path.exists(image_path)

    info = {
        "image_exists": exists,
        "image_width": None,
        "image_height": None,
        "image_mode": None,
    }

    if exists and Image is not None:
        try:
            with Image.open(image_path) as img:
                info["image_width"] = img.width
                info["image_height"] = img.height
                info["image_mode"] = img.mode
        except Exception:
            info["image_exists"] = False

    return info


def save_counter_to_csv(counter_obj: Counter, output_csv: str, column_name="comment"):
    df = pd.DataFrame(counter_obj.items(), columns=[column_name, "count"])
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(output_csv, index=False)


def plot_counter(counter_obj: Counter, title: str, xlabel: str, output_png: str, top_n=None):
    items = counter_obj.most_common(top_n) if top_n else counter_obj.most_common()
    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def plot_majority_ratio_histogram(series: pd.Series, output_png: str):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=10)
    plt.xlabel("Majority Ratio")
    plt.ylabel("Number of Images")
    plt.title("Majority Ratio Distribution")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def summarize_image_labels(df_comments: pd.DataFrame, folder_path: str, threshold: float = 0.6):
    summary_rows = []
    review_rows = []

    grouped = df_comments.groupby("image_file")

    for image_file, group in grouped:
        label_counts = Counter(group["canonical_label"])
        valid_counts = {
            "rock": label_counts.get("rock", 0),
            "paper": label_counts.get("paper", 0),
            "scissors": label_counts.get("scissors", 0),
        }

        valid_total = sum(valid_counts.values())
        noise_count = label_counts.get("noise", 0)
        ambiguous_count = label_counts.get("ambiguous", 0)

        if valid_total > 0:
            majority_label, majority_votes = max(valid_counts.items(), key=lambda x: x[1])
            majority_ratio = majority_votes / valid_total
        else:
            majority_label, majority_votes, majority_ratio = None, 0, None

        if valid_total == 0:
            final_label = "review"
            review_reason = "no_valid_comments"
        elif majority_ratio is not None and majority_ratio >= threshold and ambiguous_count == 0:
            final_label = majority_label
            review_reason = ""
        else:
            final_label = "review"
            if ambiguous_count > 0:
                review_reason = "ambiguous_comments_present"
            else:
                review_reason = "low_majority_ratio"

        image_info = get_image_info(folder_path, image_file)

        row = {
            "image_file": image_file,
            "total_comments": len(group),
            "valid_comment_count": valid_total,
            "rock_votes": valid_counts["rock"],
            "paper_votes": valid_counts["paper"],
            "scissors_votes": valid_counts["scissors"],
            "noise_count": noise_count,
            "ambiguous_count": ambiguous_count,
            "majority_label": majority_label,
            "majority_votes": majority_votes,
            "majority_ratio": majority_ratio,
            "final_label": final_label,
            "review_reason": review_reason,
            **image_info,
        }
        summary_rows.append(row)

        if final_label == "review":
            review_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(review_rows)


def build_normalization_stats(df_comments: pd.DataFrame):
    stats = []

    stats.append({
        "metric": "total_comments",
        "value": len(df_comments),
    })
    stats.append({
        "metric": "unique_raw_comments",
        "value": df_comments["raw_comment"].nunique(),
    })
    stats.append({
        "metric": "unique_basic_normalized",
        "value": df_comments["basic_normalized"].nunique(),
    })
    stats.append({
        "metric": "unique_advanced_normalized",
        "value": df_comments["advanced_normalized"].nunique(),
    })
    stats.append({
        "metric": "canonical_rock",
        "value": int((df_comments["canonical_label"] == "rock").sum()),
    })
    stats.append({
        "metric": "canonical_paper",
        "value": int((df_comments["canonical_label"] == "paper").sum()),
    })
    stats.append({
        "metric": "canonical_scissors",
        "value": int((df_comments["canonical_label"] == "scissors").sum()),
    })
    stats.append({
        "metric": "canonical_noise",
        "value": int((df_comments["canonical_label"] == "noise").sum()),
    })
    stats.append({
        "metric": "canonical_ambiguous",
        "value": int((df_comments["canonical_label"] == "ambiguous").sum()),
    })

    method_counts = df_comments["normalization_method"].value_counts()
    for method, count in method_counts.items():
        stats.append({
            "metric": f"method_{method}",
            "value": int(count),
        })

    return pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser(description="Bulk process RPS metadata JSON files.")
    parser.add_argument("folder", type=str, help="Folder containing JSON metadata and images")
    parser.add_argument("--threshold", type=float, default=0.6, help="Majority vote threshold")
    args = parser.parse_args()

    folder_path = args.folder
    threshold = args.threshold

    json_files = sorted(
        f for f in os.listdir(folder_path)
        if f.endswith(".supplemental-metadata.json")
    )

    print(f"Found {len(json_files)} metadata files.")

    all_rows = []
    for file_name in json_files:
        json_path = os.path.join(folder_path, file_name)
        rows = extract_metadata_rows(json_path)
        all_rows.extend(rows)

    if not all_rows:
        print("No comments found.")
        return

    df_comments = pd.DataFrame(all_rows)

    os.makedirs("output", exist_ok=True)

    # Save comment-level audit
    df_comments.to_csv("output/comment_level_audit.csv", index=False)

    # Counters
    raw_counts = Counter(df_comments["raw_comment"])
    basic_counts = Counter(df_comments["basic_normalized"])
    advanced_counts = Counter(df_comments["advanced_normalized"])
    canonical_counts = Counter(df_comments["canonical_label"])

    save_counter_to_csv(raw_counts, "output/raw_comment_counts.csv", "comment")
    save_counter_to_csv(basic_counts, "output/basic_normalized_counts.csv", "comment")
    save_counter_to_csv(advanced_counts, "output/advanced_normalized_counts.csv", "comment")
    save_counter_to_csv(canonical_counts, "output/canonical_label_counts.csv", "label")

    # Image-level summary
    df_image_summary, df_review = summarize_image_labels(df_comments, folder_path, threshold=threshold)
    df_image_summary.to_csv("output/image_label_summary.csv", index=False)
    df_review.to_csv("output/review_queue.csv", index=False)

    # Stats
    df_stats = build_normalization_stats(df_comments)
    df_stats.to_csv("output/normalization_stats.csv", index=False)

    # Plots
    plot_counter(raw_counts, "Raw Comment Histogram", "Raw Comments", "output/raw_hist.png", top_n=20)
    plot_counter(basic_counts, "Basic Normalized Histogram", "Normalized Comments", "output/basic_norm_hist.png", top_n=20)
    plot_counter(advanced_counts, "Advanced Normalized Histogram", "Advanced Normalized Comments", "output/advanced_norm_hist.png", top_n=20)
    plot_counter(canonical_counts, "Canonical Label Histogram", "Canonical Labels", "output/canonical_hist.png")

    if "majority_ratio" in df_image_summary.columns and not df_image_summary.empty:
        plot_majority_ratio_histogram(df_image_summary["majority_ratio"], "output/majority_ratio_hist.png")

    print("\n=== FINAL CANONICAL COUNTS ===")
    for label, count in canonical_counts.most_common():
        print(f"{label}: {count}")

    print("\n=== IMAGE LABEL SUMMARY ===")
    print(df_image_summary["final_label"].value_counts(dropna=False))

    print("\nDone. Outputs saved in /output")


if __name__ == "__main__":
    main()