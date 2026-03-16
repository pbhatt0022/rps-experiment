import os
import json
import argparse
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


def normalize_basic(comment: str) -> str:
    return comment.strip().lower()


def canonicalize_label(comment: str) -> str:
    c = comment.strip().lower()

    if c in ["rock", "stone"]:
        return "rock"
    elif c in ["paper"]:
        return "paper"
    elif c in ["scissor", "scissors"]:
        return "scissors"
    else:
        return "noise"


def extract_comments_from_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    comments = []
    for entry in data.get("sharedAlbumComments", []):
        text = entry.get("text", "")
        if text and text.strip():
            comments.append(text)

    return comments


def save_counter_to_csv(counter_obj: Counter, output_csv: str):
    df = pd.DataFrame(counter_obj.items(), columns=["comment", "count"])
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(output_csv, index=False)


def plot_counter(counter_obj: Counter, title: str, xlabel: str, output_png: str):
    labels = list(counter_obj.keys())
    counts = list(counter_obj.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Process RPS metadata JSON files.")
    parser.add_argument("folder", type=str, help="Folder path containing JSON files")
    args = parser.parse_args()

    folder_path = args.folder

    raw_comments = []
    basic_norm_comments = []
    canonical_labels = []

    json_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".supplemental-metadata.json")
    ]

    print(f"Found {len(json_files)} metadata files.")

    for file_name in json_files:
        json_path = os.path.join(folder_path, file_name)
        comments = extract_comments_from_file(json_path)

        for comment in comments:
            raw_comments.append(comment)
            basic_norm_comments.append(normalize_basic(comment))
            canonical_labels.append(canonicalize_label(comment))

    raw_counts = Counter(raw_comments)
    basic_norm_counts = Counter(basic_norm_comments)
    canonical_counts = Counter(canonical_labels)

    os.makedirs("output", exist_ok=True)

    save_counter_to_csv(raw_counts, "output/raw_comment_counts.csv")
    save_counter_to_csv(basic_norm_counts, "output/basic_normalized_counts.csv")
    save_counter_to_csv(canonical_counts, "output/canonical_label_counts.csv")

    plot_counter(raw_counts, "Raw Comment Histogram", "Raw Comments", "output/raw_hist.png")
    plot_counter(basic_norm_counts, "Basic Normalized Histogram", "Normalized Comments", "output/basic_norm_hist.png")
    plot_counter(canonical_counts, "Canonical Label Histogram", "Labels", "output/canonical_hist.png")

    print("\n=== FINAL COUNTS ===")
    for label, count in canonical_counts.most_common():
        print(f"{label}: {count}")

    print("\nDone. Outputs saved in /output")


if __name__ == "__main__":
    main()