import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from process_rps_comments import canonicalize_comment


MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".dng", ".mp4"}


def classify_origin(origin_value):
    if isinstance(origin_value, dict) and origin_value:
        return ",".join(sorted(origin_value.keys()))
    return "unknown"


def collect_commenter_stats(dataset_dir: Path):
    contributor_rows = []
    origin_counter = Counter()
    uploader_name_keys = Counter()
    total_json = 0

    for path in dataset_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        total_json += 1
        origin_counter[classify_origin(data.get("googlePhotosOrigin"))] += 1

        comments = data.get("sharedAlbumComments") or []
        image_title = data.get("title", "")

        for comment in comments:
            owner = (comment.get("contentOwnerName") or "").strip()
            if not owner:
                owner = "UNKNOWN_COMMENTER"

            raw_text = comment.get("text", "")
            normalized = canonicalize_comment(raw_text)
            contributor_rows.append(
                {
                    "comment_owner": owner,
                    "image_file": image_title,
                    "raw_comment": raw_text,
                    "canonical_label": normalized["canonical_label"],
                    "normalization_method": normalized["method"],
                    "comment_timestamp": (comment.get("creationTime") or {}).get("timestamp", ""),
                }
            )

        stack = [("", data)]
        while stack:
            prefix, obj = stack.pop()
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    lowered = key.lower()
                    if any(term in lowered for term in ["uploader", "contributor", "author", "creator", "owner"]):
                        if full_key != "sharedAlbumComments.contentOwnerName":
                            uploader_name_keys[full_key] += 1
                    stack.append((full_key, value))
            elif isinstance(obj, list):
                for value in obj[:5]:
                    stack.append((f"{prefix}[]", value))

    comments_df = pd.DataFrame(contributor_rows)
    if comments_df.empty:
        stats_df = pd.DataFrame(
            columns=[
                "comment_owner",
                "total_comments",
                "unique_images_labeled",
                "valid_label_comments",
                "noise_comments",
                "ambiguous_comments",
                "rock_comments",
                "paper_comments",
                "scissors_comments",
                "valid_comment_rate",
            ]
        )
    else:
        stats_df = (
            comments_df.groupby("comment_owner")
            .agg(
                total_comments=("raw_comment", "size"),
                unique_images_labeled=("image_file", "nunique"),
                valid_label_comments=("canonical_label", lambda s: int(s.isin(["rock", "paper", "scissors"]).sum())),
                noise_comments=("canonical_label", lambda s: int((s == "noise").sum())),
                ambiguous_comments=("canonical_label", lambda s: int((s == "ambiguous").sum())),
                rock_comments=("canonical_label", lambda s: int((s == "rock").sum())),
                paper_comments=("canonical_label", lambda s: int((s == "paper").sum())),
                scissors_comments=("canonical_label", lambda s: int((s == "scissors").sum())),
            )
            .reset_index()
        )
        stats_df["valid_comment_rate"] = (stats_df["valid_label_comments"] / stats_df["total_comments"]).round(4)
        stats_df = stats_df.sort_values(
            ["valid_label_comments", "unique_images_labeled", "total_comments", "comment_owner"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "total_metadata_json_files_scanned": total_json,
        "origin_type_counts": dict(origin_counter),
        "collection_uploader_name_available": False,
        "collection_uploader_name_reason": (
            "Current Takeout item metadata exposes upload origin types such as webUpload/mobileUpload/"
            "fromSharedAlbum, but no per-image uploader person name field was found."
        ),
        "owner_like_non_comment_keys_found": dict(uploader_name_keys),
        "comment_contributor_count": int(len(stats_df)),
        "total_comment_rows": int(len(comments_df)),
    }

    return comments_df, stats_df, summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate contributor statistics from Google Photos Takeout metadata."
    )
    parser.add_argument("dataset_dir", type=str, help="Path to the Takeout album folder")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/contributor_stats/current",
        help="Directory to write contributor statistics",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comments_df, stats_df, summary = collect_commenter_stats(dataset_dir)

    comments_df.to_csv(output_dir / "comment_contribution_audit.csv", index=False)
    stats_df.to_csv(output_dir / "comment_contributor_stats.csv", index=False)
    (output_dir / "contributor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote comment contribution audit: {(output_dir / 'comment_contribution_audit.csv').resolve()}")
    print(f"Wrote comment contributor stats: {(output_dir / 'comment_contributor_stats.csv').resolve()}")
    print(f"Wrote summary JSON: {(output_dir / 'contributor_summary.json').resolve()}")
    print()
    print("Collection uploader name available:", summary["collection_uploader_name_available"])
    print(summary["collection_uploader_name_reason"])
    if not stats_df.empty:
        print()
        print("Top 10 comment contributors:")
        print(stats_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
