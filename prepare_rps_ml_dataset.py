import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from process_rps_comments import VALID_LABELS, run_comment_pipeline


RESAMPLE_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def parse_rgb_triplet(value: str):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pad color must be three comma-separated integers, e.g. 0,0,0")

    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pad color values must be integers") from exc

    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("pad color values must be between 0 and 255")

    return rgb


def resize_and_pad(image: Image.Image, size: int, fill_color: tuple[int, int, int], resample_name: str):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size

    scale = size / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = image.resize((new_width, new_height), RESAMPLE_MAP[resample_name])
    canvas = Image.new("RGB", (size, size), color=fill_color)

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    canvas.paste(resized, (x_offset, y_offset))
    return canvas, width, height


def build_processed_filename(image_name: str, output_format: str):
    source = Path(image_name)
    stem = source.stem.replace(" ", "_")
    digest = hashlib.md5(source.name.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}.{output_format}"


def build_augmented_filename(processed_image_path: str, augmentation_suffix: str):
    source = Path(processed_image_path)
    return f"{source.stem}_{augmentation_suffix}{source.suffix}"


def parse_split_ratios(value: str):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split ratios must be three comma-separated values, e.g. 0.8,0.1,0.1")

    try:
        ratios = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split ratios must be numeric") from exc

    if any(ratio < 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("split ratios cannot be negative")

    total = sum(ratios)
    if total <= 0:
        raise argparse.ArgumentTypeError("split ratios must sum to a positive value")

    normalized = [ratio / total for ratio in ratios]
    return {
        "train": normalized[0],
        "val": normalized[1],
        "test": normalized[2],
    }


def assign_split_labels(manifest_df: pd.DataFrame, split_ratios: dict[str, float], seed: int):
    if manifest_df.empty:
        result = manifest_df.copy()
        result["split"] = pd.Series(dtype="object")
        return result

    rng = random.Random(seed)
    split_order = ["train", "val", "test"]
    split_rows = []

    for label, group in manifest_df.groupby("label", sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        total = len(indices)

        raw_counts = {split: split_ratios[split] * total for split in split_order}
        base_counts = {split: int(raw_counts[split]) for split in split_order}
        assigned = sum(base_counts.values())
        remainder = total - assigned

        remainders = sorted(
            split_order,
            key=lambda split: (raw_counts[split] - base_counts[split], split_order.index(split)),
            reverse=True,
        )
        for split in remainders[:remainder]:
            base_counts[split] += 1

        cursor = 0
        for split in split_order:
            split_size = base_counts[split]
            selected = indices[cursor:cursor + split_size]
            cursor += split_size
            for idx in selected:
                row = manifest_df.loc[idx].to_dict()
                row["split"] = split
                split_rows.append(row)

    result = pd.DataFrame(split_rows, columns=list(manifest_df.columns))
    return result.sort_values(["split", "label", "source_image_file"]).reset_index(drop=True)


def export_split_directories(manifest_df: pd.DataFrame, dataset_root: Path):
    split_root = dataset_root / "splits"
    for split_name in ["train", "val", "test"]:
        split_subset = manifest_df[manifest_df["split"] == split_name]
        for label in sorted(VALID_LABELS):
            (split_root / split_name / label).mkdir(parents=True, exist_ok=True)

        for row in split_subset.to_dict(orient="records"):
            source = Path(row["processed_image_path"])
            destination = split_root / split_name / row["label"] / source.name
            shutil.copy2(source, destination)
            row["split_image_path"] = str(destination.resolve())


def export_train_flip_augmentation(split_manifest_df: pd.DataFrame, dataset_root: Path):
    train_rows = split_manifest_df[split_manifest_df["split"] == "train"].copy()
    if train_rows.empty:
        return split_manifest_df, 0

    augmented_rows = []
    for row in train_rows.to_dict(orient="records"):
        source_path = Path(row["split_image_path"])
        destination = source_path.parent / build_augmented_filename(source_path.name, "flip")

        with Image.open(source_path) as image:
            flipped = ImageOps.mirror(image)
            save_kwargs = {}
            if source_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs.update({"quality": 95, "subsampling": 0})
            flipped.save(destination, **save_kwargs)

        augmented_row = dict(row)
        augmented_row["processed_image_path"] = str(destination.resolve())
        augmented_row["split_image_path"] = str(destination.resolve())
        augmented_row["is_augmented"] = True
        augmented_row["augmentation_type"] = "horizontal_flip"
        augmented_row["source_base_image"] = row["processed_image_path"]
        augmented_rows.append(augmented_row)

    if not augmented_rows:
        return split_manifest_df, 0

    augmented_df = pd.DataFrame(augmented_rows, columns=split_manifest_df.columns)
    combined = pd.concat([split_manifest_df, augmented_df], ignore_index=True)
    combined = combined.sort_values(["split", "label", "source_image_file", "is_augmented"]).reset_index(drop=True)
    return combined, len(augmented_rows)


def build_split_summary(manifest_df: pd.DataFrame):
    if manifest_df.empty:
        return {}

    summary = {}
    for split_name, split_group in manifest_df.groupby("split", sort=True):
        summary[split_name] = split_group["label"].value_counts().sort_index().to_dict()
    return summary


def write_split_manifests(manifest_df: pd.DataFrame, dataset_root: Path):
    for split_name in ["train", "val", "test"]:
        split_subset = manifest_df[manifest_df["split"] == split_name].copy()
        split_subset.to_csv(dataset_root / f"{split_name}_manifest.csv", index=False)


def create_zip_bundle(dataset_root: Path):
    archive_base = dataset_root.parent / dataset_root.name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=dataset_root)
    return archive_path


def prepare_dataset(
    dataset_folder: str,
    label_output_dir: str,
    dataset_output_dir: str,
    label_threshold: float,
    image_size: int,
    fill_color: tuple[int, int, int],
    output_format: str,
    jpeg_quality: int,
    resample_name: str,
    include_review: bool,
    create_splits: bool,
    split_ratios: dict[str, float],
    split_seed: int,
    augment_train_horizontal_flip: bool,
    zip_output: bool,
):
    manifest_columns = [
        "source_image_file",
        "source_image_path",
        "processed_image_path",
        "split_image_path",
        "label",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "valid_comment_count",
        "majority_label",
        "majority_votes",
        "majority_ratio",
        "noise_count",
        "ambiguous_count",
        "review_reason",
        "split",
        "is_augmented",
        "augmentation_type",
        "source_base_image",
    ]
    if augment_train_horizontal_flip and not create_splits:
        raise ValueError("train-only augmentation requires --create-splits")

    _, df_image_summary, _ = run_comment_pipeline(
        dataset_folder,
        threshold=label_threshold,
        output_dir=label_output_dir,
    )
    if df_image_summary is None or df_image_summary.empty:
        raise RuntimeError("No image labels were generated. Cannot prepare model dataset.")

    df_image_summary = df_image_summary.copy()
    if include_review:
        selected = df_image_summary[df_image_summary["final_label"].isin(list(VALID_LABELS) + ["review"])].copy()
    else:
        selected = df_image_summary[df_image_summary["final_label"].isin(VALID_LABELS)].copy()

    review_manifest = df_image_summary[df_image_summary["final_label"] == "review"].copy()

    dataset_root = Path(dataset_output_dir)
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    for label in sorted(VALID_LABELS):
        (images_root / label).mkdir(parents=True, exist_ok=True)
    if include_review:
        (images_root / "review").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    skipped_rows = []

    for row in selected.to_dict(orient="records"):
        label = row["final_label"]
        source_name = row["image_file"]
        source_path = Path(dataset_folder) / source_name

        if not source_path.exists():
            skipped_rows.append({
                "image_file": source_name,
                "reason": "missing_source_image",
            })
            continue

        try:
            with Image.open(source_path) as image:
                processed, original_width, original_height = resize_and_pad(
                    image,
                    size=image_size,
                    fill_color=fill_color,
                    resample_name=resample_name,
                )
        except Exception as exc:
            skipped_rows.append({
                "image_file": source_name,
                "reason": f"open_or_resize_failed:{type(exc).__name__}",
            })
            continue

        processed_name = build_processed_filename(source_name, output_format)
        output_path = images_root / label / processed_name
        save_kwargs = {}
        if output_format.lower() in {"jpg", "jpeg"}:
            save_kwargs.update({"quality": jpeg_quality, "subsampling": 0})
        processed.save(output_path, **save_kwargs)

        manifest_rows.append({
            "source_image_file": source_name,
            "source_image_path": str(source_path.resolve()),
            "processed_image_path": str(output_path.resolve()),
            "split_image_path": "",
            "label": label,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": image_size,
            "processed_height": image_size,
            "valid_comment_count": row["valid_comment_count"],
            "majority_label": row["majority_label"],
            "majority_votes": row["majority_votes"],
            "majority_ratio": row["majority_ratio"],
            "noise_count": row["noise_count"],
            "ambiguous_count": row["ambiguous_count"],
            "review_reason": row["review_reason"],
            "split": "",
            "is_augmented": False,
            "augmentation_type": "",
            "source_base_image": "",
        })

    manifest_df = pd.DataFrame(manifest_rows, columns=manifest_columns)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["label", "source_image_file"])

    split_manifest_df = pd.DataFrame(columns=manifest_columns)
    augmented_train_count = 0
    if create_splits:
        manifest_df = assign_split_labels(manifest_df, split_ratios=split_ratios, seed=split_seed)
        export_split_directories(manifest_df, dataset_root)
        split_rows = []
        for split_name in ["train", "val", "test"]:
            split_subset = manifest_df[manifest_df["split"] == split_name].copy()
            for row in split_subset.to_dict(orient="records"):
                split_image_path = dataset_root / "splits" / split_name / row["label"] / Path(row["processed_image_path"]).name
                row["split_image_path"] = str(split_image_path.resolve())
                split_rows.append(row)
        split_manifest_df = pd.DataFrame(split_rows, columns=manifest_columns)
        if augment_train_horizontal_flip:
            split_manifest_df, augmented_train_count = export_train_flip_augmentation(split_manifest_df, dataset_root)
        write_split_manifests(split_manifest_df, dataset_root)
    else:
        manifest_df = manifest_df.reset_index(drop=True)

    manifest_df.to_csv(dataset_root / "manifest.csv", index=False)

    review_manifest.to_csv(dataset_root / "review_manifest.csv", index=False)

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(dataset_root / "skipped_images.csv", index=False)

    config = {
        "dataset_folder": str(Path(dataset_folder).resolve()),
        "label_output_dir": str(Path(label_output_dir).resolve()),
        "dataset_output_dir": str(dataset_root.resolve()),
        "label_threshold": label_threshold,
        "image_size": image_size,
        "resize_strategy": "pad",
        "pad_color": list(fill_color),
        "output_format": output_format,
        "jpeg_quality": jpeg_quality,
        "resample": resample_name,
        "include_review": include_review,
        "create_splits": create_splits,
        "split_ratios": split_ratios,
        "split_seed": split_seed,
        "train_augmentation_enabled": augment_train_horizontal_flip,
        "train_augmentation_types": ["horizontal_flip"] if augment_train_horizontal_flip else [],
        "augmented_train_count": augmented_train_count,
        "class_counts": manifest_df["label"].value_counts().sort_index().to_dict(),
        "split_counts": build_split_summary(split_manifest_df if not split_manifest_df.empty else manifest_df) if create_splits else {},
        "num_review_images": int((df_image_summary["final_label"] == "review").sum()),
        "num_skipped_images": len(skipped_rows),
    }
    with open(dataset_root / "dataset_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    archive_path = create_zip_bundle(dataset_root) if zip_output else None

    return manifest_df, review_manifest, skipped_rows, archive_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a model-ready RPS dataset by combining cleaned comment labels with preprocessed images."
    )
    parser.add_argument("folder", type=str, help="Folder containing source images and metadata JSON files")
    parser.add_argument("--label-threshold", type=float, default=0.6, help="Majority-vote threshold for final labels")
    parser.add_argument("--label-output-dir", type=str, default="output_model_handoff", help="Where to save comment-processing outputs")
    parser.add_argument("--dataset-output-dir", type=str, default="prepared_dataset", help="Where to save processed images and manifests")
    parser.add_argument("--image-size", type=int, default=128, help="Square output size for processed images")
    parser.add_argument("--pad-color", type=parse_rgb_triplet, default=(0, 0, 0), help="RGB pad color, e.g. 0,0,0")
    parser.add_argument("--output-format", choices=["jpg", "png"], default="jpg", help="File format for processed images")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality used when output format is jpg")
    parser.add_argument("--resample", choices=sorted(RESAMPLE_MAP.keys()), default="lanczos", help="Resampling filter used during resize")
    parser.add_argument("--include-review", action="store_true", help="Also export review images into a review/ subfolder")
    parser.add_argument("--create-splits", action="store_true", help="Create stratified train/val/test manifests and split folders")
    parser.add_argument("--split-ratios", type=parse_split_ratios, default=parse_split_ratios("0.8,0.1,0.1"), help="Train,val,test split ratios, e.g. 0.8,0.1,0.1")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed used for reproducible dataset splits")
    parser.add_argument("--augment-train-horizontal-flip", action="store_true", help="Export horizontal-flip augmentation for training split images only")
    parser.add_argument("--zip-output", action="store_true", help="Create a zip archive of the final dataset folder")
    args = parser.parse_args()

    if args.image_size <= 0:
        raise ValueError("image size must be positive")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("jpeg quality must be between 1 and 100")

    manifest_df, review_manifest, skipped_rows, archive_path = prepare_dataset(
        dataset_folder=args.folder,
        label_output_dir=args.label_output_dir,
        dataset_output_dir=args.dataset_output_dir,
        label_threshold=args.label_threshold,
        image_size=args.image_size,
        fill_color=args.pad_color,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        resample_name=args.resample,
        include_review=args.include_review,
        create_splits=args.create_splits,
        split_ratios=args.split_ratios,
        split_seed=args.split_seed,
        augment_train_horizontal_flip=args.augment_train_horizontal_flip,
        zip_output=args.zip_output,
    )

    print("\n=== MODEL-READY DATASET SUMMARY ===")
    if manifest_df.empty:
        print("No labeled images were exported.")
    else:
        print(manifest_df["label"].value_counts().sort_index())
    if create_splits := ("split" in manifest_df.columns and not manifest_df["split"].replace("", pd.NA).isna().all()):
        print("\nSplit counts:")
        split_manifest_path = Path(args.dataset_output_dir) / "train_manifest.csv"
        if split_manifest_path.exists():
            split_counts_df = pd.concat(
                [
                    pd.read_csv(Path(args.dataset_output_dir) / "train_manifest.csv"),
                    pd.read_csv(Path(args.dataset_output_dir) / "val_manifest.csv"),
                    pd.read_csv(Path(args.dataset_output_dir) / "test_manifest.csv"),
                ],
                ignore_index=True,
            )
            print(split_counts_df.groupby(["split", "label"]).size())
    print(f"\nReview images listed: {len(review_manifest)}")
    print(f"Skipped images: {len(skipped_rows)}")
    if archive_path:
        print(f"Zip bundle: {archive_path}")
    print(f"\nDataset outputs saved in {os.path.abspath(args.dataset_output_dir)}")


if __name__ == "__main__":
    main()
