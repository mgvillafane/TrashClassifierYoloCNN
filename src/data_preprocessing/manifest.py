"""Build, validate and summarize the unified manifest (one row per classification example or box)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .datasets import CONFLICT_REASON, list_images, realwaste, sortwaste, taco, trashnet
from .labels import apply_mapping, build_rules
from .paths import DataPaths
from .splits import assign_whole_image_splits, load_taco_folds

COLUMNS = [
    "dataset", "split", "taco_batch", "taco_fold", "example_type",
    "image_path", "label_path", "source_image_id", "taco_annotation_id", "content_md5",
    "img_w", "img_h", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "bbox_clipped",
    "original_class_id", "original_label", "common_label", "common_class_id",
    "use_detector_train", "use_detector_eval", "use_classifier_train", "use_classifier_eval",
    "exclude_reason",
]
INT_COLUMNS = ["taco_batch", "taco_fold", "taco_annotation_id", "original_class_id", "common_class_id"]
BBOX = ["bbox_cx", "bbox_cy", "bbox_w", "bbox_h"]

# Verified against the raw data on 2026-11-05. SortWaste docs quote 87,252 boxes,
# but the YOLO and COCO annotations both contain 87,525.
EXPECTED = {
    "images": {"realwaste": 4752, "trashnet": 2527, "sortwaste": 5261, "taco": 1500},
    "boxes": {"sortwaste": 87525, "taco": 4784},
    "sortwaste_split_images": {"train": 3705, "val": 780, "test": 776},
    "sortwaste_no_object_images": 4,
    "trashnet_conflicting_files": 6,
}


def build_manifest(paths: DataPaths, config: dict) -> pd.DataFrame:
    rules = build_rules(config)
    sortwaste_names = {int(k): v["name"] for k, v in config["sortwaste"].items()}
    raw_rows = (realwaste.rows(paths) + trashnet.rows(paths)
                + sortwaste.rows(paths, sortwaste_names) + taco.rows(paths))
    df = pd.DataFrame([apply_mapping(r, rules, config) for r in raw_rows]).reindex(columns=COLUMNS)

    split_cfg = config["splits"]
    whole = df["example_type"] == "whole_image"
    df.loc[whole, "split"] = assign_whole_image_splits(
        df[whole], split_cfg["whole_image_seed"], split_cfg["whole_image_ratios"])

    folds_file = paths.root / split_cfg["taco_folds_file"]
    if folds_file.exists():
        folds = load_taco_folds(folds_file)
        df["taco_fold"] = df["taco_batch"].map(folds)

    for col in INT_COLUMNS:
        df[col] = df[col].astype("Int64")
    df["bbox_clipped"] = df["bbox_clipped"].astype("boolean")
    return df


def read_manifest(path) -> pd.DataFrame:
    """Load a saved manifest with the same dtypes build_manifest produces."""
    bools = ["bbox_clipped", "use_detector_train", "use_detector_eval",
             "use_classifier_train", "use_classifier_eval"]
    dtypes = {**dict.fromkeys(INT_COLUMNS, "Int64"), **dict.fromkeys(bools, "boolean")}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


def check_raw_files(paths: DataPaths, expected: dict = EXPECTED) -> pd.DataFrame:
    """Count raw files per dataset against the verified totals (catches unfinished unzips/uploads).

    Only lists folders, so it is quick even on a network volume. Truncated files are not detected
    here; they fail later when the export opens them.
    """
    def count(folder: Path, pattern: str | None = None) -> int:
        if not folder.is_dir():
            return 0
        return len(list(folder.glob(pattern))) if pattern else len(list_images(folder))

    rows = [("realwaste images", paths.realwaste, expected["images"]["realwaste"], count(paths.realwaste)),
            ("trashnet images", paths.trashnet, expected["images"]["trashnet"], count(paths.trashnet))]
    for split, n in expected["sortwaste_split_images"].items():
        rows += [(f"sortwaste {split} images", paths.sortwaste_images / split / "images", n,
                  count(paths.sortwaste_images / split / "images")),
                 (f"sortwaste {split} labels", paths.sortwaste_labels / split / "labels", n,
                  count(paths.sortwaste_labels / split / "labels", "*.txt")),
                 (f"sortwaste {split} COCO json", paths.sortwaste_images / split / "annotations", 1,
                  count(paths.sortwaste_images / split / "annotations", "*.json"))]
    annotations = paths.taco / "annotations.json"
    if annotations.is_file():
        files = [im["file_name"] for im in json.loads(annotations.read_text(encoding="utf-8"))["images"]]
        found = sum((paths.taco / f).is_file() for f in files)
    else:
        found = 0
    rows.append(("taco images listed in annotations.json", paths.taco, expected["images"]["taco"], found))
    table = pd.DataFrame(rows, columns=["what", "folder", "expected", "found"])
    table["ok"] = table["expected"] == table["found"]
    return table


def validate_manifest(df: pd.DataFrame, config: dict, expected: dict = EXPECTED) -> None:
    """Raise ValueError listing every failed check."""
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    images = df.groupby("dataset")["source_image_id"].nunique().to_dict()
    for ds, n in expected["images"].items():
        check(images.get(ds) == n, f"{ds}: {images.get(ds)} images, expected {n}")
    boxes = df[df["example_type"] == "bounding_box"].groupby("dataset").size().to_dict()
    for ds, n in expected["boxes"].items():
        check(boxes.get(ds) == n, f"{ds}: {boxes.get(ds)} boxes, expected {n}")

    sw = df[df["dataset"] == "sortwaste"]
    split_images = sw.groupby("split")["source_image_id"].nunique().to_dict()
    check(split_images == expected["sortwaste_split_images"], f"sortwaste split images {split_images}")
    check((sw["example_type"] == "no_object").sum() == expected["sortwaste_no_object_images"],
          "unexpected number of empty SortWaste label files")
    n_conflicts = (df["exclude_reason"] == CONFLICT_REASON).sum()
    check(n_conflicts == expected["trashnet_conflicting_files"], f"{n_conflicts} conflicting duplicate files")

    for ds, group in df[df["dataset"] != "taco"].groupby("dataset"):
        check(group.groupby("source_image_id")["split"].nunique().max() == 1, f"{ds}: an image spans splits")
        check(group["split"].isin(["train", "val", "test"]).all(), f"{ds}: missing or unknown split")
    taco_rows = df[df["dataset"] == "taco"]
    check(taco_rows["split"].isna().all(), "TACO rows must not carry a fixed split (they use folds)")
    if taco_rows["taco_fold"].notna().any():
        check(taco_rows["taco_fold"].notna().all(), "some TACO batches have no fold")
        check(taco_rows.groupby("source_image_id")["taco_fold"].nunique().max() == 1, "a TACO image spans folds")

    is_box = df["example_type"] == "bounding_box"
    check(df.loc[is_box, BBOX].apply(lambda c: c.between(0, 1)).all().all(), "bbox outside [0, 1]")
    check((df.loc[is_box, ["bbox_w", "bbox_h"]] > 0).all().all(), "zero-area bbox")
    check(df.loc[~is_box, BBOX].isna().all().all(), "non-box rows must have blank bbox columns")
    check(not df["image_path"].str.startswith("/").any(), "absolute image path in manifest")
    check(df.loc[is_box | (df["example_type"] == "whole_image"), "common_label"]
          .isin(config["common_labels"]).all(), "unknown common label")
    check(not (df["use_classifier_train"] & df["exclude_reason"].notna()).any(), "excluded row marked trainable")
    check(not (df["use_classifier_train"] & (df["dataset"] == "taco")).any(), "TACO used for classifier training")

    if problems:
        raise ValueError("manifest validation failed:\n  " + "\n  ".join(problems))


def partition_column(df: pd.DataFrame) -> pd.Series:
    """train/val/test for fixed splits, fold0..k for TACO."""
    fold = "fold" + df["taco_fold"].astype("string")
    return df["split"].astype("string").fillna(fold).fillna("unassigned")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: dataset x partition x common label with detector-box and classifier-example counts."""
    labelled = df[df["example_type"] != "no_object"].assign(partition=partition_column(df))
    keys = ["dataset", "partition", "common_label"]
    return (
        labelled.groupby(keys)
        .agg(detector_boxes=("use_detector_eval", "sum"),
             classifier_examples=("use_classifier_eval", "sum"),
             all_rows=("dataset", "size"))
        .reset_index()
    )


def class_table(df: pd.DataFrame, config: dict, column: str = "use_classifier_eval") -> pd.DataFrame:
    rows = df[df[column]]
    return pd.crosstab(rows["dataset"], rows["common_label"]).reindex(
        index=["realwaste", "trashnet", "sortwaste", "taco"], columns=config["common_labels"], fill_value=0)


def inventory(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    structure = {
        "realwaste": "class folders; 80/10/10 split generated (seed {seed})",
        "trashnet": "class folders; 80/10/10 split generated (seed {seed})",
        "sortwaste": "official train/val/test",
        "taco": "15 batches -> {k} batch folds (external test)",
    }
    seed, k = config["splits"]["whole_image_seed"], config["splits"]["taco_n_folds"]
    out = []
    for ds in ["realwaste", "trashnet", "sortwaste", "taco"]:
        g = df[df["dataset"] == ds]
        boxes = (g["example_type"] == "bounding_box").sum()
        out.append({
            "dataset": ds,
            "task": "detection" if boxes else "classification",
            "images": g["source_image_id"].nunique(),
            "boxes": boxes or None,
            "original_classes": g["original_label"].dropna().nunique(),
            "common_classes_covered": g.loc[g["use_classifier_eval"], "common_label"].nunique(),
            "classifier_examples": int(g["use_classifier_eval"].sum()),
            "structure": structure[ds].format(seed=seed, k=k),
        })
    return pd.DataFrame(out).astype({"boxes": "Int64"}).set_index("dataset")


def exclusion_table(df: pd.DataFrame) -> pd.DataFrame:
    excluded = df[df["exclude_reason"].notna()]
    return (excluded.groupby(["dataset", "exclude_reason", "original_label"], dropna=False).size()
            .rename("rows").reset_index().sort_values(["dataset", "rows"], ascending=[True, False]))
