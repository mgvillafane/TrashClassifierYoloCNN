"""Per-dataset readers. Each returns manifest rows (dicts) before label mapping."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CONFLICT_REASON = "identical image filed under different classes"


def list_images(folder: Path) -> list[Path]:
    """Image files under folder, skipping macOS metadata (__MACOSX, ._* resource forks, dotfiles)."""
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith(".")
        and "__MACOSX" not in p.parts
    )


def whole_image_rows(dataset: str, root: Path, rel) -> list[dict]:
    """Classification datasets: class folder name = original label, one image = one example.

    Byte-identical images filed under two different classes are flagged for exclusion,
    because their label is contradictory.
    """
    rows = []
    for class_dir in sorted(d for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith(".")):
        for img in list_images(class_dir):
            with Image.open(img) as im:
                w, h = im.size
            rows.append({
                "dataset": dataset,
                "example_type": "whole_image",
                "image_path": rel(img),
                "source_image_id": rel(img),
                "content_md5": hashlib.md5(img.read_bytes()).hexdigest(),
                "img_w": w, "img_h": h,
                "original_class_id": None,
                "original_label": class_dir.name,
            })

    labels_by_hash = defaultdict(set)
    for row in rows:
        labels_by_hash[row["content_md5"]].add(row["original_label"])
    for row in rows:
        if len(labels_by_hash[row["content_md5"]]) > 1:
            row["exclude_reason"] = CONFLICT_REASON
    return rows
