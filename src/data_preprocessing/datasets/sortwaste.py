"""SortWaste: YOLO labels (splited_all_dataset_yolo) paired with images from the COCO copy.

The paths inside data.yaml point at the dataset author's machine and are ignored.
YOLO class ids are list positions 0-7; the numbers embedded in data.yaml names are not ids.
"""

from __future__ import annotations

import json

from ..paths import DataPaths
from . import list_images

SPLITS = ("train", "val", "test")


def parse_yolo_line(line: str, n_classes: int, where: str = "") -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{where}: expected 5 fields, got {len(parts)}")
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:])
    if not 0 <= cls < n_classes:
        raise ValueError(f"{where}: class id {cls} out of range")
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        raise ValueError(f"{where}: coordinates not in [0, 1]")
    if w <= 0 or h <= 0:
        raise ValueError(f"{where}: zero-area box")
    return cls, cx, cy, w, h


def rows(paths: DataPaths, class_names: dict[int, str]) -> list[dict]:
    """One YOLO box = one row; an empty label file becomes one 'no_object' row (detector negative)."""
    out = []
    for split in SPLITS:
        coco_file = next((paths.sortwaste_images / split / "annotations").glob("*.json"))
        coco = json.loads(coco_file.read_text(encoding="utf-8"))
        sizes = {im["file_name"].rsplit(".", 1)[0]: (im["width"], im["height"]) for im in coco["images"]}
        images = {p.stem: p for p in list_images(paths.sortwaste_images / split / "images")}
        label_files = sorted((paths.sortwaste_labels / split / "labels").glob("*.txt"))

        stems = {f.stem for f in label_files}
        if stems != set(images):
            raise ValueError(f"sortwaste/{split}: {len(stems - set(images))} labels without image, "
                             f"{len(set(images) - stems)} images without label")

        for label_file in label_files:
            img = images[label_file.stem]
            img_w, img_h = sizes[label_file.stem]
            base = {
                "dataset": "sortwaste", "split": split,
                "image_path": paths.rel(img), "label_path": paths.rel(label_file),
                "source_image_id": paths.rel(img), "img_w": img_w, "img_h": img_h,
            }
            lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                out.append({**base, "example_type": "no_object"})
            for k, line in enumerate(lines):
                cls, cx, cy, w, h = parse_yolo_line(line, len(class_names), f"{label_file.name}:{k + 1}")
                out.append({
                    **base, "example_type": "bounding_box",
                    "bbox_cx": cx, "bbox_cy": cy, "bbox_w": w, "bbox_h": h,
                    "original_class_id": cls, "original_label": class_names[cls],
                })
    return out
