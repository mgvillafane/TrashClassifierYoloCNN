"""TACO: COCO annotations over 1,500 images in 15 batches.

Boxes are relative to the EXIF-upright image, and the annotation width/height
already describe that upright canvas (verified for all 1,500 images).
"""

from __future__ import annotations

import json
import math

from ..paths import DataPaths


def coco_to_yolo_box(box, width: int, height: int) -> tuple[float, float, float, float, bool]:
    """COCO pixel [x, y, w, h] -> normalized (cx, cy, w, h), clipped to the canvas.

    The last value reports whether clipping changed the box.
    """
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"invalid COCO box: {box}")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box lies outside the image: {box}")
    clipped = (x1, y1, x2, y2) != (x, y, x + w, y + h)
    return (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height, clipped


def batch_number(file_name: str) -> int:
    return int(file_name.split("/")[0].removeprefix("batch_"))


def rows(paths: DataPaths) -> list[dict]:
    coco = json.loads((paths.taco / "annotations.json").read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    out = []
    for ann in coco["annotations"]:
        im = images[ann["image_id"]]
        img_path = paths.taco / im["file_name"]
        cx, cy, w, h, clipped = coco_to_yolo_box(ann["bbox"], im["width"], im["height"])
        out.append({
            "dataset": "taco",
            "example_type": "bounding_box",
            "image_path": paths.rel(img_path),
            "source_image_id": paths.rel(img_path),
            "taco_batch": batch_number(im["file_name"]),
            "taco_annotation_id": ann["id"],
            "img_w": im["width"], "img_h": im["height"],
            "bbox_cx": cx, "bbox_cy": cy, "bbox_w": w, "bbox_h": h,
            "bbox_clipped": clipped,
            "original_class_id": ann["category_id"],
            "original_label": categories[ann["category_id"]],
        })
    return out
