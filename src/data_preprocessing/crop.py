"""Turn manifest rows into image crops on the fly (nothing is written to disk).

Normalized YOLO boxes are converted to pixels, optionally padded, and clamped to the
image. Images are EXIF-transposed first because TACO boxes refer to the upright image.
"""

from __future__ import annotations

from PIL import Image, ImageOps

from .paths import DataPaths


def pixel_box(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int,
              padding: float = 0.0) -> tuple[int, int, int, int]:
    """Normalized (cx, cy, w, h) -> clamped integer (left, top, right, bottom).

    padding is a fraction of the box size added on every side (0.1 = 10%).
    """
    if padding < 0:
        raise ValueError("padding must be >= 0")
    half_w, half_h = w * img_w * (0.5 + padding), h * img_h * (0.5 + padding)
    left = max(0, int(round(cx * img_w - half_w)))
    top = max(0, int(round(cy * img_h - half_h)))
    right = min(img_w, int(round(cx * img_w + half_w)))
    bottom = min(img_h, int(round(cy * img_h + half_h)))
    if right <= left or bottom <= top:
        raise ValueError(f"empty crop for box {(cx, cy, w, h)} on {img_w}x{img_h}")
    return left, top, right, bottom


def load_crop(row, paths: DataPaths, padding: float = 0.0) -> Image.Image:
    """RGB crop for a 'bounding_box' row, or the whole image for a 'whole_image' row."""
    with Image.open(paths.resolve(row["image_path"])) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    if row["example_type"] == "whole_image":
        return image
    if image.size != (row["img_w"], row["img_h"]):
        raise ValueError(f"{row['image_path']}: image is {image.size}, manifest says "
                         f"{(row['img_w'], row['img_h'])}")
    box = pixel_box(row["bbox_cx"], row["bbox_cy"], row["bbox_w"], row["bbox_h"],
                    row["img_w"], row["img_h"], padding)
    return image.crop(box)
