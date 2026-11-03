"""Verify a relabeled dataset and draw the same annotations before and after."""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw

from taco_yolo import PREPARED, ROOT, font, render_panel, write_json


def label_rows(root, record):
    return [line.split() for line in (root / record["label"]).read_text().splitlines() if line.strip()]


def verify_relabel(previous, relabeled):
    previous, relabeled = Path(previous), Path(relabeled)
    old = json.loads((previous / "split_manifest.json").read_text())
    new = json.loads((relabeled / "split_manifest.json").read_text())
    policy = json.loads((relabeled / "label_mapping.json").read_text())
    by_name = {row["taco_name"]: row["class_id"] for row in policy["categories"]}
    old_records = {row["image_id"]: row for row in old["images"]}
    new_records = {row["image_id"]: row for row in new["images"]}
    if old_records.keys() != new_records.keys():
        raise ValueError("Image IDs changed during relabeling")
    counts = Counter()
    for image_id, before in old_records.items():
        after = new_records[image_id]
        if any(before[key] != after[key] for key in ("split", "source_file", "image", "label")):
            raise ValueError(f"Image {image_id}: source or split changed")
        if (previous / before["image"]).read_bytes() != (relabeled / after["image"]).read_bytes():
            raise ValueError(f"Image {image_id}: image bytes changed")
        old_rows, new_rows = label_rows(previous, before), label_rows(relabeled, after)
        if len(old_rows) != len(new_rows):
            raise ValueError(f"Image {image_id}: object count changed")
        for original, updated in zip(old_rows, new_rows):
            if len(original) != 5 or len(updated) != 5 or original[1:] != updated[1:]:
                raise ValueError(f"Image {image_id}: bounding-box coordinates changed")
            expected = by_name[old["class_names"][original[0]]]
            if int(updated[0]) != expected:
                raise ValueError(f"Image {image_id}: incorrect class mapping")
            counts[new["class_names"][updated[0]]] += 1
        if [int(row[0]) for row in new_rows] != after["class_ids"]:
            raise ValueError(f"Image {image_id}: labels disagree with manifest")
    summary = {"images": len(new_records), "objects": sum(counts.values()),
               "class_counts": dict(counts), "image_counts": new["image_counts"],
               "coordinates_unchanged": True, "splits_unchanged": True,
               "image_bytes_unchanged": True,
               "mapping_sha256": policy["mapping_sha256"]}
    return summary


def preview(previous, relabeled, output, image_id=601):
    previous, relabeled, output = Path(previous), Path(relabeled), Path(output)
    manifests = [json.loads((root / "split_manifest.json").read_text()) for root in (previous, relabeled)]
    panels, transitions = [], []
    for index, (root, manifest) in enumerate(zip((previous, relabeled), manifests)):
        record = next(row for row in manifest["images"] if row["image_id"] == image_id)
        with Image.open(root / record["image"]) as raw:
            image = raw.convert("RGB")
        rows = label_rows(root, record)
        boxes = []
        for cls, cx, cy, width, height in rows:
            cx, cy, width, height = map(float, (cx, cy, width, height))
            boxes.append({"class_id": int(cls), "xyxy": [
                (cx - width / 2) * image.width, (cy - height / 2) * image.height,
                (cx + width / 2) * image.width, (cy + height / 2) * image.height]})
        names = {int(key): value for key, value in manifest["class_names"].items()}
        transitions.append([names[box["class_id"]] for box in boxes])
        title = "Before: 60 TACO classes" if index == 0 else "After: 6 TrashNet classes"
        panel = render_panel(image, boxes, names, title, "#1661a9" if index == 0 else "#138253",
                             f"Image {image_id:06d} | {len(boxes)} objects | identical bounding boxes")
        panels.append(panel)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, panel in zip(("before", "after"), panels):
        path = output / f"image_{image_id:06d}_{name}.png"
        panel.save(path)
        paths[name] = str(path)
    changes = list(dict.fromkeys(zip(*transitions)))
    combined = Image.new("RGB", (1640, 966 + len(changes) * 28), "#e5ebf1")
    combined.paste(panels[0], (0, 0))
    combined.paste(panels[1], (820, 0))
    draw = ImageDraw.Draw(combined)
    draw.text((22, 916), "Same image and boxes; only the object class IDs and names change.",
              fill="#334155", font=font(19))
    for index, (before, after) in enumerate(changes):
        draw.text((22, 950 + index * 28), f"{before}  ->  {after}", fill="#334155", font=font(19))
    path = output / f"image_{image_id:06d}_comparison.jpg"
    combined.save(path, quality=94)
    paths["comparison"] = str(path)
    write_json(output / f"image_{image_id:06d}_transitions.json", [
        {"object": i + 1, "before": before, "after": after}
        for i, (before, after) in enumerate(zip(*transitions))])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=ROOT / "data/yolo/taco")
    parser.add_argument("--relabeled", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/taco_relabel")
    parser.add_argument("--image-id", type=int, default=601)
    args = parser.parse_args()
    summary = verify_relabel(args.previous, args.relabeled)
    write_json(args.output / "verification.json", summary)
    paths = preview(args.previous, args.relabeled, args.output, args.image_id)
    print(json.dumps({**summary, "previews": paths}, indent=2))


if __name__ == "__main__":
    main()
