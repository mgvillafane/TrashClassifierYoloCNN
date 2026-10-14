"""Small, source-preserving previews of four waste datasets.

python dataset_tools.py fetch
python dataset_tools.py report
python dataset_tools.py import-coco --help
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import textwrap
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parent
CATALOG = json.loads((ROOT / "config/datasets.json").read_text(encoding="utf-8"))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
COLORS = ["#267d9c", "#df7b43", "#6a8c4d", "#a369a0", "#d5a83a", "#587cb8"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fetch_file(url, destination):
    """Cache downloads; a failed download never replaces a complete file."""
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "WasteDatasetPreview/1.0"})
            with urlopen(request, timeout=30) as response, partial.open("wb") as stream:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, stream)
            if not partial.stat().st_size:
                raise ValueError(f"Empty response from {url}")
            if expected and partial.stat().st_size != int(expected):
                raise ValueError(f"Incomplete download from {url}")
            partial.replace(destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            write_json(destination.with_name(destination.name + ".source.json"), {
                "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digest, "bytes": destination.stat().st_size,
            })
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def local_path(path):
    """Use project-relative paths where possible, portable absolute paths otherwise."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_image(row):
    return ROOT / row["path"]


def safe_child(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes its root: {filename}")
    return path


def image_size(path):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def balanced_sample(items, label_key, per_class, seed):
    grouped = defaultdict(list)
    for item in items:
        grouped[label_key(item)].append(item)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        pool = sorted(grouped[label], key=str)
        selected.extend(rng.sample(pool, min(per_class, len(pool))))
    return selected


def diverse_coco_sample(coco, limit, seed):
    """Greedily cover categories, favoring rare labels; deterministic tie order."""
    labels = defaultdict(set)
    for ann in coco["annotations"]:
        labels[ann["image_id"]].add(ann["category_id"])
    frequency = Counter(label for values in labels.values() for label in values)
    pool = sorted(coco["images"], key=lambda image: image["id"])
    random.Random(seed).shuffle(pool)
    seen, selected = set(), []
    while pool and len(selected) < limit:
        best = max(pool, key=lambda image: sum(
            1 / frequency[label] for label in labels[image["id"]] - seen
        ))
        selected.append(best)
        seen.update(labels[best["id"]])
        pool.remove(best)
    return selected


def make_classification_row(dataset, path, label, source_file, source_url):
    width, height = image_size(path)
    return {
        "dataset": dataset, "task": "classification", "image_id": source_file,
        "path": local_path(path), "source_file": source_file, "source_url": source_url,
        "split": "unspecified", "width": width, "height": height,
        "annotations": [{"label": label, "supercategory": None, "bbox_xywh": None}],
    }


def coco_rows(coco, images, image_root, dataset, split="unspecified"):
    categories = {category["id"]: category for category in coco["categories"]}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        category = categories[ann["category_id"]]
        annotations[ann["image_id"]].append({
            "annotation_id": ann.get("id"), "category_id": category["id"],
            "label": category["name"], "supercategory": category.get("supercategory"),
            "bbox_xywh": ann.get("bbox"), "iscrowd": ann.get("iscrowd", 0),
        })
    rows, missing = [], []
    for image in images:
        path = safe_child(image_root, image["file_name"])
        if not path.is_file():
            missing.append(image["file_name"])
            continue
        width, height = image_size(path)
        rows.append({
            "dataset": dataset, "task": "detection", "image_id": image["id"],
            "path": local_path(path), "source_file": image["file_name"],
            "source_url": image.get("flickr_url") or image.get("coco_url"),
            "license_id": image.get("license"), "split": split,
            "width": width, "height": height,
            "annotation_width": image["width"], "annotation_height": image["height"],
            "annotations": annotations[image["id"]],
        })
    return rows, missing


def save_rows(rows, data_dir, dataset, split=None):
    path = data_dir / "processed/manifest.jsonl"
    old = read_rows(path)
    old = [row for row in old if not (
        row["dataset"] == dataset and (split is None or row["split"] == split)
    )]
    rows = sorted(old + rows, key=lambda row: (row["dataset"], row["split"], str(row["image_id"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path=None):
    path = Path(path or ROOT / "data/processed/manifest.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def save_counts(data_dir, dataset, counts, unit, scope):
    write_json(data_dir / "raw" / dataset / "class_counts.json", {
        "unit": unit, "scope": scope, "counts": dict(counts),
    })


def fetch_realwaste(data_dir, per_class, seed):
    spec = CATALOG["realwaste"]
    tree_path = fetch_file(spec["tree_url"], data_dir / "raw/realwaste/tree.json")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise ValueError("GitHub's file list was truncated; use the UCI archive instead.")
    files = [entry["path"] for entry in tree["tree"] if
             entry["path"].startswith("RealWaste/") and Path(entry["path"]).suffix.lower() in EXTENSIONS]
    label_key = lambda name: Path(name).parent.name
    save_counts(data_dir, "realwaste", Counter(map(label_key, files)), "images", "full repository file list")
    rows = []
    for filename in balanced_sample(files, label_key, per_class, seed):
        url = spec["raw_base"] + quote(filename, safe="/")
        path = fetch_file(url, safe_child(data_dir / "preview/realwaste", filename))
        rows.append(make_classification_row("realwaste", path, label_key(filename), filename, url))
    return rows


def fetch_trashnet(data_dir, per_class, seed):
    url = CATALOG["trashnet"]["archive_url"]
    path = fetch_file(url, data_dir / "raw/trashnet/dataset-resized.zip")
    rows = []
    with zipfile.ZipFile(path) as archive:
        files = [name for name in archive.namelist() if
                 name.startswith("dataset-resized/") and Path(name).suffix.lower() in EXTENSIONS]
        label_key = lambda name: Path(name).parent.name
        save_counts(data_dir, "trashnet", Counter(map(label_key, files)), "images", "full resized archive")
        for filename in balanced_sample(files, label_key, per_class, seed):
            destination = safe_child(data_dir / "preview/trashnet", filename)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(filename) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
            rows.append(make_classification_row("trashnet", destination, label_key(filename), filename, url))
    return rows


def fetch_taco(data_dir, limit, seed):
    path = fetch_file(CATALOG["taco"]["annotations_url"], data_dir / "raw/taco/annotations.json")
    coco = json.loads(path.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in coco["categories"]}
    counts = Counter({name: 0 for name in category_names.values()})
    counts.update(category_names[ann["category_id"]] for ann in coco["annotations"])
    save_counts(data_dir, "taco", counts, "objects", "full reviewed annotations.json")
    selected = diverse_coco_sample(coco, limit, seed)
    failures = []
    for image in selected:
        try:
            # Do not use flickr_640_url: its coordinates differ from the annotation canvas.
            url = image.get("flickr_url") or image.get("coco_url")
            if not url:
                raise ValueError("No original-size image URL")
            fetch_file(url, safe_child(data_dir / "preview/taco", image["file_name"]))
        except Exception as exc:
            failures.append({"file": image["file_name"], "error": str(exc)})
    write_json(data_dir / "raw/taco/download_failures.json", failures)
    rows, _ = coco_rows(coco, selected, data_dir / "preview/taco", "taco")
    if failures:
        print(f"  TACO: {len(failures)} unavailable images; see download_failures.json", flush=True)
    if not rows:
        raise ValueError("No TACO images were available; see download_failures.json")
    return rows


def fetch_preview(data_dir, per_class=2, scenes=12, seed=42):
    errors = {}
    jobs = {
        "realwaste": lambda: fetch_realwaste(data_dir, per_class, seed),
        "trashnet": lambda: fetch_trashnet(data_dir, per_class, seed),
        "taco": lambda: fetch_taco(data_dir, scenes, seed),
    }
    for dataset, job in jobs.items():
        print(f"Preparing {dataset} ...", flush=True)
        try:
            rows = job()
            save_rows(rows, data_dir, dataset)
            print(f"  Indexed {len(rows)} images", flush=True)
        except Exception as exc:
            errors[dataset] = str(exc)
            print(f"  Failed: {exc}", flush=True)
    try:
        fetch_file("https://raw.githubusercontent.com/sarainacio/SortWaste/main/images/teaser%282%29.png",
                   data_dir / "raw/sortwaste/reference.png")
        save_counts(data_dir, "sortwaste", CATALOG["sortwaste"]["published_object_counts"],
                    "objects", "published full dataset table; reference figure is not indexed")
    except Exception as exc:
        errors["sortwaste_reference"] = str(exc)
    write_json(data_dir / "raw/preview_run.json", {
        "seed": seed, "images_per_class": per_class, "requested_taco_scenes": scenes, "errors": errors,
    })
    return errors


def import_coco(annotation_path, image_root, dataset, split, data_dir, limit=None):
    """Index local images, optionally limiting selection to COCO JSON image order."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    coco = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    selected = coco["images"] if limit is None else coco["images"][:limit]
    rows, missing = coco_rows(coco, selected, image_root, dataset, split)
    save_rows(rows, data_dir, dataset, split)
    # Keep source COCO JSON, including polygons and license metadata, untouched.
    write_json(data_dir / "raw" / dataset / f"import_{split}.json", {
        "annotation_path": str(Path(annotation_path).resolve()), "image_root": str(Path(image_root).resolve()),
        "split": split, "indexed_images": len(rows), "missing_images": missing,
        "limit": limit, "requested_images": len(selected), "total_source_images": len(coco["images"]),
        "annotation_sha256": hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest(),
    })
    print(f"Indexed {len(rows)} {dataset} images ({split}); missing {len(missing)}", flush=True)
    return rows


def draw_samples(rows, output, columns=4, limit=12, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not rows:
        raise ValueError("No images match the requested selection")
    # Spread display examples across classes rather than taking the first sorted rows.
    if all(row["task"] == "classification" for row in rows):
        rows = balanced_sample(rows, lambda row: row["annotations"][0]["label"], 1, 42)[:limit]
        columns = min(columns, 3)
    else:
        rows = rows[:limit]
    columns = min(columns, len(rows))
    fig, axes = plt.subplots(math.ceil(len(rows) / columns), columns,
                             figsize=(columns * 2.5, math.ceil(len(rows) / columns) * 2.5), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, rows):
        with Image.open(resolve_image(row)) as image:
            # Deliberately preserve stored orientation to match COCO coordinates.
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((800, 800))
            axis.imshow(thumbnail, extent=(0, row["width"], row["height"], 0))
        sx = row["width"] / row.get("annotation_width", row["width"])
        sy = row["height"] / row.get("annotation_height", row["height"])
        labels = sorted({ann["label"] for ann in row["annotations"]})
        for ann in row["annotations"]:
            box = ann.get("bbox_xywh")
            if box:
                x, y, w, h = box
                color = COLORS[int(hashlib.md5(ann["label"].encode()).hexdigest(), 16) % len(COLORS)]
                axis.add_patch(Rectangle((x * sx, y * sy), w * sx, h * sy,
                                         fill=False, edgecolor=color, linewidth=1.3))
                axis.text(x * sx, y * sy, ann["label"], fontsize=5, color="white",
                          bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "edgecolor": "none"})
        caption = ", ".join(labels[:2]) + (f" +{len(labels)-2}" if len(labels) > 2 else "")
        axis.set_title(textwrap.fill(caption or "Unannotated", 30), fontsize=8)
    fig.suptitle(title or "Waste dataset preview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    return output


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(data_dir, report_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(data_dir / "processed/manifest.jsonl")
    overview, count_rows, flat, issues = [], [], [], []
    for row in rows:
        if not resolve_image(row).is_file():
            issues.append({"file": row["path"], "issue": "missing image"})
        if row.get("annotation_width") and (row["width"], row["height"]) != (row["annotation_width"], row["annotation_height"]):
            issues.append({"file": row["path"], "issue": "image size differs from annotation canvas; preview scales boxes"})
        for ann in row["annotations"]:
            box = ann.get("bbox_xywh")
            if box:
                x, y, w, h = box
                if not all(math.isfinite(v) for v in box) or min(x, y) < 0 or min(w, h) <= 0 or x+w > row.get("annotation_width", row["width"])+1 or y+h > row.get("annotation_height", row["height"])+1:
                    issues.append({"file": row["path"], "issue": "invalid or out-of-bounds box", "box": box})
            flat.append({"dataset": row["dataset"], "task": row["task"], "image_id": row["image_id"],
                         "split": row["split"], "path": row["path"], "width": row["width"], "height": row["height"],
                         "label": ann["label"], "supercategory": ann.get("supercategory"),
                         "bbox_xywh": json.dumps(box) if box else ""})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (dataset, spec) in zip(axes.flat, CATALOG.items()):
        subset = [row for row in rows if row["dataset"] == dataset]
        preview_counts = Counter(ann["label"] for row in subset for ann in row["annotations"])
        count_file = data_dir / "raw" / dataset / "class_counts.json"
        source_counts = json.loads(count_file.read_text(encoding="utf-8")) if count_file.exists() else None
        counts = source_counts["counts"] if source_counts else preview_counts
        scope = source_counts["scope"] if source_counts else "locally indexed images only"
        unit = source_counts["unit"] if source_counts else ("objects" if spec["task"] == "detection" else "images")
        top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:12]
        if top:
            names, values = zip(*reversed(top))
            axis.barh(names, values, color=COLORS[list(CATALOG).index(dataset)])
            axis.tick_params(axis="y", labelsize=8)
        axis.set_title(f"{spec['name']} | {unit} per class" + (" (top 12)" if len(counts) > 12 else ""))
        axis.set_xlabel(scope, fontsize=8)
        overview.append({"dataset": dataset, "task": spec["task"], "published_images": spec["expected_images"],
                         "native_labels": spec["label_count"], "indexed_images": len(subset),
                         "indexed_annotations": sum(preview_counts.values()), "indexed_labels": len(preview_counts),
                         "status": "images indexed" if subset else "reference only / not indexed"})
        for label in sorted(counts):
            count_rows.append({"dataset": dataset, "label": label, "unit": unit,
                               "source_count": counts[label], "scope": scope,
                               "indexed_count": preview_counts[label]})
        if subset:
            draw_samples(subset, report_dir / f"{dataset}_samples.png", title=f"{spec['name']} | local examples")
    fig.suptitle("Waste datasets | source class distributions", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(report_dir / "class_distributions.png", dpi=150, facecolor="white")
    plt.close(fig)
    write_csv(report_dir / "overview.csv", overview, list(overview[0]))
    write_csv(report_dir / "class_counts.csv", count_rows,
              ["dataset", "label", "unit", "source_count", "scope", "indexed_count"])
    write_csv(report_dir / "annotations.csv", flat,
              ["dataset", "task", "image_id", "split", "path", "width", "height", "label", "supercategory", "bbox_xywh"])
    write_json(report_dir / "quality_issues.json", issues)
    print(f"Reports written to {report_dir}; {len(rows)} images, {len(issues)} quality issues", flush=True)
    return overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch a small preview, plus source metadata")
    fetch.add_argument("--per-class", type=int, default=2)
    fetch.add_argument("--scenes", type=int, default=12)
    fetch.add_argument("--seed", type=int, default=42)
    sub.add_parser("report", help="Build CSV summaries and small image sheets offline")
    coco = sub.add_parser("import-coco", help="Index local COCO images without changing their source files")
    coco.add_argument("--dataset", choices=["sortwaste", "taco"], required=True)
    coco.add_argument("--annotations", type=Path, required=True)
    coco.add_argument("--images", type=Path, required=True, help="Folder relative to which COCO file_name resolves")
    coco.add_argument("--split", choices=["train", "val", "test", "unspecified"], default="unspecified")
    coco.add_argument("--limit", type=int, help="Index only the first N image records in the COCO JSON")
    args = parser.parse_args()
    if args.command == "fetch":
        if args.per_class < 1 or args.scenes < 1:
            parser.error("--per-class and --scenes must be positive")
        errors = fetch_preview(args.data_dir, args.per_class, args.scenes, args.seed)
        report(args.data_dir, args.report_dir)
        if errors:
            raise SystemExit("Some sources failed. See data/raw/preview_run.json; rerun to retry.")
    elif args.command == "import-coco":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        import_coco(args.annotations, args.images, args.dataset, args.split, args.data_dir, args.limit)
        report(args.data_dir, args.report_dir)
    else:
        report(args.data_dir, args.report_dir)


if __name__ == "__main__":
    main()
