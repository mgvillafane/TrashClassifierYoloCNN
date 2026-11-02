"""Train, evaluate, and visualize a YOLO11 detector on the local TACO dataset.

See TACO_YOLO_GUIDE.md for setup, commands, and how to interpret the results.
Run without arguments to execute DEFAULT_EXPERIMENT, or use the CLI subcommands.
Raw images and COCO annotations are never edited by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
STORAGE = Path(os.environ.get("TACO_HOME", ROOT)).expanduser().resolve()
SOURCE = STORAGE / "data/raw/taco"
PREPARED = STORAGE / "data/yolo/taco"
RUNS = STORAGE / "reports/taco_yolo"
# Keep package settings and caches inside this project, including on Windows.
os.environ.setdefault("YOLO_CONFIG_DIR", str(STORAGE / "data/yolo/settings"))
os.environ.setdefault("MPLCONFIGDIR", str(STORAGE / "data/yolo/matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


@dataclass
class ExperimentConfig:
    source: Path | str = SOURCE
    prepared: Path | str = PREPARED
    project: Path | str = RUNS
    weights: Path | str = "yolo11n.pt"  # Or a TACO-trained best.pt to fine-tune.
    name: str = "taco"
    epochs: int = 60
    imgsz: int = 640
    batch: int = 8
    device: str = "auto"  # auto selects the first CUDA GPU, otherwise CPU.
    patience: int = 20  # 0 disables early stopping.
    lr: float = 0.001
    fraction: float = 1.0
    freeze: int = 0
    workers: int = 0
    threads: int = 4
    seed: int = 42
    max_side: int = 1280
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    prediction_count: int = 8
    conf: float = 0.25


# Edit this block, then run taco_yolo.py without command-line arguments.
# Databricks users can edit the equivalent cell in taco_databricks.py.
DEFAULT_EXPERIMENT = ExperimentConfig(
    epochs=60,
    imgsz=640,
    batch=8,
    device="auto",
    patience=20,
    lr=0.001,
    weights="yolo11n.pt",
    name="taco",
    prediction_count=8,
    conf=0.25,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalized_box(box, width, height):
    """Clip COCO xywh to the canvas, then return YOLO's normalized cxcywh."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid COCO box: {box}")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box lies outside the image: {box}")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def split_by_batch(images, seed=42, val_ratio=0.15, test_ratio=0.15):
    """Keep each source batch together to reduce leakage between similar photos."""
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation/test ratios must be positive and sum to less than 1")
    groups = sorted({Path(image["file_name"]).parts[0] for image in images})
    if len(groups) < 3:
        raise ValueError("At least three source batches are needed for grouped splitting")
    random.Random(seed).shuffle(groups)
    n_val, n_test = max(1, round(len(groups) * val_ratio)), max(1, round(len(groups) * test_ratio))
    if n_val + n_test >= len(groups):
        raise ValueError("The split leaves no training batches")
    group_splits = {group: ("val" if i < n_val else "test" if i < n_val + n_test else "train")
                    for i, group in enumerate(groups)}
    return {image["id"]: group_splits[Path(image["file_name"]).parts[0]] for image in images}


def write_dataset_yaml(output, names):
    """Point YOLO at the current prepared location, including after a move."""
    output = Path(output)
    path = output / "dataset.yaml"
    yaml = f"path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {int(key)}: {json.dumps(name)}\n"
                    for key, name in sorted(names.items(), key=lambda item: int(item[0])))
    if not path.is_file() or path.read_text(encoding="utf-8") != yaml:
        path.write_text(yaml, encoding="utf-8")
    return path


def prepared_files_exist(output, records):
    """Check each split with one listing instead of a remote stat per file."""
    expected = defaultdict(set)
    for record in records:
        for key in ("image", "label"):
            path = Path(record[key])
            expected[path.parent].add(path.name)
    for directory, names in expected.items():
        try:
            with os.scandir(output / directory) as entries:
                existing = {entry.name for entry in entries if entry.is_file()}
        except (FileNotFoundError, NotADirectoryError):
            return False
        if not names.issubset(existing):
            return False
    return True


def prepare(source=SOURCE, output=PREPARED, seed=42, max_side=1280, val_ratio=0.15, test_ratio=0.15):
    source, output = Path(source).resolve(), Path(output).resolve()
    annotation_path = source / "annotations.json"
    payload = annotation_path.read_bytes()
    coco = json.loads(payload)
    configuration = {"version": 1, "annotation_sha256": hashlib.sha256(payload).hexdigest(),
                     "source": str(source), "seed": seed, "max_side": max_side,
                     "val_ratio": val_ratio, "test_ratio": test_ratio, "split_method": "source_batch"}
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The absolute source location is metadata: moving identical data between
        # Windows and Databricks does not change the split or generated labels.
        saved_settings = {key: value for key, value in previous["configuration"].items() if key != "source"}
        requested_settings = {key: value for key, value in configuration.items() if key != "source"}
        if saved_settings != requested_settings:
            changes = "; ".join(
                f"{key}: saved={saved_settings.get(key)!r}, requested={requested_settings.get(key)!r}"
                for key in sorted(saved_settings.keys() | requested_settings.keys())
                if saved_settings.get(key) != requested_settings.get(key))
            raise ValueError(
                f"Prepared data uses different settings in {manifest_path}: {changes}. "
                "Choose another --output directory (or CONFIG.prepared).")
        if prepared_files_exist(output, previous["images"]):
            data = write_dataset_yaml(output, previous["class_names"])
            print(f"Reusing prepared data: {data}", flush=True)
            return data
    if max_side < 32:
        raise ValueError("max_side must be at least 32")
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    class_ids = {category["id"]: index for index, category in enumerate(categories)}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    splits = split_by_batch(coco["images"], seed, val_ratio, test_ratio)
    records, issues = [], []
    for index, image in enumerate(coco["images"]):
        source_path = (source / image["file_name"]).resolve()
        if not source_path.is_relative_to(source):
            raise ValueError(f"Image path escapes source: {image['file_name']}")
        split = splits[image["id"]]
        stem = f"{image['id']:06d}"
        image_path = output / "images" / split / f"{stem}.jpg"
        label_path = output / "labels" / split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as raw:
            # TACO coordinates use the upright image. Bake EXIF rotation into a copy.
            rgb = ImageOps.exif_transpose(raw).convert("RGB")
            if rgb.size != (image["width"], image["height"]):
                raise ValueError(f"Annotation canvas mismatch for {source_path}: {rgb.size}")
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            rgb.save(image_path, quality=92)
        labels = []
        for annotation in annotations[image["id"]]:
            if annotation.get("iscrowd", 0):
                issues.append({"annotation_id": annotation["id"], "reason": "crowd annotation skipped"})
                continue
            try:
                box = normalized_box(annotation["bbox"], image["width"], image["height"])
            except ValueError as exc:
                issues.append({"annotation_id": annotation["id"], "reason": str(exc)})
                continue
            x, y, w, h = annotation["bbox"]
            if x < 0 or y < 0 or x + w > image["width"] or y + h > image["height"]:
                issues.append({"annotation_id": annotation["id"], "reason": "box clipped to canvas"})
            class_id = class_ids[annotation["category_id"]]
            labels.append((class_id, *box))
        label_path.write_text("".join(f"{row[0]} " + " ".join(f"{v:.8f}" for v in row[1:]) + "\n" for row in labels), encoding="utf-8")
        records.append({"image_id": image["id"], "source_file": image["file_name"], "split": split,
                        "image": image_path.relative_to(output).as_posix(), "label": label_path.relative_to(output).as_posix(),
                        "class_ids": [row[0] for row in labels]})
        if (index + 1) % 100 == 0:
            print(f"Prepared {index + 1}/{len(coco['images'])} images", flush=True)
    names = {i: category["name"] for i, category in enumerate(categories)}
    data = write_dataset_yaml(output, names)
    split_counts = Counter(row["split"] for row in records)
    class_counts = {split: Counter(cls for row in records if row["split"] == split for cls in row["class_ids"])
                    for split in ("train", "val", "test")}
    write_json(manifest_path, {"configuration": configuration, "class_names": names, "image_counts": dict(split_counts),
                               "class_counts": class_counts, "issues": issues, "images": records})
    print(f"Prepared {dict(split_counts)}; {len(issues)} annotation notes. {output / 'dataset.yaml'}", flush=True)
    return data


def load_yolo(weights, threads=4):
    try:
        import torch
        from ultralytics import YOLO, settings
    except ImportError as exc:
        raise SystemExit("Install the detector dependencies: uv pip install --python .venv/Scripts/python.exe -r requirements-yolo.txt") from exc
    torch.set_num_threads(threads)
    settings.update({"sync": False, "mlflow": False, "tensorboard": True,
                     "datasets_dir": str(STORAGE / "data/yolo"),
                     "weights_dir": str(STORAGE / "data/yolo/weights"), "runs_dir": str(RUNS)})
    return YOLO(str(weights))


def resolve_weights(value):
    if value in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"):
        path = STORAGE / "data/yolo/weights" / value
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return Path(value).resolve()


def check_class_names(names, manifest):
    expected = {int(key): value for key, value in manifest["class_names"].items()}
    if names != expected:
        raise ValueError("These weights do not have this dataset's TACO classes. Use a TACO-trained best.pt checkpoint.")


def train(args):
    data = Path(args.data).resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Run prepare first. Missing dataset configuration: {data}")
    model = load_yolo(resolve_weights(args.weights), args.threads)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
                project=str(args.project), name=args.name, exist_ok=False, patience=args.patience,
                fraction=args.fraction, freeze=args.freeze, optimizer="AdamW", lr0=args.lr,
                cos_lr=True, close_mosaic=min(10, args.epochs), plots=True, cache=False,
                amp=args.device != "cpu")
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights/best.pt"
    write_json(run_dir / "experiment.json", {"training_data": str(data), "best_weights": str(best),
               "epochs_requested": args.epochs, "training_fraction": args.fraction,
               "note": "Use validation to tune settings; reserve the test split for final evaluation."})
    print(f"Best weights: {best}", flush=True)
    return best


def evaluate(args):
    model = load_yolo(Path(args.weights).resolve(), args.threads)
    manifest_path = Path(args.data).resolve().parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest:
        check_class_names(model.names, manifest)
    locations = {}
    model.add_callback("on_val_end", lambda validator: locations.update(output=Path(validator.save_dir)))
    metrics = model.val(data=str(Path(args.data).resolve()), split=args.split, imgsz=args.imgsz,
                        batch=args.batch, device=args.device, workers=0, conf=0.001,
                        project=str(args.project), name=args.name, plots=True)
    output = locations["output"]
    report = {"split": args.split, "weights": str(Path(args.weights).resolve()),
              "averaging": "Mean over classes with ground-truth objects in this split; precision/recall use the mean-F1 operating point.",
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
              "speed_ms_per_image": metrics.speed, "per_class": []}
    # ap_class_index includes only classes with ground-truth objects in this split.
    # Report absent classes as unavailable, rather than assigning a misleading AP.
    indices = [int(i) for i in metrics.box.ap_class_index]
    report["evaluated_classes"] = len(indices)
    report["total_model_classes"] = len(model.names)
    counts = Counter()
    if manifest:
        counts.update(int(cls) for row in manifest["images"] if row["split"] == args.split for cls in row["class_ids"])
    for class_id, name in model.names.items():
        row = {"class_id": int(class_id), "class": name, "objects": counts[class_id],
               "precision": None, "recall": None, "AP50": None, "AP50_95": None}
        if class_id in indices:
            p, r, ap50, ap = metrics.box.class_result(indices.index(class_id))
            row.update(precision=float(p), recall=float(r), AP50=float(ap50), AP50_95=float(ap))
        report["per_class"].append(row)
    write_json(output / "metrics.json", report)
    with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["per_class"][0]))
        writer.writeheader()
        writer.writerows(report["per_class"])
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2), flush=True)
    print(f"Evaluation artifacts: {output}", flush=True)
    return output


def intersection_over_union(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def match_detections(predictions, ground_truth, iou=0.5):
    """Greedy confidence-order, same-class, one-to-one matching for an example."""
    used, matched = set(), []
    for prediction in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        candidates = [(intersection_over_union(prediction["xyxy"], target["xyxy"]), i)
                      for i, target in enumerate(ground_truth)
                      if i not in used and prediction["class_id"] == target["class_id"]]
        overlap, index = max(candidates, default=(0.0, -1))
        correct = index >= 0 and overlap >= iou
        if correct:
            used.add(index)
        matched.append({**prediction, "correct_at_iou50": correct, "matched_iou": overlap if correct else None})
    tp = len(used)
    return matched, {"true_positives": tp, "false_positives": len(predictions) - tp,
                     "false_negatives": len(ground_truth) - tp, "iou_threshold": iou}


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_panel(image, boxes, names, title, color, subtitle):
    image = image.copy()
    original_width, original_height = image.size
    image.thumbnail((780, 780))
    sx, sy = image.width / original_width, image.height / original_height
    canvas = Image.new("RGB", (820, 900), "#f2f5f8")
    offset_x, offset_y = (820 - image.width) // 2, 100 + (780 - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), title, fill="#15253a", font=font(26))
    draw.text((22, 58), subtitle, fill="#334155", font=font(17))
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        rect = (offset_x + x1 * sx, offset_y + y1 * sy, offset_x + x2 * sx, offset_y + y2 * sy)
        draw.rectangle(rect, outline=color, width=3)
        label = names[box["class_id"]]
        if "confidence" in box:
            label += f" {box['confidence']:.2f}"
        label_font = font(17)
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0] + 8
        tx, ty = max(0, min(rect[0], 820 - text_width)), max(95, rect[1] - 24)
        draw.rectangle((tx, ty, tx + text_width, ty + 24), fill=color)
        draw.text((tx + 4, ty + 2), label, font=label_font, fill="white")
    return canvas


def predict(args):
    model = load_yolo(Path(args.weights).resolve(), args.threads)
    data_root = Path(args.data).resolve().parent
    manifest = json.loads((data_root / "split_manifest.json").read_text(encoding="utf-8"))
    check_class_names(model.names, manifest)
    records = [r for r in manifest["images"] if r["split"] == args.split]
    if args.image_id is not None:
        records = [r for r in records if r["image_id"] == args.image_id]
        if not records:
            raise ValueError(f"Image {args.image_id} is not in the {args.split} split")
    elif args.source:
        records = [{"image_id": None, "source_file": str(args.source), "image": str(Path(args.source).resolve()), "label": None}]
    else:
        # Fixed examples, selected without consulting predictions or the test split.
        records = sorted(records, key=lambda r: r["image_id"])[:args.count]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for row in records:
        image_path = data_root / row["image"]
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        result = model.predict(image, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        predictions = [{"class_id": int(box.cls.item()), "confidence": float(box.conf.item()),
                        "xyxy": [float(v) for v in box.xyxy[0].tolist()]} for box in result.boxes]
        targets = []
        if row["label"]:
            for line in (data_root / row["label"]).read_text().splitlines():
                cls, cx, cy, w, h = map(float, line.split())
                targets.append({"class_id": int(cls), "xyxy": [(cx - w / 2) * image.width, (cy - h / 2) * image.height,
                                                                                (cx + w / 2) * image.width, (cy + h / 2) * image.height]})
        counts = None
        if row["label"]:
            predictions, counts = match_detections(predictions, targets)
        left = render_panel(image, targets, model.names, "Ground truth" if row["label"] else "Input image", "#15955b",
                            f"{len(targets)} annotated objects" if row["label"] else "No ground truth supplied")
        right = render_panel(image, predictions, model.names, "YOLO predictions", "#c64d2b", f"confidence >= {args.conf:.2f} | TP {counts['true_positives']}  FP {counts['false_positives']}  FN {counts['false_negatives']}" if row["label"] else f"confidence >= {args.conf:.2f} | labels unavailable")
        comparison = Image.new("RGB", (1640, 952), "#e5ebf1")
        comparison.paste(left, (0, 0))
        comparison.paste(right, (820, 0))
        ImageDraw.Draw(comparison).text((22, 916), f"{row['source_file']} | {args.split if row['label'] else 'external image'} | {Path(args.weights).name}", fill="#334155", font=font(19))
        stem = f"image_{row['image_id']:06d}" if row["image_id"] is not None else image_path.stem
        comparison.save(output / f"{stem}.jpg", quality=94)
        write_json(output / f"{stem}.json", {"image_id": row["image_id"], "source_file": row["source_file"],
                   "weights": str(Path(args.weights).resolve()), "confidence_threshold": args.conf,
                   "ground_truth": targets if row["label"] else None, "predictions": predictions, "example_metrics": counts})
        print(f"Saved {output / (stem + '.jpg')}", flush=True)


def run_experiment(config=None):
    """Prepare, train, validate, and save example predictions from one configuration."""
    config = config if config is not None else DEFAULT_EXPERIMENT
    if config.epochs < 1 or config.batch < 1 or config.imgsz < 32:
        raise ValueError("epochs/batch must be positive and imgsz must be at least 32")
    if not 0 < config.fraction <= 1 or not 0 <= config.conf <= 1 or config.prediction_count < 1:
        raise ValueError("fraction must be in (0, 1], conf in [0, 1], and prediction_count positive")
    if config.patience < 0 or config.lr <= 0 or config.threads < 1 or config.workers < 0:
        raise ValueError("patience/workers must be nonnegative; lr/threads must be positive")
    source = Path(config.source).expanduser().resolve()
    if not (source / "annotations.json").is_file():
        raise FileNotFoundError(f"Missing {source / 'annotations.json'}. Set source to the raw TACO folder, including its image batches.")

    device = str(config.device)
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}; requested epochs: {config.epochs}", flush=True)
    prepared = Path(config.prepared).expanduser().resolve()
    data = prepare(source, prepared, config.seed, config.max_side, config.val_ratio, config.test_ratio)
    arguments = asdict(config)
    arguments.update(source=source, prepared=prepared, data=data, device=device,
                     project=Path(config.project).expanduser().resolve())
    best = Path(train(argparse.Namespace(**arguments)))
    run_dir = best.parent.parent
    # Follow the trainer's actual directory, including any automatically added suffix.
    write_json(run_dir / "run_config.json", {key: str(value) if isinstance(value, Path) else value
                                           for key, value in arguments.items()})
    evaluation_dir = evaluate(argparse.Namespace(
        data=data, weights=best, imgsz=config.imgsz, batch=config.batch,
        device=device, threads=config.threads, split="val", project=run_dir, name="validation"))
    predictions_dir = run_dir / "predictions"
    predict(argparse.Namespace(
        data=data, weights=best, imgsz=config.imgsz, device=device, threads=config.threads,
        split="val", image_id=None, source=None, count=config.prediction_count,
        conf=config.conf, output=predictions_dir))
    tensorboard_dir = prepare_tensorboard_logs(run_dir)
    artifacts = {"run_dir": str(run_dir), "best_weights": str(best), "data": str(data),
                 "evaluation_dir": str(evaluation_dir), "predictions_dir": str(predictions_dir),
                 "tensorboard_dir": str(tensorboard_dir)}
    write_json(run_dir / "artifacts.json", artifacts)
    print(f"Experiment complete. Results: {run_dir}", flush=True)
    return artifacts


def prepare_tensorboard_logs(run_dir):
    """Add data images and backfill older CSV-only runs without retraining."""
    run_dir = Path(run_dir).expanduser().resolve()
    event_path = run_dir / "events.out.tfevents.taco"
    if event_path.is_file():
        return run_dir
    native_events = any(run_dir.glob("events.out.tfevents.*"))
    csv_path = run_dir / "results.csv"
    if not native_events and not csv_path.is_file():
        raise FileNotFoundError(f"No TensorBoard events or results.csv in {run_dir}")

    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    # Close the writer before copying: volume files do not support append/seek.
    with tempfile.TemporaryDirectory(prefix="taco_tensorboard_") as temporary:
        with SummaryWriter(temporary) as writer:
            if not native_events:
                with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        row = {key.strip(): value for key, value in row.items() if key is not None}
                        step = int(float(row.pop("epoch")))
                        for tag, value in row.items():
                            try:
                                value = float(value)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(value):
                                writer.add_scalar(tag, value, step)
            images = [run_dir / "labels.jpg", *sorted(run_dir.glob("train_batch*.jpg"))[:3],
                      *sorted((run_dir / "predictions").glob("*.jpg"))[:4]]
            for path in images:
                if path.is_file():
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        rgb.thumbnail((1280, 1280))
                        writer.add_image(f"data/{path.stem}", np.asarray(rgb), 0, dataformats="HWC")
        event_file = next(Path(temporary).glob("events.out.tfevents.*"))
        shutil.copyfile(event_file, event_path)
    print(f"TensorBoard logs: {run_dir}", flush=True)
    return run_dir


def display_tensorboard(run_dir):
    """Open TensorBoard inline in an active Databricks or Jupyter notebook."""
    from shlex import quote
    from tensorboard import notebook

    log_dir = prepare_tensorboard_logs(run_dir)
    return notebook.start(f"--logdir {quote(str(log_dir))} --port 6006 --load_fast=false")


def display_results(run_dir, table_display=None):
    """Display saved metrics and plots in Databricks/Jupyter without retraining."""
    import pandas as pd
    from IPython.display import Image as NotebookImage, display

    run_dir = Path(run_dir)
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    evaluation_dir = Path(artifacts["evaluation_dir"])
    predictions_dir = Path(artifacts["predictions_dir"])
    report = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    show_table = table_display if table_display is not None else display
    print("Validation metrics (0 to 1; higher is better)")
    show_table(pd.DataFrame([{key: report[key] for key in ("precision", "recall", "mAP50", "mAP50_95")}]))
    print("Per-class validation metrics; blank values indicate no evaluation support")
    show_table(pd.read_csv(evaluation_dir / "per_class.csv"))
    plots = [run_dir / "results.png", evaluation_dir / "confusion_matrix_normalized.png",
             *sorted(evaluation_dir.glob("*PR_curve.png"))]
    for path in plots:
        if path.is_file():
            print(path.name)
            display(NotebookImage(filename=str(path)))
    print("Ground truth (left) and predictions (right)")
    for path in sorted(predictions_dir.glob("*.jpg")):
        display(NotebookImage(filename=str(path)))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return run_experiment(DEFAULT_EXPERIMENT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Convert COCO boxes, orient copies, and split by batch")
    prep.add_argument("--source", type=Path, default=SOURCE)
    prep.add_argument("--output", type=Path, default=PREPARED)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--max-side", type=int, default=1280)
    prep.add_argument("--val-ratio", type=float, default=0.15)
    prep.add_argument("--test-ratio", type=float, default=0.15)
    for name in ("train", "evaluate", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", type=Path, default=PREPARED / "dataset.yaml")
        sub.add_argument("--weights", default="yolo11n.pt" if name == "train" else str(RUNS / "taco/weights/best.pt"))
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--device", default="cpu", help="cpu, or 0 for the first CUDA GPU")
        sub.add_argument("--threads", type=int, default=4)
        if name == "train":
            sub.add_argument("--project", type=Path, default=RUNS)
            sub.add_argument("--epochs", type=int, default=60)
            sub.add_argument("--batch", type=int, default=8)
            sub.add_argument("--workers", type=int, default=0)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--patience", type=int, default=15)
            sub.add_argument("--fraction", type=float, default=1.0)
            sub.add_argument("--freeze", type=int, default=0, help="Number of initial layers to freeze")
            sub.add_argument("--lr", type=float, default=0.001)
            sub.add_argument("--name", default="taco")
        elif name == "evaluate":
            sub.add_argument("--project", type=Path, default=RUNS)
            sub.add_argument("--split", choices=("val", "test"), default="val")
            sub.add_argument("--batch", type=int, default=8)
            sub.add_argument("--name", default="evaluation")
        else:
            sub.add_argument("--split", choices=("train", "val", "test"), default="val")
            sub.add_argument("--image-id", type=int)
            sub.add_argument("--source", type=Path, help="An external image; no ground-truth comparison is available")
            sub.add_argument("--count", type=int, default=4)
            sub.add_argument("--conf", type=float, default=0.25)
            sub.add_argument("--output", type=Path, default=RUNS / "predictions")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.source, args.output, args.seed, args.max_side, args.val_ratio, args.test_ratio)
    elif args.command == "train":
        if args.epochs < 1 or args.batch < 1 or not 0 < args.fraction <= 1:
            parser.error("epochs/batch must be positive and fraction must be in (0, 1]")
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        if not 0 <= args.conf <= 1 or args.count < 1:
            parser.error("conf must be in [0, 1] and count must be positive")
        if args.source and args.image_id is not None:
            parser.error("Choose either --source or --image-id")
        predict(args)


if __name__ == "__main__":
    main()
