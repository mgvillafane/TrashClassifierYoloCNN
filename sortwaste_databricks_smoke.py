# Databricks notebook source
# MAGIC %md
# MAGIC # SortWaste YOLO training smoke test
# MAGIC Fine-tune the saved TACO YOLO on a small, deterministic sample of the
# MAGIC processed seven-class SortWaste detection data. All data/cache/training
# MAGIC writes use temporary local storage; only run artifacts go to the volume.
# MAGIC The official test partition is not used. This verifies training, not quality.
# MAGIC Dependencies and A10 GPU compute are supplied by the job environment.

# COMMAND ----------

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import traceback


DEFAULT_WEIGHTS = (
    "/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/"
    "databricks_trashnet_yolo26m_40epochs_2184af/weights/best.pt"
)
DEFAULT_ROOT = "/Volumes/workspace/default/waste_data/processed/yolo_classes"
DEFAULT_OUTPUT = "/Volumes/workspace/default/tacotrashdata/reports/sortwaste_smoke"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "carton"]

for name, default in {
    "weights": DEFAULT_WEIGHTS,
    "dataset_root": DEFAULT_ROOT,
    "output_root": DEFAULT_OUTPUT,
    "run_name": datetime.now(timezone.utc).strftime("sortwaste_%Y%m%dT%H%M%SZ"),
    "train_images": "32",
    "val_images": "8",
    "source_sha256": "",
}.items():
    dbutils.widgets.text(name, default)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def split_images(root, partition):
    list_path = root / f"sortwaste_{partition}.txt"
    images = []
    expected = root / "images/sortwaste" / partition
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        listed = Path(line.strip())
        if ".." in listed.parts:
            raise ValueError(f"Parent traversal in {list_path}: {listed}")
        # Validate list entries lexically. Resolving every remote image would do
        # thousands of volume metadata lookups before sampling just 40 images.
        image = root / listed
        if not image.is_relative_to(expected):
            raise ValueError(f"Unexpected {partition} image path: {image}")
        images.append(image)
    if len(images) != len(set(images)):
        raise ValueError(f"Duplicate images in {list_path}")
    return images


def stage_split(root, local, partition, selected):
    counts = Counter()
    for image in selected:
        relative = image.relative_to(root / "images")
        label = (root / "labels" / relative).with_suffix(".txt")
        # Fail on missing labels rather than silently treating them as backgrounds.
        rows = label.read_text(encoding="utf-8").splitlines()
        for row in rows:
            if not row.strip():
                continue
            values = [float(value) for value in row.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Invalid detection label: {label}")
            identifier, x, y, width, height = values
            if identifier != int(identifier) or not 0 <= identifier < len(CLASSES):
                raise ValueError(f"Invalid class ID in {label}")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid normalized box in {label}")
            counts[CLASSES[int(identifier)]] += 1
        destination = local / "images" / partition / image.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, destination)
        target_label = (local / "labels" / partition / image.name).with_suffix(".txt")
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(label, target_label)
    if not counts:
        raise ValueError(f"No annotated objects in the {partition} sample")
    return dict(counts)


def run_smoke():
    import torch
    import yaml
    from ultralytics import YOLO, settings

    if not torch.cuda.is_available():
        raise RuntimeError("The smoke test requires the job's serverless A10 GPU.")
    settings.update({"mlflow": False, "tensorboard": False, "wandb": False})
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    root = Path(dbutils.widgets.get("dataset_root")).resolve()
    source = Path(dbutils.widgets.get("weights")).resolve()
    run_name = dbutils.widgets.get("run_name")
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    print("Verifying TACO initialization checkpoint:", source, flush=True)
    source_hash = sha256(source)
    expected_hash = dbutils.widgets.get("source_sha256")
    if expected_hash and source_hash != expected_hash:
        raise ValueError("The cloud initialization checkpoint differs from YOLO_full_TACO.pt")
    source_yaml = root / "e1_sortwaste.yaml"
    configuration = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = configuration["names"]
    actual_names = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    if actual_names != CLASSES:
        raise ValueError(f"Unexpected processed dataset class order: {actual_names}")
    print("Reading the official SortWaste partition lists", flush=True)
    train = split_images(root, "train")
    val = split_images(root, "val")
    print("Official split sizes:", len(train), "train;", len(val), "validation", flush=True)
    if set(train) & set(val) or {path.name for path in train} & {path.name for path in val}:
        raise ValueError("The train and validation partitions overlap")
    selected = {}
    for partition, paths in (("train", train), ("val", val)):
        count = int(dbutils.widgets.get(f"{partition}_images"))
        if not 1 <= count <= 128 or count > len(paths):
            raise ValueError("Smoke-test sample sizes must be between 1 and 128")
        selected[partition] = random.Random(42).sample(sorted(paths), count)

    manifest = {
        "purpose": "one-epoch training smoke test, not a model-quality evaluation",
        "source_checkpoint": str(source),
        "source_sha256": source_hash,
        "dataset_yaml": str(source_yaml),
        "classes": CLASSES,
        "seed": 42,
        "epochs": 1,
        "image_size": 320,
        "batch_size": 4,
        "gpu": torch.cuda.get_device_name(0),
        "official_partition_sizes": {"train": len(train), "val": len(val)},
        "samples": {key: [str(path) for path in paths] for key, paths in selected.items()},
    }
    saved = Path(dbutils.widgets.get("output_root")) / run_name
    saved.mkdir(parents=True, exist_ok=False)
    write_json(saved / "run_config.json", manifest)
    write_json(saved / "status.json", {"state": "preparing"})
    try:
        with tempfile.TemporaryDirectory(prefix="sortwaste_smoke_") as temporary:
            local = Path(temporary)
            print("Staging the selected images and labels on local storage", flush=True)
            manifest["sample_box_counts"] = {
                key: stage_split(root, local / "data", key, paths)
                for key, paths in selected.items()
            }
            write_json(saved / "run_config.json", manifest)
            data_yaml = local / "data/dataset.yaml"
            data_yaml.write_text(yaml.safe_dump({
                "path": str(local / "data"),
                "train": "images/train", "val": "images/val", "names": CLASSES,
            }), encoding="utf-8")
            local_weights = local / "YOLO_full_TACO.pt"
            shutil.copyfile(source, local_weights)
            model = YOLO(str(local_weights))
            if model.task != "detect":
                raise ValueError(f"Expected a YOLO detector, got {model.task}")
            manifest["source_classes"] = model.names
            write_json(saved / "run_config.json", manifest)
            write_json(saved / "status.json", {"state": "training"})
            print("Starting one training epoch:", manifest["sample_box_counts"], flush=True)
            model.train(
                data=str(data_yaml), epochs=1, imgsz=320, batch=4, device="0",
                workers=0, seed=42, deterministic=True, optimizer="AdamW",
                lr0=0.001, warmup_epochs=0, amp=False, pretrained=True,
                resume=False, single_cls=False, val=True, save=True,
                plots=False, cache=False, mosaic=0.0, mixup=0.0,
                project=str(local / "runs"), name="train", exist_ok=False,
            )
            trained = Path(model.trainer.save_dir)
            with (trained / "results.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            if len(history) != 1:
                raise RuntimeError(f"Expected one completed training epoch, found {len(history)}")
            losses = {
                key.strip(): float(value) for key, value in history[-1].items()
                if key.strip().startswith("train/") and key.strip().endswith("_loss")
            }
            if not losses or not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError(f"Missing/non-finite training losses: {losses}")
            best = trained / "weights/best.pt"
            if not best.is_file() or not (trained / "weights/last.pt").is_file():
                raise RuntimeError("Training did not save best.pt and last.pt")
            reloaded = YOLO(str(best))
            if list(reloaded.names.values()) != CLASSES:
                raise RuntimeError(f"Trained checkpoint has incorrect class names: {reloaded.names}")
            image = next((local / "data/images/val").glob("*.jpg"))
            prediction = reloaded.predict(str(image), imgsz=320, device="0", verbose=False)[0]
            for artifact in trained.rglob("*"):
                if artifact.is_file():
                    target = saved / artifact.relative_to(trained)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(artifact, target)
            result = {
                "state": "success", "epochs_completed": len(history),
                "train_images": len(selected["train"]), "val_images": len(selected["val"]),
                "training_losses": losses,
                "metrics": {key: float(value) for key, value in model.trainer.metrics.items()},
                "checkpoint": str(saved / "weights/best.pt"),
                "checkpoint_sha256": sha256(best),
                "prediction_smoke_detections": len(prediction.boxes),
                "classes": CLASSES, "gpu": manifest["gpu"],
            }
            write_json(saved / "status.json", result)
            print(json.dumps(result, indent=2))
            return result
    except Exception:
        write_json(saved / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


with tempfile.TemporaryDirectory(prefix="sortwaste_settings_") as settings_directory:
    os.environ["YOLO_CONFIG_DIR"] = settings_directory
    os.environ["MPLCONFIGDIR"] = settings_directory
    RESULT = run_smoke()
dbutils.notebook.exit(json.dumps(RESULT))
