# Databricks notebook source
# MAGIC %md
# MAGIC # Full SortWaste training with live TensorBoard
# MAGIC Fine-tune the preserved TACO YOLO26 Medium checkpoint on every official
# MAGIC SortWaste training image. Validate on the official validation split; reserve
# MAGIC test data. Stop after 8 epochs without improved validation fitness, or 40
# MAGIC epochs maximum. Publish closed event files and checkpoints every epoch.

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

for key, default in {
    "code_dir": str(Path.cwd()),
    "weights": "/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_trashnet_yolo26m_40epochs_2184af/weights/best.pt",
    "dataset_root": "/Volumes/workspace/default/waste_data/processed/yolo_classes",
    "output_root": "/Volumes/workspace/default/tacotrashdata/reports/sortwaste_yolo",
    "run_name": datetime.now(timezone.utc).strftime("sortwaste_yolo26m_40epochs_%Y%m%dT%H%M%SZ"),
    "source_sha256": "",
    "epochs": "40",
    "patience": "8",
}.items():
    dbutils.widgets.text(key, default)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "carton"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def partition_images(root, partition):
    expected = root / "images/sortwaste" / partition
    images = []
    for line in (root / f"sortwaste_{partition}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        listed = Path(line.strip())
        image = root / listed
        if ".." in listed.parts or not image.is_relative_to(expected):
            raise ValueError(f"Unexpected {partition} image path: {image}")
        images.append(image)
    if not images or len(images) != len(set(images)) or len(images) != len({p.name for p in images}):
        raise ValueError(f"Empty or duplicate {partition} image list")
    return images


def run_training(local, saved):
    import torch
    import yaml
    from torch.utils.tensorboard import SummaryWriter
    from ultralytics import YOLO, settings

    sys.path.insert(0, dbutils.widgets.get("code_dir"))
    from taco_yolo import publish_tensorboard_epoch, publish_training_checkpoints

    if not torch.cuda.is_available():
        raise RuntimeError("This job requires a serverless A10 GPU")
    settings.update({"mlflow": False, "tensorboard": False, "wandb": False})
    epochs, patience = (int(dbutils.widgets.get(key)) for key in ("epochs", "patience"))
    if not 1 <= patience <= epochs <= 40:
        raise ValueError("Expected 1 <= patience <= epochs <= 40")
    root = Path(dbutils.widgets.get("dataset_root")).resolve()
    source = Path(dbutils.widgets.get("weights")).resolve()
    names = yaml.safe_load((root / "e1_sortwaste.yaml").read_text(encoding="utf-8"))["names"]
    actual_names = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    if actual_names != CLASSES:
        raise ValueError(f"Unexpected SortWaste class order: {actual_names}")
    images = {partition: partition_images(root, partition) for partition in ("train", "val")}
    if {p.name for p in images["train"]} & {p.name for p in images["val"]}:
        raise ValueError("SortWaste train and validation splits overlap")
    weights = local / "YOLO_full_TACO.pt"
    shutil.copyfile(source, weights)
    source_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    expected_hash = dbutils.widgets.get("source_sha256")
    if expected_hash and source_hash != expected_hash:
        raise ValueError("Cloud initialization weights differ from the preserved TACO model")
    config = {
        "source_checkpoint": str(source), "source_sha256": source_hash,
        "dataset_root": str(root), "classes": CLASSES,
        "train_images": len(images["train"]), "val_images": len(images["val"]),
        "epochs": epochs, "patience": patience, "image_size": 640, "batch_size": 8,
        "seed": 42, "optimizer": "AdamW", "learning_rate": 0.001,
        "gpu": torch.cuda.get_device_name(0), "test_partition_used": False,
        "checkpoint_selection": "validation fitness (Ultralytics)",
    }
    write_json(saved / "run_config.json", config)
    # A closed startup event makes configuration visible before epoch one.
    with SummaryWriter(str(local / "startup_events")) as writer:
        writer.add_text("configuration", json.dumps(config, indent=2), 0)
    event = next((local / "startup_events").glob("events.out.tfevents.*"))
    shutil.copyfile(event, saved / "events.out.tfevents.000000.startup")

    def copy_pair(item):
        partition, image = item
        relative = image.relative_to(root / "images")
        label = (root / "labels" / relative).with_suffix(".txt")
        shutil.copyfile(image, local / "data/images" / partition / image.name)
        shutil.copyfile(label, (local / "data/labels" / partition / image.name).with_suffix(".txt"))

    print("Staging all official train/validation images:", config, flush=True)
    for partition in images:
        for kind in ("images", "labels"):
            (local / "data" / kind / partition).mkdir(parents=True, exist_ok=True)
    items = [(partition, image) for partition, paths in images.items() for image in paths]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(copy_pair, items), 1):
            if index % 500 == 0:
                print(f"Staged {index}/{len(items)} image-label pairs", flush=True)
    data_yaml = local / "data/dataset.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(local / "data"), "train": "images/train", "val": "images/val", "names": CLASSES,
    }), encoding="utf-8")
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError("Initialization checkpoint must be an object detector")
    model.add_callback("on_fit_epoch_end", lambda trainer: publish_tensorboard_epoch(trainer, saved))
    model.add_callback("on_model_save", lambda trainer: publish_training_checkpoints(trainer, saved))
    model.add_callback("on_train_end", lambda trainer: publish_training_checkpoints(trainer, saved, finished=True))
    write_json(saved / "status.json", {"state": "training"})
    model.train(
        data=str(data_yaml), epochs=epochs, patience=patience,
        imgsz=640, batch=8, device="0", workers=0, seed=42, deterministic=True,
        optimizer="AdamW", lr0=0.001, warmup_epochs=3, amp=False,
        pretrained=True, resume=False, single_cls=False,
        val=True, save=True, plots=False, cache=False,
        project=str(local / "runs"), name="train", exist_ok=False,
    )
    trained = Path(model.trainer.save_dir)
    for artifact in trained.rglob("*"):
        if artifact.is_file():
            target = saved / artifact.relative_to(trained)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, target)
    result = {
        "state": "success", "run_dir": str(saved),
        "best_weights": str(saved / "weights/best.pt"),
        "last_weights": str(saved / "weights/last.pt"),
        "tensorboard_dir": str(saved),
        "metrics": {key: float(value) for key, value in model.trainer.metrics.items()},
    }
    write_json(saved / "artifacts.json", result)
    write_json(saved / "status.json", result)
    return result


run_name = dbutils.widgets.get("run_name")
if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
    raise ValueError("run_name must be a single directory name")
saved = Path(dbutils.widgets.get("output_root")) / run_name
saved.mkdir(parents=True, exist_ok=False)
write_json(saved / "status.json", {"state": "preparing"})
try:
    with tempfile.TemporaryDirectory(prefix="sortwaste_training_") as temporary:
        local = Path(temporary)
        os.environ["YOLO_CONFIG_DIR"] = str(local / "settings")
        os.environ["MPLCONFIGDIR"] = str(local / "matplotlib")
        RESULT = run_training(local, saved)
except Exception:
    write_json(saved / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
    write_json(saved / "training_status.json", {"training_finished": True, "result": "failed"})
    raise
dbutils.notebook.exit(json.dumps(RESULT))
