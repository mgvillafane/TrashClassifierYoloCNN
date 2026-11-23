# Databricks notebook source
# MAGIC %md
# MAGIC # Train the models for the YOLO experiments
# MAGIC Pick one model in the `model` widget and run all cells on **serverless GPU (A10)**. Run the
# MAGIC notebook once per model (three runs, they can run at the same time in separate sessions):
# MAGIC
# MAGIC | `model` | What it trains | Data (train / val, images or crops) |
# MAGIC |---|---|---|
# MAGIC | `yolo_sortwaste` | YOLO on SortWaste | 3,705 / 780 photos |
# MAGIC | `yolo_sortwaste_trashnet_realwaste` | YOLO on SortWaste + TrashNet + RealWaste (whole photo = one box) | 8,593 / 1,390 photos |
# MAGIC | `classifier_sortwaste` | ResNet-18 classifier on SortWaste crops only (experiment C5, class-balanced) | 48,193 crops / 750 |
# MAGIC
# MAGIC Rules the code enforces: every YOLO starts from the COCO-pretrained `yolo26m.pt`; no TACO photo is
# MAGIC in training or validation; each model is selected on the validation splits of its own training
# MAGIC datasets; test photos are never opened. Both YOLO runs use identical settings (those of the earlier
# MAGIC SortWaste run). Results are copied to `output_root` after every epoch, so an interrupted run keeps
# MAGIC its progress; to continue it, put its folder name in `resume_run` and run all again.
# MAGIC
# MAGIC Needs `notebooks/databricks_prepare_data` to have run (it builds the YOLO experiment data).

# COMMAND ----------

# MAGIC %pip uninstall -y ultralytics opencv-python opencv-contrib-python

# COMMAND ----------

# MAGIC %pip install torch==2.9.1 torchvision==0.24.1 ultralytics-opencv-headless==8.4.193 opencv-python-headless==4.11.0.86 scikit-learn matplotlib pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "yolo_sortwaste": "yolo_sortwaste.yaml",
    "yolo_sortwaste_trashnet_realwaste": "yolo_sortwaste_trashnet_realwaste.yaml",
    "classifier_sortwaste": None,
}
dbutils.widgets.dropdown("model", "yolo_sortwaste", list(MODELS))
dbutils.widgets.text("data_root", "/Volumes/workspace/default/waste_data/processed")
dbutils.widgets.text("weights", "/Volumes/workspace/default/tacotrashdata/yolo/weights/yolo26m.pt")
dbutils.widgets.text("output_root", "/Volumes/workspace/default/models/yolo_experiments")
dbutils.widgets.text("epochs", "40")
dbutils.widgets.text("patience", "8")
dbutils.widgets.text("workers", "0")  # data loader processes; 0 worked reliably on serverless GPU
dbutils.widgets.text("resume_run", "")  # folder name under output_root to continue; empty = new run

MODEL = dbutils.widgets.get("model")
DATA_ROOT = Path(dbutils.widgets.get("data_root"))
WEIGHTS = Path(dbutils.widgets.get("weights"))
OUTPUT_ROOT = Path(dbutils.widgets.get("output_root"))
EPOCHS, PATIENCE, WORKERS = (int(dbutils.widgets.get(k)) for k in ("epochs", "patience", "workers"))
RESUME_RUN = dbutils.widgets.get("resume_run").strip()


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "train_classifier.py").is_file() and (candidate / "src/model_experiments").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from the repository folder (train_classifier.py not found)")


REPO_ROOT = find_repo_root(Path.cwd())
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]
from model_experiments import yolo_data as yd  # noqa: E402

print("model:", MODEL, "| repo:", REPO_ROOT)

# COMMAND ----------

# Check the data before anything is copied or trained (fails loudly on any problem).
if MODEL.startswith("yolo"):
    YAML = MODELS[MODEL]
    YOLO_ROOT = DATA_ROOT / "yolo_classes"
    if not (YOLO_ROOT / "_SUCCESS_experiments").exists():
        raise FileNotFoundError(f"{YOLO_ROOT} has no YOLO experiment data; run notebooks/databricks_prepare_data first")
    SPEC = yd.read_spec(YOLO_ROOT / YAML)
    TRAIN = yd.read_images(YOLO_ROOT, SPEC["train"])
    VAL = yd.read_images(YOLO_ROOT, SPEC["val"])
    yd.check_dataset(YAML, SPEC, TRAIN, VAL, yd.read_images(YOLO_ROOT, yd.TEST_LISTS))
    print(f"{YAML}: train {len(TRAIN):,} photos from {SPEC['train']}, val {len(VAL):,} from {SPEC['val']}")
    print("classes:", SPEC["names"])
else:
    import train_classifier  # noqa: E402

    train_classifier.main(["C5", "--model", "resnet-layer4", "--data-root", str(DATA_ROOT / "classifier"),
                           "--dry-run"])

# COMMAND ----------

# Train.
if MODEL == "classifier_sortwaste":
    # Same trainer, settings and output layout as the C1-C4 classifiers (train_classifier.py):
    # /Volumes/workspace/default/models/classifier_experiments/C5/<run>/resnet-layer4/best.pt
    train_classifier.main(["C5", "--model", "resnet-layer4", "--data-root", str(DATA_ROOT / "classifier")])
else:
    LOCAL = Path(tempfile.gettempdir()) / "yolo_experiments"  # fixed paths, so a resumed run finds its data
    os.environ["YOLO_CONFIG_DIR"] = str(LOCAL / "settings")
    os.environ["MPLCONFIGDIR"] = str(LOCAL / "matplotlib")
    import torch  # noqa: E402
    import ultralytics  # noqa: E402
    from ultralytics import YOLO, settings  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU: attach serverless GPU compute (A10)")
    settings.update({"mlflow": False, "tensorboard": False, "wandb": False})

    run_name = RESUME_RUN or f"{Path(YAML).stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    saved = OUTPUT_ROOT / run_name
    if RESUME_RUN:
        if not (saved / "weights/last.pt").exists():
            raise FileNotFoundError(f"nothing to resume: {saved}/weights/last.pt")
        if json.loads((saved / "run_info.json").read_text())["yaml"] != YAML:
            raise ValueError(f"{RESUME_RUN} was trained on another dataset; pick the matching model")
    else:
        saved.mkdir(parents=True, exist_ok=False)

    data_dir = LOCAL / "data" / Path(YAML).stem
    print(f"copying {len(TRAIN) + len(VAL):,} photos and labels to {data_dir} ...", flush=True)
    data_yaml = yd.write_dataset_yaml(data_dir, yd.stage(YOLO_ROOT, TRAIN, data_dir),
                                      yd.stage(YOLO_ROOT, VAL, data_dir), SPEC["names"])

    # Identical for both YOLO runs; the same settings as the earlier SortWaste run.
    train_args = dict(epochs=EPOCHS, patience=PATIENCE, imgsz=640, batch=8, device=0, workers=WORKERS,
                      seed=42, deterministic=True, optimizer="AdamW", lr0=0.001, warmup_epochs=3, amp=False,
                      pretrained=True, single_cls=False, val=True, save=True, plots=False, cache=False,
                      project=str(LOCAL / "runs"), name=run_name, exist_ok=True)
    run_info = {
        "model": MODEL, "yaml": YAML, "run_name": run_name, "classes": SPEC["names"],
        "train_lists": SPEC["train"], "val_lists": SPEC["val"], "train_images": len(TRAIN), "val_images": len(VAL),
        "train_fingerprint": yd.fingerprint(TRAIN), "val_fingerprint": yd.fingerprint(VAL),
        "test_images_used": 0, "starting_weights": str(WEIGHTS), "starting_weights_sha256": yd.sha256_file(WEIGHTS),
        "checkpoint_selection": "Ultralytics best.pt (validation fitness on the val lists above)",
        "ultralytics": ultralytics.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "train_args": train_args, "started": datetime.now(timezone.utc).isoformat(),
    }
    if not RESUME_RUN:
        (saved / "run_info.json").write_text(json.dumps(run_info, indent=2))

    def publish(trainer):
        """Copy results and checkpoints to the volume after every epoch (training continues on failure)."""
        try:
            run_dir = Path(trainer.save_dir)
            for rel in ("results.csv", "args.yaml", "weights/last.pt", "weights/best.pt"):
                if (run_dir / rel).exists():
                    (saved / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run_dir / rel, saved / rel)
            (saved / "status.json").write_text(json.dumps({"state": "training", "epochs_done": trainer.epoch + 1}))
        except Exception as error:  # noqa: BLE001
            print("copy to volume failed (training continues):", error, flush=True)

    if RESUME_RUN:
        last = LOCAL / "runs" / run_name / "weights/last.pt"
        last.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(saved / "weights/last.pt", last)
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", publish)
        model.train(resume=True)
    else:
        local_weights = LOCAL / "yolo26m_coco.pt"
        local_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WEIGHTS, local_weights)
        model = YOLO(str(local_weights))
        yd.check_coco_weights(model.names)
        if model.task != "detect":
            raise ValueError("starting weights must be a detection model")
        model.add_callback("on_fit_epoch_end", publish)
        model.train(data=str(data_yaml), **train_args)

    run_dir = Path(model.trainer.save_dir)
    shutil.copytree(run_dir, saved, dirs_exist_ok=True)
    final = {"state": "finished", "best_weights": str(saved / "weights/best.pt"),
             "finished": datetime.now(timezone.utc).isoformat(),
             "final_val_metrics": {k: float(v) for k, v in (model.trainer.metrics or {}).items()}}
    (saved / "status.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    print("\nrun folder:", saved)
