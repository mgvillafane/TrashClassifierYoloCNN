# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluate end-to-end YOLO vs YOLO boxes + classifier
# MAGIC Scores six systems with one set of rules (`src/model_experiments/detection_metrics.py`):
# MAGIC
# MAGIC | System | Finds the objects | Names the material |
# MAGIC |---|---|---|
# MAGIC | `e2e_sortwaste` | YOLO trained on SortWaste | the same YOLO |
# MAGIC | `pipeline_sortwaste` | same boxes | classifier trained on SortWaste (C5) |
# MAGIC | `e2e_all` | YOLO trained on SortWaste + TrashNet + RealWaste | the same YOLO |
# MAGIC | `pipeline_all` | same boxes | classifier trained on SortWaste + TrashNet + RealWaste (C4) |
# MAGIC | `pipeline_no_sortwaste` | same boxes | classifier trained on TrashNet + RealWaste (C3) |
# MAGIC | `reference_taco` | YOLO trained on TACO training batches | the same YOLO |
# MAGIC
# MAGIC Test sets: `taco_heldout` (177 TACO photos no model trained on, main), `taco_all` (1,500 TACO photos,
# MAGIC for the models that never saw TACO), `sortwaste_test` (776, in-domain check). Each YOLO predicts
# MAGIC every photo once (confidence >= 0.01, one box per object); the pipelines re-label exactly those
# MAGIC boxes. Outputs go to `out_dir`. Run on **serverless GPU**, after all three training runs finished.

# COMMAND ----------

# MAGIC %pip uninstall -y ultralytics opencv-python opencv-contrib-python

# COMMAND ----------

# MAGIC %pip install torch==2.9.1 torchvision==0.24.1 ultralytics-opencv-headless==8.4.193 opencv-python-headless==4.11.0.86 pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_ROOT = "/Volumes/workspace/default/models"
dbutils.widgets.text("data_root", "/Volumes/workspace/default/waste_data/processed")
dbutils.widgets.text("yolo_sortwaste", "latest")  # run folder, best.pt path, or "latest"
dbutils.widgets.text("yolo_all", "latest")
dbutils.widgets.text("classifier_sortwaste", "latest")  # C5 best.pt, or "latest"
dbutils.widgets.text("classifier_all", f"{MODELS_ROOT}/classifier_experiments/C4/20261120T034908Z_9bfaa9fb/resnet-layer4/best.pt")
dbutils.widgets.text("classifier_no_sortwaste", f"{MODELS_ROOT}/classifier_experiments/C3/20261119T061308Z_4adf9ebf/resnet-layer4/best.pt")
dbutils.widgets.text("yolo_taco_reference", "/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_trashnet_yolo26m_40epochs_2184af/weights/best.pt")
dbutils.widgets.text("out_dir", f"{MODELS_ROOT}/model_eval")
dbutils.widgets.text("bootstrap", "1000")

DATA_ROOT = Path(dbutils.widgets.get("data_root"))
YOLO_ROOT = DATA_ROOT / "yolo_classes"
OUT_DIR = Path(dbutils.widgets.get("out_dir"))
REPEATS = int(dbutils.widgets.get("bootstrap"))


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "classifier.py").is_file() and (candidate / "src/model_experiments").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from the repository folder")


REPO_ROOT = find_repo_root(Path.cwd())
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]
from classifier import CLASSES as CLASSIFIER_CLASSES, PREPROCESSING, build_model  # noqa: E402
from model_experiments import detection_metrics as dm  # noqa: E402
from model_experiments import yolo_data as yd  # noqa: E402


def latest(pattern_root: Path, glob: str) -> Path:
    found = sorted(pattern_root.glob(glob), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no {glob} under {pattern_root}")
    return found[-1]


def yolo_weights(value: str, stem: str) -> Path:
    if value == "latest":
        run = latest(Path(MODELS_ROOT) / "yolo_experiments", f"{stem}_*/weights/best.pt").parents[1]
    else:
        run = Path(value) if not value.endswith(".pt") else Path(value).parents[1]
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    if status.get("state") != "finished":
        raise RuntimeError(f"{run} has not finished training (status: {status.get('state')})")
    info = json.loads((run / "run_info.json").read_text())
    if info["yaml"] != f"{stem}.yaml":
        raise ValueError(f"{run} was trained on {info['yaml']}, expected {stem}.yaml")
    return run / "weights/best.pt"


WEIGHTS = {
    "yolo_sortwaste": yolo_weights(dbutils.widgets.get("yolo_sortwaste"), "yolo_sortwaste"),
    "yolo_all": yolo_weights(dbutils.widgets.get("yolo_all"), "yolo_sortwaste_trashnet_realwaste"),
    "yolo_taco_reference": Path(dbutils.widgets.get("yolo_taco_reference")),
}
value = dbutils.widgets.get("classifier_sortwaste")
CLASSIFIERS = {
    "classifier_sortwaste": latest(Path(MODELS_ROOT) / "classifier_experiments/C5", "*/resnet-layer4/best.pt")
    if value == "latest" else Path(value),
    "classifier_all": Path(dbutils.widgets.get("classifier_all")),
    "classifier_no_sortwaste": Path(dbutils.widgets.get("classifier_no_sortwaste")),
}
# system -> (YOLO, labeller); "yolo" = the YOLO's own label
SYSTEMS = {
    "e2e_sortwaste": ("yolo_sortwaste", "yolo"),
    "pipeline_sortwaste": ("yolo_sortwaste", "classifier_sortwaste"),
    "e2e_all": ("yolo_all", "yolo"),
    "pipeline_all": ("yolo_all", "classifier_all"),
    "pipeline_no_sortwaste": ("yolo_all", "classifier_no_sortwaste"),
    "reference_taco": ("yolo_taco_reference", "yolo"),
}
KNOWS_ALL_SIX = {"e2e_all", "pipeline_all", "pipeline_no_sortwaste", "reference_taco"}
TEST_SETS = {"taco_heldout": "taco_heldout.txt", "taco_all": "taco_all.txt", "sortwaste_test": "sortwaste_test.txt"}
# the TACO-trained reference saw the other TACO batches, so it is only scored on the held-out photos
SKIP = {("reference_taco", "taco_all"), ("reference_taco", "sortwaste_test")}
for name, path in {**WEIGHTS, **CLASSIFIERS}.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    print(f"{name:24s} {path}")

# COMMAND ----------

# Ground truth and local copies of the test photos (never used for training or selection).
LOCAL = Path(tempfile.mkdtemp(prefix="model_eval_"))
lists = {name: yd.read_images(YOLO_ROOT, [file]) for name, file in TEST_SETS.items()}
assert len(lists["taco_heldout"]) == 177 and len(lists["taco_all"]) == 1500 and len(lists["sortwaste_test"]) == 776
assert set(lists["taco_heldout"]) <= set(lists["taco_all"])
photos = sorted(set(lists["taco_all"]) | set(lists["sortwaste_test"]))
start = time.time()
yd.stage(YOLO_ROOT, photos, LOCAL)
print(f"{len(photos):,} test photos copied in {time.time() - start:.0f}s")


def ground_truth(rel: str):
    rows = [line.split() for line in (LOCAL / yd.label_for(rel)).read_text().splitlines() if line.strip()]
    labels = np.array([int(r[0]) for r in rows], int)
    cx, cy, w, h = (np.array([float(r[i]) for r in rows]) for i in range(1, 5))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).reshape(-1, 4), labels


GT = {rel: ground_truth(rel) for rel in photos}
print(pd.DataFrame({name: np.bincount(np.concatenate([GT[r][1] for r in rels]), minlength=7)
                    for name, rels in lists.items()}, index=dm.CLASSES).T)

# COMMAND ----------

# Predictions: each YOLO once per photo; its boxes are re-labelled by the classifiers that use them.
import torch  # noqa: E402
from PIL import Image  # noqa: E402

os.environ["YOLO_CONFIG_DIR"] = str(LOCAL / "settings")
from ultralytics import YOLO  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor(PREPROCESSING["mean"], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(PREPROCESSING["std"], device=DEVICE).view(1, 3, 1, 1)
HEIGHT, WIDTH = PREPROCESSING["image_size"]
RESAMPLE = {"bilinear": Image.Resampling.BILINEAR}[PREPROCESSING["interpolation"]]


def load_classifier(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if tuple(config["classes"]) != CLASSIFIER_CLASSES or config["preprocessing"] != PREPROCESSING:
        raise ValueError(f"{path}: class order or preprocessing differs from classifier.py")
    model = build_model(config["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(DEVICE).eval(), config


def classify(model, crops: np.ndarray, batch: int = 512):
    """Same preprocessing as training-time validation: resize to 224x224 (bilinear), ImageNet norm."""
    labels, confidences = [], []
    with torch.inference_mode():
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i:i + batch]).to(DEVICE).permute(0, 3, 1, 2).float()
            p = model((x / PREPROCESSING["pixel_scale"] - MEAN) / STD).softmax(dim=1)
            conf, lab = p.max(dim=1)
            labels.append(lab.cpu()), confidences.append(conf.cpu())
    if not labels:
        return np.zeros(0, int), np.zeros(0)
    return torch.cat(labels).numpy(), torch.cat(confidences).numpy()


def crops_of(rel: str, boxes: np.ndarray) -> np.ndarray:
    with Image.open(LOCAL / rel) as opened:
        image = opened.convert("RGB")
    w, h = image.size
    out = []
    for x1, y1, x2, y2 in boxes:
        left, upper = min(w - 1, max(0, math.floor(x1 * w))), min(h - 1, max(0, math.floor(y1 * h)))
        right, lower = max(left + 1, min(w, math.ceil(x2 * w))), max(upper + 1, min(h, math.ceil(y2 * h)))
        out.append(np.asarray(image.crop((left, upper, right, lower)).resize((WIDTH, HEIGHT), RESAMPLE)))
    return np.stack(out) if out else np.zeros((0, HEIGHT, WIDTH, 3), np.uint8)


PREDICTIONS = {}  # yolo name -> {rel: dict of arrays}
for yolo_name, weights in WEIGHTS.items():
    needed = lists["taco_heldout"] if yolo_name == "yolo_taco_reference" else photos
    local_weights = LOCAL / f"{yolo_name}.pt"
    shutil.copyfile(weights, local_weights)
    model = YOLO(str(local_weights))
    names = [model.names[i] for i in sorted(model.names)]
    if names != list(dm.CLASSES[:len(names)]) or len(names) not in (6, 7):
        raise ValueError(f"{yolo_name}: unexpected classes {names}")
    labellers = {SYSTEMS[s][1] for s in SYSTEMS if SYSTEMS[s][0] == yolo_name} - {"yolo"}
    classifiers = {name: load_classifier(CLASSIFIERS[name])[0] for name in labellers}
    start, result = time.time(), {}
    by_path = {os.path.realpath(LOCAL / rel): rel for rel in needed}
    for r in model.predict(source=[str(LOCAL / rel) for rel in needed], stream=True, imgsz=640, conf=0.01,
                           iou=0.7, max_det=100, agnostic_nms=True, device=0 if DEVICE.type == "cuda" else "cpu",
                           verbose=False):
        rel = by_path[os.path.realpath(r.path)]
        boxes = r.boxes.xyxyn.cpu().numpy().reshape(-1, 4)
        conf, cls = r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        keep = dm.agnostic_nms(boxes, conf)  # one box per object, whatever the model's own NMS did
        item = {"boxes": boxes[keep], "det_conf": conf[keep], "yolo_label": cls[keep], "yolo_score": conf[keep]}
        if classifiers:
            crops = crops_of(rel, item["boxes"])
            for name, classifier in classifiers.items():
                label, confidence = classify(classifier, crops)
                item[f"{name}_label"], item[f"{name}_score"] = label, item["det_conf"] * confidence
        result[rel] = item
    if set(result) != set(needed):
        raise RuntimeError(f"{yolo_name}: predictions for {len(result)} of {len(needed)} photos")
    PREDICTIONS[yolo_name] = result
    n_boxes = sum(len(v["boxes"]) for v in result.values())
    print(f"{yolo_name}: {len(result):,} photos, {n_boxes:,} boxes, {time.time() - start:.0f}s", flush=True)
    del model, classifiers
    torch.cuda.empty_cache()

# COMMAND ----------

# Score every system on every test set; 95% intervals by resampling photos (paired across systems).
rows_class, rows_summary, scored = [], [], {}
for test_set, rels in lists.items():
    weights = dm.bootstrap_weights(len(rels), REPEATS)
    for system, (yolo_name, labeller) in SYSTEMS.items():
        if (system, test_set) in SKIP:
            continue
        images = [{"gt_boxes": GT[r][0], "gt_labels": GT[r][1], **PREDICTIONS[yolo_name][r]} for r in rels]
        s = dm.Scored(images, f"{labeller}_label", f"{labeller}_score")
        for c, values in s.per_class().items():
            rows_class.append({"test_set": test_set, "system": system, "class": dm.CLASSES[c], **values})
        for scope, classes in (("shared3", dm.SHARED3), ("all6", dm.ALL6)):
            if scope == "all6" and (system not in KNOWS_ALL_SIX or test_set == "sortwaste_test"):
                continue  # SortWaste has only cardboard, metal and plastic
            point = s.summary(classes)
            samples = [s.summary(classes, w) for w in weights]
            scored[(test_set, system, scope)] = samples
            row = {"test_set": test_set, "system": system, "classes_scored": scope, **point}
            for key in ("map50", "macro_f1"):
                row[f"{key}_low"], row[f"{key}_high"] = dm.interval([x[key] for x in samples])
            rows_summary.append(row)
per_class = pd.DataFrame(rows_class)
summary = pd.DataFrame(rows_summary)

# Paired differences (same resampled photos for both systems)
COMPARISONS = [
    ("pipeline_sortwaste", "e2e_sortwaste", "classifier vs YOLO label, SortWaste training"),
    ("pipeline_all", "e2e_all", "classifier vs YOLO label, all three datasets"),
    ("pipeline_no_sortwaste", "e2e_all", "classifier without SortWaste vs YOLO label"),
    ("pipeline_no_sortwaste", "pipeline_all", "leaving SortWaste out of the classifier"),
    ("e2e_all", "e2e_sortwaste", "adding TrashNet + RealWaste to YOLO"),
    ("pipeline_all", "pipeline_sortwaste", "adding TrashNet + RealWaste to the pipeline"),
    ("reference_taco", "e2e_all", "TACO-trained reference vs YOLO on all three"),
]
rows_diff = []
for test_set in lists:
    for a, b, meaning in COMPARISONS:
        for scope in ("shared3", "all6"):
            if (test_set, a, scope) not in scored or (test_set, b, scope) not in scored:
                continue
            for key in ("map50", "macro_f1"):
                d = [x[key] - y[key] for x, y in zip(scored[(test_set, a, scope)], scored[(test_set, b, scope)])]
                point = summary.set_index(["test_set", "system", "classes_scored"])
                diff = point.loc[(test_set, a, scope), key] - point.loc[(test_set, b, scope), key]
                low, high = dm.interval(d)
                rows_diff.append({"test_set": test_set, "comparison": f"{a} - {b}", "meaning": meaning,
                                  "classes_scored": scope, "metric": key, "difference": diff, "low": low, "high": high,
                                  "clear": not (low <= 0 <= high)})
differences = pd.DataFrame(rows_diff)

# COMMAND ----------

pd.set_option("display.width", 250)


def table(test_set: str, scope: str) -> pd.DataFrame:
    s = summary[(summary["test_set"] == test_set) & (summary["classes_scored"] == scope)].set_index("system")
    classes = [dm.CLASSES[c] for c in (dm.SHARED3 if scope == "shared3" else dm.ALL6)]
    ap = per_class[per_class["test_set"] == test_set].pivot(index="system", columns="class", values="ap50")[classes]
    out = pd.DataFrame({
        "mean AP50": s["map50"].round(3).astype(str) + " [" + s["map50_low"].round(3).astype(str) + ", "
        + s["map50_high"].round(3).astype(str) + "]",
        "macro F1": s["macro_f1"].round(3).astype(str) + " [" + s["macro_f1_low"].round(3).astype(str) + ", "
        + s["macro_f1_high"].round(3).astype(str) + "]",
    }).join(ap.add_prefix("AP50 ").round(3))
    return out.reindex([x for x in SYSTEMS if x in out.index])


SCOPES = {t: list(summary.loc[summary["test_set"] == t, "classes_scored"].unique()) for t in lists}
for test_set in lists:
    for scope in SCOPES[test_set]:
        print(f"\n== {test_set}, {'cardboard/metal/plastic' if scope == 'shared3' else 'all six materials'}")
        display(table(test_set, scope).reset_index())
print("\n== Paired differences (clear = the 95% interval excludes 0)")
display(differences[differences["test_set"] == "taco_heldout"])

# COMMAND ----------

# Save tables and every prediction (so any further analysis needs no GPU).
STAGE = Path(tempfile.mkdtemp(prefix="model_eval_out_"))
per_class.to_csv(STAGE / "per_class.csv", index=False)
summary.to_csv(STAGE / "summary.csv", index=False)
differences.to_csv(STAGE / "differences.csv", index=False)
for test_set in lists:
    for scope in SCOPES[test_set]:
        table(test_set, scope).to_csv(STAGE / f"table_{test_set}_{scope}.csv")
for yolo_name, result in PREDICTIONS.items():
    rows = []
    for rel, item in result.items():
        for i in range(len(item["boxes"])):
            rows.append({"photo": rel, **dict(zip(["x1", "y1", "x2", "y2"], item["boxes"][i].round(5))),
                         **{k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in item.items() if k != "boxes"}})
    pd.DataFrame(rows).to_csv(STAGE / f"predictions_{yolo_name}.csv.gz", index=False)
(STAGE / "info.json").write_text(json.dumps({
    "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "weights": {k: str(v) for k, v in WEIGHTS.items()},
    "classifiers": {k: str(v) for k, v in CLASSIFIERS.items()}, "systems": SYSTEMS, "test_sets": TEST_SETS,
    "rules": {"iou_match": dm.IOU_MATCH, "f1_confidence": dm.F1_CONFIDENCE, "predict_confidence": 0.01,
              "max_boxes_per_photo": 100, "agnostic_nms_iou": 0.7, "carton": "ignored", "bootstrap": REPEATS},
}, indent=2))
shutil.copytree(STAGE, OUT_DIR, dirs_exist_ok=True)
print("saved to", OUT_DIR)
