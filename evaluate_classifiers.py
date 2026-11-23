# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluate the C1–C4 classifiers on held-out data
# MAGIC Loads every `best.pt` from the classifier runs (`train_classifier.py`) and scores it once on
# MAGIC data never used for training or checkpoint selection:
# MAGIC
# MAGIC - the in-domain `test` splits of TrashNet, RealWaste and SortWaste
# MAGIC - all 3,627 TACO crops (external test), also without cigarettes, on the 3 classes shared with
# MAGIC   SortWaste, and per TACO fold (for the detector + classifier pipeline comparison)
# MAGIC
# MAGIC The model reported for each experiment and architecture is picked by the validation score stored
# MAGIC in its checkpoint, never by these results. Writes `metrics.csv`, `per_class.csv`,
# MAGIC `confusion.csv`, the paper tables and per-crop predictions to `out_dir`.
# MAGIC Run from this repository folder. GPU compute takes a few minutes; CPU works but is slower.

# COMMAND ----------

# MAGIC %pip install -q torch torchvision matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

dbutils.widgets.text("data_root", "/Volumes/workspace/default/waste_data/processed/classifier")
dbutils.widgets.text("runs_root", "/Volumes/workspace/default/models/classifier_experiments")
dbutils.widgets.text("out_dir", "/Volumes/workspace/default/models/classifier_eval")
DATA_ROOT = Path(dbutils.widgets.get("data_root"))
RUNS_ROOT = Path(dbutils.widgets.get("runs_root"))
OUT_DIR = Path(dbutils.widgets.get("out_dir"))

# Final runs listed in README.md (earlier incomplete attempts are left out).
RUNS = {
    "C1": ["20261119T042847Z_50587bf0", "20261119T054151Z_b672c674"],
    "C2": ["20261119T182605Z_7c529570", "20261119T185218Z_38ad8701"],
    "C3": ["20261119T061308Z_4adf9ebf"],
    "C4": ["20261120T034908Z_9bfaa9fb", "20261120T044952Z_e9b5ffc8"],
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "classifier.py").is_file() and (candidate / "classifier_eval.py").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from the repository folder (classifier.py not found)")


sys.path.insert(0, str(find_repo_root(Path.cwd())))
import classifier_eval as ce  # noqa: E402
from classifier import CLASSES, PREPROCESSING, build_model  # noqa: E402

checkpoints = sorted(RUNS_ROOT.glob("C*/*/*/best.pt"))
used = [p for p in checkpoints if p.parts[-3] in RUNS.get(p.parts[-4], [])]
for path in checkpoints:
    print("use " if path in used else "skip", path.relative_to(RUNS_ROOT))
missing = [f"{e}/{r}" for e, runs in RUNS.items() for r in runs if not (RUNS_ROOT / e / r).is_dir()]
if missing:
    raise FileNotFoundError(f"Run folders not found under {RUNS_ROOT}: {missing}")

# COMMAND ----------

# Decode every evaluation crop once; all models reuse the same resized images.
from PIL import Image  # noqa: E402

index = pd.read_csv(DATA_ROOT / "index.csv")
ce.check_index(index)
rows = ce.eval_rows(index)
display(pd.DataFrame({name: [int(mask.sum())] for name, mask in ce.eval_sets(rows).items()}, index=["crops"]))
resample = {"bilinear": Image.Resampling.BILINEAR}[PREPROCESSING["interpolation"]]
start = time.time()
images = ce.load_images(DATA_ROOT, rows["path"], PREPROCESSING["image_size"], resample)
print(f"{len(images)} crops loaded in {time.time() - start:.0f}s, {images.nbytes / 1e9:.1f} GB")

# COMMAND ----------

import torch  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor(PREPROCESSING["mean"], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(PREPROCESSING["std"], device=DEVICE).view(1, 3, 1, 1)
print("device:", DEVICE)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if tuple(config["classes"]) != CLASSES or config["preprocessing"] != PREPROCESSING:
        raise ValueError(f"{path}: class order or preprocessing differs from classifier.py")
    model = build_model(config["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(DEVICE).eval(), checkpoint


def predict(model, batch_size=256):
    predictions, confidences = [], []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size]).to(DEVICE).permute(0, 3, 1, 2).float()
            probabilities = model((x / PREPROCESSING["pixel_scale"] - MEAN) / STD).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            predictions.append(prediction.cpu())
            confidences.append(confidence.cpu())
    return torch.cat(predictions).numpy(), torch.cat(confidences).numpy()


STAGE = Path(tempfile.mkdtemp(prefix="classifier_eval_"))  # volumes dislike seek/append; copy at the end
(STAGE / "predictions").mkdir()
models, metrics, per_class, confusion = [], [], [], []
for path in used:
    experiment, run, architecture = path.parts[-4], path.parts[-3], path.parts[-2]
    model, checkpoint = load_model(path)
    model_info = dict(experiment=experiment, architecture=architecture, run=run)
    models.append({**model_info, "best_epoch": checkpoint["epoch"],
                   "selection_validation": checkpoint["config"]["selection_validation"],
                   "val_macro_f1": checkpoint["val_metrics"]["macro_f1"],
                   "val_accuracy": checkpoint["val_metrics"]["accuracy"], "checkpoint": str(path)})
    start = time.time()
    predictions, confidences = predict(model)
    m, c, cm = ce.score_model(rows, predictions, model_info)
    metrics.append(m)
    per_class.append(c)
    confusion.append(cm)
    rows[["path", "dataset", "partition", "original_label", "label_id"]].assign(
        pred_id=predictions, pred=[CLASSES[i] for i in predictions], confidence=confidences.round(4),
    ).to_csv(STAGE / "predictions" / f"{experiment}_{architecture}_{run}.csv.gz", index=False)
    taco = m.set_index("eval_set").loc["taco"]
    print(f"{experiment} {architecture:14s} {time.time() - start:5.0f}s | TACO macro-F1 {taco.macro_f1:.3f}")
    del model

models = pd.DataFrame(models)
models["selected"] = ce.mark_selected(models)
flags = models[["run", "architecture", "selected", "selection_validation", "val_macro_f1"]]
metrics = pd.concat(metrics).merge(flags, on=["run", "architecture"])
per_class = pd.concat(per_class).merge(flags, on=["run", "architecture"])
confusion = pd.concat(confusion).merge(flags[["run", "architecture", "selected"]], on=["run", "architecture"])

# COMMAND ----------

# Paper tables: selected model per experiment and architecture.
tables = ce.paper_tables(metrics)
tables["taco_per_class_f1"] = ce.per_class_table(per_class, "taco")
print("Checkpoint loading check: these should equal the stored validation macro-F1")
display(tables["val_reproduction"])
print("In-domain test (macro-F1 and accuracy)")
display(tables["in_domain_test"].reset_index())
print("TACO external test: all crops, without cigarettes, on cardboard/metal/plastic, per-fold mean/std")
display(tables["taco"].reset_index())
print("TACO per-class F1 (last row: number of crops)")
display(tables["taco_per_class_f1"].reset_index())

# COMMAND ----------

# Row-normalized TACO confusion matrices of the selected models.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

selected = models[models["selected"]].sort_values(["experiment", "architecture"])
fig, axes = plt.subplots(int(np.ceil(len(selected) / 4)), 4, figsize=(18, 4.5 * np.ceil(len(selected) / 4)),
                         squeeze=False)
for ax in axes.flat[len(selected):]:
    ax.axis("off")
for ax, model in zip(axes.flat, selected.itertuples()):
    cm = confusion[(confusion["run"] == model.run) & (confusion["architecture"] == model.architecture)
                   & (confusion["eval_set"] == "taco")]
    matrix = cm.pivot_table(index="true", columns="pred", values="count", fill_value=0).reindex(
        index=list(CLASSES), columns=list(CLASSES), fill_value=0)
    ax.imshow(matrix.div(matrix.sum(axis=1), axis=0), cmap="Blues", vmin=0, vmax=1)
    for (i, j), count in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, count, ha="center", va="center", fontsize=8)
    ax.set(title=f"{model.experiment} {model.architecture}", xticks=range(6), yticks=range(6),
           xlabel="predicted", ylabel="true")
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
fig.tight_layout()
fig.savefig(STAGE / "taco_confusion.png", dpi=150)
display(fig)

# COMMAND ----------

# Save everything to out_dir.
models.to_csv(STAGE / "models.csv", index=False)
metrics.to_csv(STAGE / "metrics.csv", index=False)
per_class.to_csv(STAGE / "per_class.csv", index=False)
confusion.to_csv(STAGE / "confusion.csv", index=False)
for name, table in tables.items():
    table.to_csv(STAGE / f"table_{name}.csv")
(STAGE / "info.json").write_text(json.dumps(dict(
    created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), data_root=str(DATA_ROOT), runs_root=str(RUNS_ROOT),
    device=str(DEVICE), eval_sets={k: int(v.sum()) for k, v in ce.eval_sets(rows).items()}), indent=2))
shutil.copytree(STAGE, OUT_DIR, dirs_exist_ok=True)
print("saved to", OUT_DIR)
