# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare training data from the raw datasets in the volumes
# MAGIC Reads RealWaste, TrashNet and SortWaste from `waste_data/raw` and TACO from `tacotrashdata/taco`
# MAGIC (read only), builds the unified 6-class manifest, checks it against the known counts, and
# MAGIC writes ready-to-train data to `out_dir`:
# MAGIC
# MAGIC - `classifier/`: one JPEG per example plus `index.csv` (fixed `label_id`; TACO is test only).
# MAGIC - `detector/`: single-class YOLO data with `d1_sortwaste.yaml` and `d2_taco_fold{0,1,2}.yaml`.
# MAGIC - `yolo_classes/`: the same images with class labels (6 classes + carton) for the end-to-end
# MAGIC   YOLO baselines, `e1_sortwaste.yaml` and `e2_taco_fold{0,1,2}.yaml`.
# MAGIC - `README.md`: layout, counts, rules and a PyTorch dataset snippet.
# MAGIC
# MAGIC `out_dir` defaults to the shared volume folder `waste_data/processed`, so it is built once and
# MAGIC reused by everyone (also from GPU jobs). The first full build writes ~100k small files and
# MAGIC takes a while; later runs see the finished parts (`_SUCCESS` marker) and skip them. A part
# MAGIC that was interrupted has no marker and is rebuilt. Only use `rebuild=yes` when the data or
# MAGIC mapping changed, and tell the team first. Set `out_dir=/tmp/processed` for a private copy on
# MAGIC the cluster's local disk. Run this notebook from the repository (it finds `configs/` and
# MAGIC `src/` above itself). CPU compute is enough. Use `datasets` to build only what you need, e.g.
# MAGIC `trashnet,realwaste` for C1-C3 avoids reading SortWaste's 13 GB of PNGs.

# COMMAND ----------

import os
import sys
from pathlib import Path

dbutils.widgets.text("raw_dir", "/Volumes/workspace/default/waste_data/raw")
dbutils.widgets.text("taco_dir", "/Volumes/workspace/default/tacotrashdata/taco")
dbutils.widgets.text("out_dir", "/Volumes/workspace/default/waste_data/processed")
dbutils.widgets.dropdown("parts", "classifier,detector,yolo_classes",
                         ["classifier,detector,yolo_classes", "classifier", "detector", "yolo_classes"])
dbutils.widgets.text("datasets", "all")  # or a comma list: realwaste,trashnet,sortwaste,taco
dbutils.widgets.dropdown("rebuild", "no", ["no", "yes"])

RAW_DIR = dbutils.widgets.get("raw_dir")
TACO_DIR = dbutils.widgets.get("taco_dir")
OUT_DIR = Path(dbutils.widgets.get("out_dir"))
PARTS = dbutils.widgets.get("parts").split(",")
DATASETS = [] if dbutils.widgets.get("datasets").strip() == "all" else \
    [d.strip() for d in dbutils.widgets.get("datasets").split(",") if d.strip()]
REBUILD = dbutils.widgets.get("rebuild") == "yes"


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "configs" / "label_mapping.json").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository (configs/ and src/ not found above it)")


REPO_ROOT = find_repo_root(Path.cwd())
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]
import build_manifest  # noqa: E402
import export_datasets  # noqa: E402

from data_preprocessing import DataPaths, check_raw_files  # noqa: E402

PATH_ARGS = ["--raw-dir", RAW_DIR, "--taco-dir", TACO_DIR, "--manifest-dir", str(OUT_DIR / "manifests")]
print("repo:", REPO_ROOT, "| output:", OUT_DIR)

# COMMAND ----------

# Are the raw files all there? Counts every dataset folder against the verified totals, so an
# unfinished unzip or upload shows up here (run this cell alone to just check the volumes).
check = check_raw_files(DataPaths(REPO_ROOT, Path(RAW_DIR), Path(TACO_DIR)))
display(check)
if not check["ok"].all():
    raise RuntimeError("Raw data incomplete: " + "; ".join(
        f"{r.what}: {r.found} of {r.expected} in {r.folder}" for r in check[~check["ok"]].itertuples()))

# COMMAND ----------

# Build and validate the manifest (reads labels and annotation files; fails loudly if a dataset
# is missing or its counts differ from the verified ones).
if REBUILD or not (OUT_DIR / "manifests/unified_manifest.csv").exists():
    build_manifest.main(PATH_ARGS + ["--no-figures"])
else:
    print("manifest exists; set rebuild=yes to rebuild it")

# COMMAND ----------

# Export crops / YOLO data. A finished part (marker lists its datasets) is reused when it already
# has the requested datasets; otherwise it is rebuilt with the union, so nobody's data is dropped.
ALL_DATASETS = ["realwaste", "trashnet", "sortwaste", "taco"]
done, todo = [], []
for part in PARTS:
    marker = OUT_DIR / part / export_datasets.DONE_MARKER
    have = set(marker.read_text().strip().split(",")) if marker.exists() else set()
    want = set(DATASETS or ALL_DATASETS)
    if not REBUILD and want <= have:
        done.append(part)
        continue
    datasets = sorted(want | have)
    todo.append((part, datasets))
    args = PATH_ARGS + ["--out", str(OUT_DIR), "--only", part, "--datasets", *datasets,
                        "--workers", str(max(1, (os.cpu_count() or 2) - 1))]
    export_datasets.main(args + (["--force"] if (OUT_DIR / part).exists() else []))
print("built:", todo or "nothing", "| reused:", done)

# COMMAND ----------

# Data per experiment (see README.md for the full rules).
import pandas as pd  # noqa: E402

# Classifier: train on `train`, select on `val`, test on in-domain `test` and on all TACO folds.
CLASSIFIER_EXPERIMENTS = {"C1": ["trashnet"], "C2": ["realwaste"], "C3": ["trashnet", "realwaste"],
                          "C4": ["trashnet", "realwaste", "sortwaste"]}
if (OUT_DIR / "classifier/index.csv").exists():
    index = pd.read_csv(OUT_DIR / "classifier/index.csv")
    have = set(index["dataset"])
    rows = []
    for name, datasets in CLASSIFIER_EXPERIMENTS.items():
        if not set(datasets) <= have:
            continue  # not exported (see the `datasets` widget)
        chosen = index[index["dataset"].isin(datasets)]
        rows.append({"experiment": name, "datasets": "+".join(datasets),
                     **{p: int((chosen["partition"] == p).sum()) for p in ["train", "val", "test"]},
                     "TACO test": int((index["dataset"] == "taco").sum())})
    display(pd.DataFrame(rows))

# YOLO: one YAML per experiment (D = single class, E = end-to-end with classes); images per role.
# D1/E1 on TACO fold k: model.val(data="d2_taco_fold<k>.yaml" or "e2_...", split="test"), the same
# test images as D2/E2 fold k.
rows = []
for yaml in sorted(p for part in ["detector", "yolo_classes"] for p in (OUT_DIR / part).glob("*.yaml")):
    detector = yaml.parent
    spec = dict(line.split(": ", 1) for line in yaml.read_text().splitlines() if ": " in line and not line.startswith(" "))
    count = lambda value: sum(len((detector / f).read_text().splitlines()) for f in value.strip("[]").split(", "))
    rows.append({"yaml": yaml.name, **{role: count(spec[role]) for role in ["train", "val", "test"]}})
if rows:
    display(pd.DataFrame(rows))
print("data root:", OUT_DIR)
