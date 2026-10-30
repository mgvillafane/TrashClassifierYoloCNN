# Databricks notebook source
# MAGIC %md
# MAGIC # TACO YOLO experiment
# MAGIC Import this file as a Databricks notebook. Upload `taco_yolo.py` as a regular
# MAGIC workspace file beside it, and put the complete raw TACO dataset in a writable
# MAGIC Unity Catalog volume. Attach GPU ML compute for GPU training; CPU also works.
# MAGIC Run the cells in order. Edit the configuration cell to change the experiment.

# COMMAND ----------

# MAGIC %pip install ultralytics==8.4.193 pandas ipython

# COMMAND ----------

# Use the runtime's PyTorch/CUDA installation. Configure paths after this restart.
dbutils.library.restartPython()

# COMMAND ----------

from pathlib import Path
import os
import sys

# CHANGE THIS to a directory in a volume you can read and write.
# Expected raw data: TACO_ROOT/data/raw/taco/annotations.json and batch_* folders.
TACO_ROOT = Path("/Volumes/<catalog>/<schema>/<volume>/trash_yolo")
CODE_DIR = Path.cwd()  # Directory containing the regular taco_yolo.py file.
if "<" in str(TACO_ROOT):
    raise ValueError("Set TACO_ROOT to your /Volumes/catalog/schema/volume/trash_yolo directory.")
if not (CODE_DIR / "taco_yolo.py").is_file():
    raise FileNotFoundError("Upload taco_yolo.py beside this notebook, or change CODE_DIR to its directory.")

# Keep data, downloaded weights, caches, and results in the selected volume.
os.environ["TACO_HOME"] = str(TACO_ROOT)
sys.path.insert(0, str(CODE_DIR))
from taco_yolo import ExperimentConfig, run_experiment, display_results

CONFIG = ExperimentConfig(
    source=TACO_ROOT / "data/raw/taco",
    prepared=TACO_ROOT / "data/yolo/taco",
    project=TACO_ROOT / "reports/taco_yolo",
    name="databricks_taco",
    weights="yolo11n.pt",  # Set a TACO best.pt path to fine-tune saved weights.
    epochs=60,
    imgsz=640,
    batch=8,
    device="auto",  # "0" forces the first CUDA GPU; "cpu" forces CPU.
    patience=20,  # 0 runs all requested epochs, without early stopping.
    lr=0.001,
    fraction=1.0,  # 1.0 uses every training image.
    freeze=0,
    workers=0,
    threads=4,
    seed=42,
    max_side=1280,
    val_ratio=0.15,
    test_ratio=0.15,
    prediction_count=8,
    conf=0.25,
)

# COMMAND ----------

# Prepares copies, trains, evaluates validation data, and saves comparisons.
# Raw images/annotations remain unchanged; test data is reserved for final scoring.
RESULTS = run_experiment(CONFIG)
print("Saved run:", RESULTS["run_dir"])

# COMMAND ----------

# Rerun only this cell to view the results again without training.
# To reopen an older experiment, replace RESULTS["run_dir"] with its saved path.
# Databricks' display provides interactive metric tables; images display inline.
display_results(RESULTS["run_dir"], table_display=display)
