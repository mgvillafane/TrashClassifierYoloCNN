# Databricks notebook source
# MAGIC %md
# MAGIC # Small TACO YOLO GPU experiment
# MAGIC Import this file as a Databricks notebook. Upload `taco_yolo.py` as a regular
# MAGIC workspace file beside it, and put the complete raw TACO dataset in a writable
# MAGIC Unity Catalog volume. The `taco_small_gpu` bundle job requests one A10 GPU.
# MAGIC **VS Code:** open the Databricks sidebar, expand Bundle Resource Explorer,
# MAGIC select `taco_small_gpu`, and click its play button (Deploy and Run).
# MAGIC Or open the Command Palette and choose **Databricks: Deploy the bundle and
# MAGIC run a resource**, then select `taco_small_gpu`.
# MAGIC The editor's Run File as Workflow command uses the selected compute; it does
# MAGIC not apply the GPU settings from the bundle job.
# MAGIC This short run uses 2 epochs, 320-pixel inputs, and 10% of training images.
# MAGIC Run the cells in order. Edit the configuration cell to change the experiment.
# MAGIC The final TensorBoard cell can also visualize a saved run without training.

# COMMAND ----------

# MAGIC %pip uninstall -y ultralytics opencv-python opencv-contrib-python

# COMMAND ----------

# MAGIC %pip install torch==2.9.1 torchvision==0.24.1 ultralytics-opencv-headless==8.4.193 opencv-python-headless==4.11.0.86 tensorboard==2.21.0 pandas ipython

# COMMAND ----------

# Configure paths after restarting with the installed notebook dependencies.
dbutils.library.restartPython()

# COMMAND ----------

from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "This execution has no CUDA GPU. In VS Code, open the Command Palette "
        "(Ctrl+Shift+P), choose 'Databricks: Deploy the bundle and run a resource', "
        "then select 'taco_small_gpu'. This launches the A10 GPU job defined in "
        "databricks.yml. The editor's 'Run File as Workflow' command does not "
        "apply that job's GPU settings. In the Databricks website, attach this "
        "notebook to Serverless GPU with the 1xA10 accelerator."
    )
print("Training GPU:", torch.cuda.get_device_name(0))

# CHANGE THIS to a directory in a volume you can read and write.
# Expected uploaded data: TACO_ROOT/taco/annotations.json and batch_* folders.
TACO_ROOT = Path("/Volumes/workspace/default/tacotrashdata")
dbutils.widgets.text("code_dir", str(Path.cwd()))
CODE_DIR = Path(dbutils.widgets.get("code_dir"))
if "<" in str(TACO_ROOT):
    raise ValueError("Set TACO_ROOT to your /Volumes/catalog/schema/volume/trash_yolo directory.")
if not (CODE_DIR / "taco_yolo.py").is_file():
    raise FileNotFoundError("Upload taco_yolo.py beside this notebook, or change CODE_DIR to its directory.")

# Keep data, downloaded weights, and caches in the selected volume.
# Stage outputs locally: Unity Catalog volumes do not support CSV append/seek.
os.environ["TACO_HOME"] = str(TACO_ROOT)
os.environ["YOLO_CONFIG_DIR"] = str(TACO_ROOT / "yolo/settings")
os.environ["MPLCONFIGDIR"] = str(TACO_ROOT / "yolo/matplotlib")
sys.path.insert(0, str(CODE_DIR))
from taco_yolo import ExperimentConfig, run_experiment, display_results
from ultralytics import settings

settings.update({"mlflow": False, "tensorboard": True})

LOCAL_RUN_ROOT = tempfile.TemporaryDirectory(prefix="taco_yolo_")
CONFIG = ExperimentConfig(
    source=TACO_ROOT / "taco",
    prepared=TACO_ROOT / "yolo/taco",
    project=Path(LOCAL_RUN_ROOT.name),
    name="databricks_small_gpu",
    weights=TACO_ROOT / "yolo/weights/yolo11n.pt",  # Reuse the uploaded pretrained weights.
    epochs=2,
    imgsz=320,
    batch=16,
    device="0",  # Require the first CUDA GPU.
    patience=0,  # Run both epochs.
    lr=0.001,
    fraction=0.1,  # Short demonstration; use 1.0 for the full training split.
    freeze=0,
    workers=0,
    threads=4,
    seed=42,
    max_side=1280,
    val_ratio=0.15,
    test_ratio=0.15,
    prediction_count=2,
    conf=0.25,
)

# COMMAND ----------

# Prepares copies, trains, evaluates validation data, and saves comparisons.
# Raw images/annotations remain unchanged; test data is reserved for final scoring.
RESULTS = run_experiment(CONFIG)

# Copy completed outputs into one persistent experiment folder.
run_dir = Path(RESULTS["run_dir"])
saved_dir = TACO_ROOT / "reports/taco_yolo" / run_dir.name
if (saved_dir / "artifacts.json").exists():
    from ultralytics.utils.files import increment_path
    saved_dir = increment_path(saved_dir)
for path in run_dir.rglob("*"):
    if path.is_file():
        target = saved_dir / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
RESULTS = {
    key: str(saved_dir / Path(value).relative_to(run_dir)) if Path(value).is_relative_to(run_dir) else value
    for key, value in RESULTS.items()
}
(saved_dir / "artifacts.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
print("Saved run:", RESULTS["run_dir"])

# COMMAND ----------

# Rerun only this cell to view the results again without training.
# To reopen an older experiment, replace RESULTS["run_dir"] with its saved path.
# Databricks' display provides interactive metric tables; images display inline.
display_results(RESULTS["run_dir"], table_display=display)

# Return the saved outputs and validation metrics to the workflow run page/API.
metrics = json.loads((Path(RESULTS["evaluation_dir"]) / "metrics.json").read_text())
dbutils.notebook.exit(json.dumps({
    **RESULTS,
    "gpu": torch.cuda.get_device_name(0),
    "metrics": {key: metrics[key] for key in ("precision", "recall", "mAP50", "mAP50_95")},
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## TensorBoard viewer
# MAGIC Run only the cell below in an active notebook to view a saved experiment.
# MAGIC It works on CPU compute and does not train. Change RUN_TO_VIEW to choose a run.
# MAGIC Jobs finish above; an interactive TensorBoard UI needs an active notebook session.
# MAGIC Scalars shows losses, learning rates, precision/recall, and mAP; Images shows data examples.

# COMMAND ----------

import sys
from pathlib import Path

dbutils.widgets.text("code_dir", str(Path.cwd()))
sys.path.insert(0, dbutils.widgets.get("code_dir"))
from taco_yolo import display_tensorboard

RUN_TO_VIEW = globals().get("RESULTS", {}).get(
    "run_dir", "/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_small_gpu")
display_tensorboard(RUN_TO_VIEW)
