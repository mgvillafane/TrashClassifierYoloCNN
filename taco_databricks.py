# Databricks notebook source
# MAGIC %md
# MAGIC # TACO YOLO26 Medium GPU experiment
# MAGIC Import this file as a Databricks notebook. Upload `taco_yolo.py` as a regular
# MAGIC workspace file beside it, and put the complete raw TACO dataset in a writable
# MAGIC Unity Catalog volume. The `taco_small_gpu` bundle job requests one A10 GPU.
# MAGIC **VS Code:** open the Databricks sidebar, expand Bundle Resource Explorer,
# MAGIC select `taco_small_gpu`, and click its play button (Deploy and Run).
# MAGIC Or open the Command Palette and choose **Databricks: Deploy the bundle and
# MAGIC run a resource**, then select `taco_small_gpu`.
# MAGIC The editor's Run File as Workflow command uses the selected compute; it does
# MAGIC not apply the GPU settings from the bundle job.
# MAGIC This run uses YOLO26 Medium, 2 epochs, 640-pixel inputs, and all training images.
# MAGIC Run the cells in order. Edit the configuration cell to change the experiment.
# MAGIC TensorBoard publishes each completed epoch while training continues.
# MAGIC Interactive runs show the viewer before training. For a bundle job, open
# MAGIC this notebook in a separate CPU session and run only the final viewer cell.

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
from ultralytics.utils.files import increment_path

settings.update({"mlflow": False, "tensorboard": True})

LOCAL_RUN_ROOT = tempfile.TemporaryDirectory(prefix="taco_yolo_")
SAVED_RUN_DIR = increment_path(TACO_ROOT / "reports/taco_yolo/databricks_yolo26m_2epochs", mkdir=True)
CONFIG = ExperimentConfig(
    source=TACO_ROOT / "taco",
    prepared=TACO_ROOT / "yolo/taco",
    project=Path(LOCAL_RUN_ROOT.name),
    name="databricks_yolo26m_2epochs",
    weights=TACO_ROOT / "yolo/weights/yolo26m.pt",  # Reuse the uploaded pretrained weights.
    epochs=2,
    imgsz=640,
    batch=8,
    device="0",  # Require the first CUDA GPU.
    patience=0,  # Run both epochs.
    lr=0.001,
    fraction=1.0,  # Use every image in the training split; keep validation/test held out.
    freeze=0,
    workers=0,
    threads=4,
    seed=42,
    max_side=1280,
    val_ratio=0.15,
    test_ratio=0.15,
    prediction_count=2,
    conf=0.25,
    tensorboard_dir=SAVED_RUN_DIR,
)
print("Live TensorBoard and saved run:", SAVED_RUN_DIR)

# COMMAND ----------

# Open the dashboard before training in an interactive GPU notebook.
# A bundle job also publishes logs here; watch it from a separate CPU notebook.
from taco_yolo import display_tensorboard

display_tensorboard(SAVED_RUN_DIR, live=True)

# COMMAND ----------

# Prepares copies, trains, evaluates validation data, and saves comparisons.
# Raw images/annotations remain unchanged; test data is reserved for final scoring.
RESULTS = run_experiment(CONFIG)

# Copy completed outputs into one persistent experiment folder.
run_dir = Path(RESULTS["run_dir"])
saved_dir = SAVED_RUN_DIR
for path in run_dir.rglob("*"):
    if path.is_file():
        target = saved_dir / path.relative_to(run_dir)
        # Epoch files were already published; keep them immutable for live readers.
        if path.name.startswith("events.out.tfevents.epoch") and target.is_file():
            continue
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
# MAGIC ## Held-out TACO test
# MAGIC Run only the next cell after choosing a trained checkpoint for final scoring.
# MAGIC It loads the saved best.pt and scores the 177 test images without retraining.
# MAGIC Use GPU compute for faster testing. Keep validation for tuning experiments.

# COMMAND ----------

import json
import sys
from pathlib import Path

dbutils.widgets.text("code_dir", str(Path.cwd()))
sys.path.insert(0, dbutils.widgets.get("code_dir"))
from taco_yolo import test_saved_run

RUN_TO_TEST = Path("/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_yolo26m_2epochs")
TEST_DIR = test_saved_run(RUN_TO_TEST)
TEST_METRICS = json.loads((TEST_DIR / "metrics.json").read_text())
print("Held-out test metrics (0 to 1):", {
    key: TEST_METRICS[key] for key in ("precision", "recall", "mAP50", "mAP50_95", "evaluated_classes")
})
print("Test reports and plots:", TEST_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TensorBoard viewer
# MAGIC Run only the cell below in a separate active notebook to watch a GPU job live.
# MAGIC It works on CPU compute and does not train. The default lists all saved/live runs.
# MAGIC Fresh CPU session: install `tensorboard==2.21.0 Pillow` with `%pip install`,
# MAGIC then run `dbutils.library.restartPython()` before the viewer cell.
# MAGIC Select the current run in TensorBoard, or set RUN_TO_VIEW to its printed directory.
# MAGIC The dashboard refreshes while epochs finish. Keep this notebook session active.
# MAGIC Scalars shows losses, learning rates, precision/recall, and mAP; Images shows data examples.

# COMMAND ----------

import sys
from pathlib import Path

dbutils.widgets.text("code_dir", str(Path.cwd()))
sys.path.insert(0, dbutils.widgets.get("code_dir"))
from taco_yolo import display_tensorboard

RUN_TO_VIEW = str(globals().get(
    "SAVED_RUN_DIR", "/Volumes/workspace/default/tacotrashdata/reports/taco_yolo"))
display_tensorboard(RUN_TO_VIEW, live=True)
