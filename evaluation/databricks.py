# Databricks notebook source
# MAGIC %md
# MAGIC # SortWaste YOLO + ResNet
# MAGIC `operation=evaluate`: held-out comparison with fixed checkpoints and identical YOLO boxes.
# MAGIC `operation=post_training`: ten epochs adapting C4 ResNet on frozen-YOLO training crops, with validation checkpoint selection.

# COMMAND ----------

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

for key, default in {
    "code_dir": str(Path.cwd()),
    "operation": "evaluate",
    "output_root": "/Volumes/workspace/default/tacotrashdata/reports/sortwaste_yolo_resnet_test",
    "run_name": datetime.now(timezone.utc).strftime("sortwaste_yolo_resnet_test_%Y%m%dT%H%M%SZ"),
}.items():
    dbutils.widgets.text(key, default)

run_name = dbutils.widgets.get("run_name")
if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
    raise ValueError("run_name must be a single directory name")
output = Path(dbutils.widgets.get("output_root")) / run_name
sys.path.insert(0, dbutils.widgets.get("code_dir"))

try:
    with tempfile.TemporaryDirectory(prefix="sortwaste_eval_settings_") as temporary:
        os.environ["YOLO_CONFIG_DIR"] = str(Path(temporary) / "yolo")
        os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The evaluation job requires its serverless A10 GPU")
        operation = dbutils.widgets.get("operation")
        if operation == "taco_training_comparison":
            from post_training import main as train
            from sortwaste import main as evaluate, TACO_YOLO, TACO_DATA
            training_output = Path("/Volumes/workspace/default/models/taco_yolo_resnet_post_training")
            trained = train(["--dataset", "taco", "--data-root", TACO_DATA,
                             "--output", str(training_output), "--yolo", TACO_YOLO,
                             "--run-name", training_output.name, "--epochs", "10"])
            result = evaluate(["--dataset", "taco", "--data-root", TACO_DATA,
                "--output", str(output), "--yolo", TACO_YOLO, "--detector-name", "YOLO_full_TACO",
                "--post-resnet", trained["best_checkpoint"],
                "--training-metrics", str(training_output / "metrics.json"), "--device", "cuda:0"])
        elif operation == "taco_comparison":
            from post_training import main as train
            from sortwaste import main as evaluate, TACO_YOLO
            training_output = Path("/Volumes/workspace/default/models/sortwaste_taco_yolo_resnet_post_training")
            if (training_output / "status.json").is_file() and json.loads((training_output / "status.json").read_text()).get("state") == "success":
                trained = json.loads((training_output / "metrics.json").read_text())
                if trained["epochs_completed"] != 10:
                    raise ValueError("The existing training run is not complete")
            else:
                trained = train(["--output", str(training_output), "--yolo", TACO_YOLO,
                                 "--run-name", training_output.name, "--epochs", "10"])
            result = evaluate(["--output", str(output), "--yolo", TACO_YOLO,
                "--detector-name", "YOLO_full_TACO", "--post-resnet", trained["best_checkpoint"],
                "--training-metrics", str(training_output / "metrics.json"), "--device", "cuda:0"])
        elif operation == "post_training":
            from post_training import main
            result = main(["--output", str(output), "--epochs", "10"])
        elif operation == "evaluate":
            from sortwaste import main
            result = main(["--output", str(output), "--device", "cuda:0"])
        else:
            raise ValueError(f"Unknown operation: {operation}")
except Exception:
    output.mkdir(parents=True, exist_ok=True)
    (output / "status.json").write_text(json.dumps({"state": "failed", "traceback": traceback.format_exc()}, indent=2))
    raise

if dbutils.widgets.get("operation") == "post_training":
    exported = {key: result[key] for key in ["epochs_completed", "best_epoch", "best_checkpoint", "best_validation"]}
else:
    exported = {"counts": result["counts"], "comparison": {
        name: values["primary_classifier_eligible"] for name, values in result["systems"].items()}}
dbutils.notebook.exit(json.dumps({"output": str(output), **exported}))
