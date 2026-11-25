# Databricks notebook source
# MAGIC %md
# MAGIC # C4 ResNet adaptation on the original TACO split
# MAGIC Frozen TACO YOLO; maximum 40 adaptation epochs, validation macro-F1 patience 8.

# COMMAND ----------

import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

for key, default in {'code_dir': str(Path.cwd()), 'run_name': 'taco_c4_40',
                     'reference_manifest_sha256': '', 'reference_yolo_sha256': ''}.items():
    dbutils.widgets.text(key, default)
run_name = dbutils.widgets.get('run_name')
if Path(run_name).name != run_name or run_name in {'', '.', '..'}:
    raise ValueError('run_name must be a directory name')
output = Path('/Volumes/workspace/default/models') / run_name
evaluation_output = output / 'evaluation'
sys.path.insert(0, dbutils.widgets.get('code_dir'))

try:
    with tempfile.TemporaryDirectory(prefix='taco_c4_settings_') as temporary:
        os.environ['YOLO_CONFIG_DIR'] = str(Path(temporary) / 'yolo')
        os.environ['MPLCONFIGDIR'] = str(Path(temporary) / 'matplotlib')
        import torch
        from post_training import main as train
        from sortwaste import main as paired_evaluate, TACO_YOLO, TACO_DATA, DEFAULT_RESNET, write_json
        from taco_detection import evaluate as detection_evaluate

        if not dbutils.widgets.get('reference_manifest_sha256') or not dbutils.widgets.get('reference_yolo_sha256'):
            raise ValueError('Original split manifest and YOLO checkpoint hashes are required')
        trained = train(['--dataset', 'taco', '--data-root', TACO_DATA,
                         '--output', str(output), '--yolo', TACO_YOLO,
                         '--resnet', DEFAULT_RESNET, '--run-name', run_name,
                         '--epochs', '40', '--patience', '8',
                         '--reference-manifest-sha256', dbutils.widgets.get('reference_manifest_sha256'),
                         '--reference-yolo-sha256', dbutils.widgets.get('reference_yolo_sha256')])
        gc.collect(); torch.cuda.empty_cache()
        paired = paired_evaluate(['--dataset', 'taco', '--data-root', TACO_DATA,
                '--output', str(evaluation_output), '--yolo', TACO_YOLO, '--resnet', DEFAULT_RESNET,
                '--detector-name', 'TACO YOLO26 Medium', '--post-resnet', trained['best_checkpoint'],
                '--training-metrics', str(output / 'metrics.json'), '--device', 'cuda:0'])
        gc.collect(); torch.cuda.empty_cache()
        detection = detection_evaluate(TACO_DATA, TACO_YOLO, DEFAULT_RESNET, trained['best_checkpoint'], evaluation_output)
        write_json(output / 'run_status.json', dict(state='success', epochs_completed=trained['epochs_completed'],
                  best_epoch=trained['best_epoch'], training_metrics=str(output/'metrics.json'),
                  evaluation_output=str(evaluation_output)))
except Exception:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_status.json').write_text(json.dumps(dict(state='failed', traceback=traceback.format_exc()), indent=2))
    raise

dbutils.notebook.exit(json.dumps(dict(output=str(output), epochs_completed=trained['epochs_completed'],
                                     best_epoch=trained['best_epoch'], detection=detection['systems'])))
