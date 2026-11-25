# Databricks notebook source
# MAGIC %md
# MAGIC # Joint SortWaste YOLO26m + C4 ResNet18: ten epochs and held-out evaluation

# COMMAND ----------

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

for key in ['code_dir','output','source_yolo','source_resnet']:
    dbutils.widgets.text(key,'')
sys.path.insert(0,dbutils.widgets.get('code_dir'))
output = Path(dbutils.widgets.get('output'))
try:
    with tempfile.TemporaryDirectory(prefix='joint_settings_') as temporary:
        os.environ['YOLO_CONFIG_DIR'] = str(Path(temporary)/'yolo')
        os.environ['MPLCONFIGDIR'] = str(Path(temporary)/'matplotlib')
        from sortwaste_joint import main
        result = main(['--output',str(output),'--yolo',dbutils.widgets.get('source_yolo'),
                       '--resnet',dbutils.widgets.get('source_resnet'),'--epochs','10',
                       '--test-hash','db5f78052ab584af207be689e6cdf8059ba17bf3c7742da5a9398529d8179efc'])
except Exception:
    output.mkdir(parents=True,exist_ok=True)
    (output/'status.json').write_text(json.dumps({'state':'failed','traceback':traceback.format_exc()},indent=2))
    raise
dbutils.notebook.exit(json.dumps({'output':str(output),'selected_epoch':result['selected_epoch'],
                                'baseline_reproduction_passed':result['baseline_reproduction_passed'],
                                'systems':result['systems']}))
