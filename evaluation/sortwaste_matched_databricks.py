# Databricks notebook source
import json
from pathlib import Path
import sys
import traceback

for key in ['code_dir', 'output', 'archive', 'input_hashes']:
    dbutils.widgets.text(key, '')
sys.path.insert(0, dbutils.widgets.get('code_dir'))
output = Path(dbutils.widgets.get('output'))
try:
    from sortwaste_matched import evaluate
    result = evaluate(
        archive=dbutils.widgets.get('archive'),
        data='/Volumes/workspace/default/waste_data/processed/yolo_classes',
        classifier='/Volumes/workspace/default/models/classifier_experiments/C4/20261120T034908Z_9bfaa9fb/resnet-layer4/best.pt',
        adapted='/Volumes/workspace/default/models/sortwaste_yolo_resnet_post_training/best.pt',
        output=output, input_hashes=json.loads(dbutils.widgets.get('input_hashes')))
except Exception:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'status.json').write_text(json.dumps({'state': 'failed', 'traceback': traceback.format_exc()}, indent=2))
    raise
dbutils.notebook.exit(json.dumps({'output': str(output), 'matched_objects': result['matched_objects'],
    'systems': {s: {k:v for k,v in m.items() if k != 'per_class'} for s,m in result['systems'].items()}}))
