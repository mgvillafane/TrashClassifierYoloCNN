"""Run and retrieve the ten-epoch jointly trained SortWaste pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/sortwaste_joint_pipeline_10epochs'
CLI = 'C:/Users/marco/.vscode/extensions/databricks.databricks-2.24.0-win32-x64/bin/databricks.exe'
PROFILE = 'Marcos Gomez Villafane'
WORKSPACE = '/Workspace/Users/mgomezvillafane@berkeley.edu'
YOLO = '/Volumes/workspace/default/models/yolo_experiments/yolo_sortwaste_trashnet_realwaste_20261123T073012Z/weights/best.pt'
RESNET = '/Volumes/workspace/default/models/classifier_experiments/C4/20261120T034908Z_9bfaa9fb/resnet-layer4/best.pt'


def command(*args):
    result = subprocess.run([CLI, *map(str, args), '-p', PROFILE, '-o', 'json'],
                            capture_output=True, text=True, encoding='utf-8', timeout=180)
    if result.returncode:
        raise RuntimeError(f'{args}: {result.stderr}')
    return result.stdout


def copy(remote, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command('fs', 'cp', 'dbfs:' + remote, destination, '--overwrite')


def inspect():
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in [WORKSPACE + '/TrashClassifierYoloCNN-evaluation',
                      WORKSPACE + '/TrashClassifierYoloCNN', WORKSPACE + '/CS289']:
        entries = json.loads(command('workspace', 'list', directory))
        print(json.dumps({'directory': directory, 'entries': entries}), flush=True)
    copy(YOLO, OUT / 'reference/yolo_all.pt')
    for name in ['run_info.json', 'args.yaml']:
        copy(YOLO.rsplit('/weights/', 1)[0] + '/' + name, OUT / 'reference' / name)
    copy('/Volumes/workspace/default/waste_data/processed/yolo_classes/sortwaste_test.txt', OUT / 'reference/sortwaste_test.txt')
    expected = ROOT / 'reports/sortwaste_yolo_resnet_test/archive/data_metadata/sortwaste_test.txt'
    assert (OUT / 'reference/sortwaste_test.txt').read_bytes() == expected.read_bytes()
    print('Baseline checkpoint and test membership downloaded and verified.', flush=True)


def launch():
    state = OUT / 'launch.json'
    if state.exists():
        raise RuntimeError('A run already exists; use status to avoid duplicate training.')
    OUT.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime('sortwaste_joint_c4_10epochs_%Y%m%dT%H%M%SZ')
    code_dir = WORKSPACE + '/TrashClassifierYoloCNN-evaluation/' + run_name
    output = '/Volumes/workspace/default/models/' + run_name
    job = json.loads((ROOT / 'reports/taco_yolo_resnet_test/archive/job.json').read_text())
    job.update(name=run_name, timeout_seconds=21600,
               description='Ten full epochs jointly updating all YOLO26m and C4 ResNet18 weights on SortWaste train. Validation selects the pair; original 776-photo test is evaluated once.',
               tags={'dataset': 'sortwaste', 'purpose': 'joint_yolo_resnet_10epochs',
                     'aidevkit_project':'TrashClassifierYoloCNN'})
    job['tasks'][0].update(task_key='joint_train_and_test')
    job['tasks'][0]['notebook_task'].update(notebook_path=code_dir + '/train_and_test',
        base_parameters={'code_dir': code_dir, 'output': output, 'source_yolo': YOLO, 'source_resnet': RESNET})
    (OUT / 'job.json').write_text(json.dumps(job, indent=2))
    command('workspace', 'mkdirs', code_dir)
    hashes = {}
    for name in ['sortwaste.py', 'sortwaste_joint.py', 'sortwaste_joint_metrics.py']:
        source = ROOT / 'evaluation' / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        command('workspace', 'import', code_dir + '/' + name, '--format', 'RAW', '--file', source)
    source = ROOT / 'evaluation/sortwaste_joint_databricks.py'
    hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    command('workspace', 'import', code_dir + '/train_and_test', '--format', 'SOURCE',
            '--language', 'PYTHON', '--file', source)
    created = json.loads(command('jobs', 'create', '--json', '@' + str(OUT / 'job.json')))
    record = {'run_name': run_name, 'job_id': created['job_id'], 'remote_output': output,
              'code_dir': code_dir, 'source_sha256': hashes}
    state.write_text(json.dumps(record, indent=2))
    request = {'job_id': created['job_id'], 'idempotency_token': run_name}
    (OUT / 'run_request.json').write_text(json.dumps(request))
    submitted = json.loads(command('jobs', 'run-now', '--json', '@' + str(OUT / 'run_request.json'), '--no-wait'))
    record['run_id'] = submitted['run_id']
    state.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def evaluator():
    jobs = json.loads(command('jobs', 'list', '--limit', '100'))
    for job in jobs:
        name = job.get('settings', {}).get('name', '')
        print(json.dumps({'job_id':job.get('job_id'),'name':name}),flush=True)
        if any(key in name.lower() for key in ['model_eval', 'yolo_sortwaste', 'classifier', 'waste_eval']):
            detail = json.loads(command('jobs', 'get', job['job_id']))
            print(json.dumps(detail), flush=True)
    runs = json.loads(command('jobs','list-runs','--limit','100'))
    for run in runs:
        if any(s in run.get('run_name','').lower() for s in ['waste','eval','yolo','classifier']):
            print(json.dumps({'run_id':run.get('run_id'),'name':run.get('run_name'),
                              'task':run.get('notebook_task'),'job_id':run.get('job_id')}),flush=True)


def labels():
    destination = OUT/'reference/test_labels'
    destination.mkdir(parents=True,exist_ok=True)
    command('fs','cp','-r','dbfs:/Volumes/workspace/default/waste_data/processed/yolo_classes/labels/sortwaste/test',destination,'--overwrite')
    print('Downloaded test annotation files for scorer verification:',len(list(destination.glob('*.txt'))),flush=True)


def taco_assets():
    target = ROOT/'reports/taco_three_model_transfer/reference'
    copy('/Volumes/workspace/default/models/taco_yolo_resnet_post_training/best.pt',target/'taco_post_resnet.pt')
    copy('/Volumes/workspace/default/models/taco_yolo_resnet_post_training/metrics.json',target/'taco_post_training_metrics.json')
    print(str(target),flush=True)


def status():
    record = json.loads((OUT / 'launch.json').read_text())
    run = json.loads(command('jobs', 'get-run', record['run_id']))
    (OUT / 'run.json').write_text(json.dumps(run, indent=2))
    print(json.dumps({k: run.get(k) for k in ['run_id', 'run_page_url', 'state']}, indent=2), flush=True)
    for name in ['status.json', 'training_metrics.json']:
        try:
            copy(record['remote_output'] + '/' + name, OUT / name)
            value = json.loads((OUT / name).read_text())
            if name=='training_metrics.json':
                value = {key:value.get(key) for key in ['epochs_completed','best_epoch','best_validation','weight_updates_verified']}
                full = json.loads((OUT / name).read_text())
                value['last_epoch'] = full.get('history',[None])[-1] if full.get('history') else None
            print(name + ': ' + json.dumps(value,indent=2), flush=True)
        except RuntimeError as error:
            if not any(s in str(error).lower() for s in ['does not exist', 'not found', 'no such file']):
                raise


def output():
    record = json.loads((OUT / 'launch.json').read_text())
    run = json.loads(command('jobs', 'get-run', record['run_id']))
    for task in run.get('tasks', []):
        result = command('jobs', 'get-run-output', task['run_id'])
        (OUT / 'notebook_output.json').write_text(result)
        print(result, flush=True)


def download():
    record = json.loads((OUT / 'launch.json').read_text())
    for name in ['status.json', 'training_metrics.json', 'history.csv', 'best_yolo.pt', 'best_resnet.pt',
                 'last_yolo.pt', 'last_resnet.pt', 'evaluation/metrics.json', 'evaluation/per_class.csv',
                 'evaluation/predictions.json.gz', 'evaluation/baseline_reproduction.json']:
        copy(record['remote_output'] + '/' + name, OUT / name)
    print(str(OUT), flush=True)


def publish():
    record = json.loads((OUT/'launch.json').read_text())
    result = json.loads((OUT/'evaluation/metrics.json').read_text())
    assert result['baseline_reproduction_passed']
    files = ['SortWaste_YOLO_ResNet_comparison.pdf','comparison_data.csv','report_provenance.json',
             'status.json','status_from_training_job.json','evaluation/metrics.json','evaluation/per_class.csv',
             'evaluation/baseline_reproduction_exact.json','evaluation/protocol_reconciliation.json',
             'evaluation/job_metrics_before_protocol_reconciliation.json']
    for name in files:
        command('fs','cp',OUT/name,'dbfs:'+record['remote_output']+'/'+name,'--overwrite')
    for name in ['sortwaste_legacy_metrics.py','finalize_sortwaste_joint.py']:
        command('fs','cp',ROOT/'evaluation'/name,'dbfs:'+record['remote_output']+'/evaluation/'+name,'--overwrite')
    request = {'job_id':record['job_id'],'new_settings':{'tags':{
        'dataset':'sortwaste','purpose':'joint_yolo_resnet_10epochs','aidevkit_project':'TrashClassifierYoloCNN'}}}
    (OUT/'job_metadata_update.json').write_text(json.dumps(request))
    command('jobs','update','--json','@'+str(OUT/'job_metadata_update.json'))
    print('Verified report and test results saved alongside the model checkpoints.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'evaluator', 'labels', 'taco_assets', 'launch', 'status', 'output', 'download', 'publish'])
    globals()[parser.parse_args().action]()
