"""Launch, monitor and retrieve the reproducible 40-epoch TACO C4 experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/taco_c4_40'
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / 'launch.json'
CLI = 'C:/Users/marco/.vscode/extensions/databricks.databricks-2.24.0-win32-x64/bin/databricks.exe'
PROFILE = 'Marcos Gomez Villafane'
DATA = '/Volumes/workspace/default/tacotrashdata/yolo/taco_trashnet'
ORIGINAL_RUN = '/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_trashnet_yolo26m_40epochs_2184af'


def command(*args):
    result = subprocess.run([CLI, *args, '-p', PROFILE, '-o', 'json'], capture_output=True,
                            text=True, encoding='utf-8', timeout=90)
    if result.returncode:
        raise RuntimeError(f'{args}: {result.stderr}')
    return result.stdout


def download_file(remote, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command('fs', 'cp', 'dbfs:' + remote, str(destination), '--overwrite')


def inspect():
    for name in ['dataset.yaml', 'split_manifest.json', 'label_mapping.json']:
        download_file(DATA + '/' + name, OUT / 'reference' / name)
    download_file(ORIGINAL_RUN + '/run_config.json', OUT / 'reference/original_run_config.json')
    archive = ROOT / 'reports/taco_yolo_resnet_test/archive/data_metadata/split_manifest.json'
    actual = hashlib.sha256((OUT / 'reference/split_manifest.json').read_bytes()).hexdigest()
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Remote TACO split differs from the archived original manifest')
    config = json.loads((OUT / 'reference/original_run_config.json').read_text())
    if config['data'] != DATA + '/dataset.yaml':
        raise ValueError('Original standalone YOLO used a different dataset root')
    manifest = json.loads((OUT / 'reference/split_manifest.json').read_text())
    record = dict(original_run=ORIGINAL_RUN, split_manifest_sha256=actual,
                  identical_to_archived_original=True, image_counts=manifest['image_counts'],
                  source_batches={s: sorted({r['source_file'].split('/')[0] for r in manifest['images'] if r['split'] == s})
                                  for s in ['train', 'val', 'test']})
    (OUT / 'reference/verification.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def launch():
    if STATE.exists():
        raise RuntimeError('An experiment has already been launched; use status to avoid duplicate GPU runs')
    inspect()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_name = 'taco_yolo_c4_40epochs_' + timestamp
    code_dir = '/Workspace/Users/mgomezvillafane@berkeley.edu/TrashClassifierYoloCNN-evaluation/' + run_name
    notebook = code_dir + '/train_taco_c4'
    reference = json.loads((OUT / 'reference/verification.json').read_text())
    yolo = json.loads((ROOT / 'app/models/YOLO_full_TACO.source.json').read_text())
    job = json.loads((ROOT / 'reports/taco_yolo_resnet_test/archive/job.json').read_text())
    job.update(name=run_name,
               description='C4 ResNet layer4/head adaptation on frozen TACO YOLO crops: 40 epochs max, validation macro-F1 patience 8; original standalone split hash verified before training.',
               tags={'dataset': 'taco', 'purpose': 'c4_40epochs_patience8', 'aidevkit_project': 'TrashClassifierYoloCNN'})
    task = job['tasks'][0]
    task['task_key'] = 'taco_c4_train_and_evaluate'
    task['notebook_task'].update(notebook_path=notebook, base_parameters={
        'code_dir': code_dir, 'run_name': run_name,
        'reference_manifest_sha256': reference['split_manifest_sha256'],
        'reference_yolo_sha256': yolo['sha256']})
    (OUT / 'job.json').write_text(json.dumps(job, indent=2))
    command('workspace', 'mkdirs', code_dir)
    code_hashes = {}
    for name in ['sortwaste.py', 'post_training.py', 'taco_detection.py']:
        source = ROOT / 'evaluation' / name
        code_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        command('workspace', 'import', code_dir + '/' + name, '--format', 'RAW', '--file', str(source))
    source = ROOT / 'evaluation/taco_c4_databricks.py'
    code_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    command('workspace', 'import', notebook, '--format', 'SOURCE', '--language', 'PYTHON', '--file', str(source))
    created = json.loads(command('jobs', 'create', '--json', '@' + str(OUT / 'job.json')))
    record = dict(run_name=run_name, job_id=created['job_id'], code_dir=code_dir,
                  remote_output='/Volumes/workspace/default/models/' + run_name, code_sha256=code_hashes,
                  split_manifest_sha256=reference['split_manifest_sha256'], yolo_sha256=yolo['sha256'])
    STATE.write_text(json.dumps(record, indent=2))
    request = dict(job_id=created['job_id'], idempotency_token=run_name)
    (OUT / 'run_request.json').write_text(json.dumps(request))
    submitted = json.loads(command('jobs', 'run-now', '--json', '@' + str(OUT/'run_request.json'), '--no-wait'))
    record['run_id'] = submitted['run_id']
    STATE.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def status():
    record = json.loads(STATE.read_text())
    run = json.loads(command('jobs', 'get-run', str(record['run_id'])))
    (OUT / 'run.json').write_text(json.dumps(run, indent=2))
    print(json.dumps({k: run.get(k) for k in ['run_id', 'run_page_url', 'state']}, indent=2), flush=True)
    for name in ['run_status.json', 'status.json', 'evaluation/status.json', 'evaluation/detection_status.json']:
        try:
            destination = OUT / name
            download_file(record['remote_output'] + '/' + name, destination)
            print(name + ': ' + destination.read_text(), flush=True)
        except RuntimeError as error:
            if not any(marker in str(error).lower() for marker in ['resource_does_not_exist', 'does not exist', 'no such file']):
                raise


def output():
    record = json.loads(STATE.read_text())
    run = json.loads(command('jobs', 'get-run', str(record['run_id'])))
    for task in run.get('tasks', []):
        text = command('jobs', 'get-run-output', str(task['run_id']))
        (OUT / 'notebook_output.json').write_text(text)
        print(text, flush=True)


def download():
    record = json.loads(STATE.read_text())
    for name in ['status.json', 'metrics.json', 'history.csv', 'best.pt', 'last.pt', 'run_status.json',
                 'evaluation/status.json', 'evaluation/metrics.json', 'evaluation/objects.csv',
                 'evaluation/detection_metrics.json', 'evaluation/detection_predictions.json']:
        download_file(record['remote_output'] + '/' + name, OUT / name)
    print('Downloaded training checkpoints, history and evaluation: ' + str(OUT), flush=True)


def watch():
    while True:
        status()
        run = json.loads((OUT / 'run.json').read_text())
        state = run.get('state', {})
        if state.get('life_cycle_state') in {'TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'}:
            if state.get('result_state') == 'SUCCESS':
                download()
            else:
                output()
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'launch', 'status', 'output', 'download', 'watch'])
    action = parser.parse_args().action
    globals()[action]()
