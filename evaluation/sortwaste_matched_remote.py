"""Submit and retrieve the one-time fixed-crop SortWaste classification comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/sortwaste_three_model_classification'
CLI = 'C:/Users/marco/.vscode/extensions/databricks.databricks-2.24.0-win32-x64/bin/databricks.exe'
PROFILE = 'Marcos Gomez Villafane'
ARCHIVE = '/Volumes/workspace/default/tacotrashdata/reports/sortwaste_yolo_resnet_test/sortwaste_yolo_resnet_test_20261122T211247Z'


def command(*args):
    completed = subprocess.run([CLI, *map(str, args), '-p', PROFILE, '-o', 'json'],
                               capture_output=True, text=True, encoding='utf-8', timeout=180)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])
    return completed.stdout


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    state = OUT / 'launch.json'
    if state.exists():
        raise RuntimeError('An evaluation run already exists; use status to avoid duplicate inference.')
    run_name = datetime.now(timezone.utc).strftime('sortwaste_matched_classification_%Y%m%dT%H%M%SZ')
    code_dir = '/Workspace/Users/mgomezvillafane@berkeley.edu/TrashClassifierYoloCNN-evaluation/' + run_name
    output = '/Volumes/workspace/default/models/' + run_name
    local_archive = ROOT / 'reports/sortwaste_yolo_resnet_test/archive/sortwaste_yolo_resnet_test_20261122T211247Z'
    local_files = {'predictions.jsonl': local_archive / 'predictions.jsonl',
                   'objects.csv': ROOT / 'reports/sortwaste_yolo_resnet_test/objects.csv',
                   'metrics.json': ROOT / 'reports/sortwaste_yolo_resnet_test/metrics.json'}
    input_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in local_files.items()}
    # Reuse the already verified serverless A10 environment and policy for this project.
    template = json.loads((ROOT / 'reports/sortwaste_yolo_resnet_post_training/archive/job.json').read_text())
    task = {'task_key': 'classify_same_test_objects', 'compute': template['tasks'][0]['compute'],
            'environment_key': template['tasks'][0]['environment_key'],
            'notebook_task': {'notebook_path': code_dir + '/compare', 'source': 'WORKSPACE',
                'base_parameters': {'code_dir': code_dir, 'output': output, 'archive': ARCHIVE,
                                    'input_hashes': json.dumps(input_hashes)}}}
    request = {'run_name': run_name, 'timeout_seconds': 3600, 'performance_target': 'STANDARD',
               'usage_policy_id': template['usage_policy_id'], 'idempotency_token': run_name,
               'tasks': [task], 'environments': template['environments']}
    (OUT / 'submit.json').write_text(json.dumps(request, indent=2))
    command('workspace', 'mkdirs', code_dir)
    source_hashes = {}
    for name in ['sortwaste.py', 'sortwaste_matched.py']:
        source = ROOT / 'evaluation' / name
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        command('workspace', 'import', code_dir + '/' + name, '--format', 'RAW', '--file', source)
    notebook = ROOT / 'evaluation/sortwaste_matched_databricks.py'
    source_hashes[notebook.name] = hashlib.sha256(notebook.read_bytes()).hexdigest()
    command('workspace', 'import', code_dir + '/compare', '--format', 'SOURCE', '--language', 'PYTHON', '--file', notebook)
    record = {'run_name': run_name, 'remote_output': output, 'code_dir': code_dir,
              'source_sha256': source_hashes, 'input_sha256': input_hashes}
    state.write_text(json.dumps(record, indent=2))
    submitted = json.loads(command('jobs', 'submit', '--json', '@' + str(OUT / 'submit.json'), '--no-wait'))
    record['run_id'] = submitted['run_id']
    state.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def status():
    record = json.loads((OUT / 'launch.json').read_text())
    run = json.loads(command('jobs', 'get-run', record['run_id']))
    (OUT / 'run.json').write_text(json.dumps(run, indent=2))
    print(json.dumps({k: run.get(k) for k in ['run_id', 'run_page_url', 'state']}, indent=2), flush=True)
    try:
        target = OUT / 'status.json'
        command('fs', 'cp', 'dbfs:' + record['remote_output'] + '/status.json', target, '--overwrite')
        print(target.read_text(), flush=True)
    except RuntimeError as exc:
        if not any(s in str(exc).lower() for s in ['not found', 'does not exist', 'no such file']):
            raise


def download():
    record = json.loads((OUT / 'launch.json').read_text())
    for name in ['metrics.json', 'objects.csv', 'comparison.csv', 'per_class.csv', 'status.json']:
        command('fs', 'cp', 'dbfs:' + record['remote_output'] + '/' + name, OUT / name, '--overwrite')
    print((OUT / 'comparison.csv').read_text(), flush=True)


def logs():
    record = json.loads((OUT / 'launch.json').read_text())
    run = json.loads(command('jobs', 'get-run', record['run_id']))
    for task in run.get('tasks', []):
        result = json.loads(command('jobs', 'get-run-output', task['run_id']))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'status', 'download', 'logs'])
    args = parser.parse_args()
    globals()[args.action]()
