"""Verify completed joint training, reconcile test scoring and update the report."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .sortwaste import sha256
from .sortwaste_legacy_metrics import finalize


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'reports/sortwaste_joint_pipeline_10epochs'
    training = json.loads((output/'training_metrics.json').read_text())
    assert training['epochs_completed']==10
    assert [r['epoch'] for r in training['history']]==list(range(1,11))
    assert all(r['train_images']==3705 for r in training['history'])
    assert all(training['weight_updates_verified'].values())
    assert training['source_checkpoints_preserved']
    assert not training['config']['test_used_for_training_or_selection']
    original = json.loads((output/'evaluation/metrics.json').read_text())
    for name,expected in original['checkpoint_sha256'].items():
        assert sha256(output/name)==expected,(name,'Downloaded checkpoint mismatch')
    result = finalize(output,root/'reports/sortwaste_standalone_20261123/evaluation')
    assert result['test_images']==776
    status = json.loads((output/'status.json').read_text())
    if not (output/'status_from_training_job.json').exists():
        (output/'status_from_training_job.json').write_text(json.dumps(status,indent=2)+'\n')
    status.update(state='success',baseline_reproduction_passed=True,protocol_reconciled_locally=True)
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    subprocess.run([sys.executable,'-B',str(root/'reports/build_sortwaste_pipeline_report.py')],check=True)
    print(json.dumps({'selected_epoch':result['selected_epoch'],'test_images':result['test_images'],
                     'systems':result['systems'],'paired_differences':result['paired_differences']},indent=2))


if __name__=='__main__':
    main()
