"""Run a sequential TrashCNN hyperparameter grid; use --dry-run to preview."""

import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from trashnet_data import DEFAULT_MANIFEST, DEFAULT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[0.0003, 0.001, 0.003])
    parser.add_argument('--weight-decays', type=float, nargs='+', default=[0.0, 0.0001])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[32])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest-path', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-root', type=Path,
                        default=Path('/Volumes/workspace/default/models/cnn_grids'))
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training or writing files.')
    args = parser.parse_args()
    if (args.epochs < 1 or args.num_workers < 0
            or any(not math.isfinite(x) or x <= 0 for x in args.learning_rates)
            or any(not math.isfinite(x) or x < 0 for x in args.weight_decays)
            or any(x < 1 for x in args.batch_sizes)):
        parser.error('epochs/batch sizes/learning rates must be positive; weight decay/workers nonnegative')
    combinations = list(product(*[list(dict.fromkeys(values)) for values in
                                 [args.learning_rates, args.weight_decays, args.batch_sizes, args.seeds]]))
    print(f'{len(combinations)} trials, {args.epochs} epochs each, run sequentially.', flush=True)
    grid_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = args.output_root / grid_id
    if not args.dry_run:
        # Fail before creating a grid directory if compute or the shared split is unavailable.
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA GPU available. Use --dry-run to preview the grid.')
        if not args.manifest_path.is_file():
            raise FileNotFoundError(f'Shared split missing: {args.manifest_path}')
        output.mkdir(parents=True, exist_ok=False)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (output / 'grid_config.json').write_text(json.dumps(config, indent=2))
    results = []
    for index, (lr, decay, batch_size, seed) in enumerate(combinations, 1):
        trial = output / f'trial_{index:03d}'
        command = [sys.executable, str(Path(__file__).with_name('train_cnn.py')),
                   '--epochs', str(args.epochs), '--lr', str(lr),
                   '--weight-decay', str(decay), '--batch-size', str(batch_size),
                   '--seed', str(seed), '--num-workers', str(args.num_workers),
                   '--data-root', str(args.data_root), '--manifest-path', str(args.manifest_path),
                   '--output-root', str(trial)]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        # A fresh process gives every trial a new model, optimizer, RNG, and CUDA context.
        completed = subprocess.run(command, check=False)
        row = dict(trial=index, lr=lr, weight_decay=decay, batch_size=batch_size, seed=seed,
                   status='failed', best_val_macro_f1=None, val_accuracy_at_best_epoch=None,
                   best_epoch=None, total_seconds=None, artifact_directory=str(trial))
        summaries = list(trial.glob('*/summary.json'))
        if completed.returncode == 0 and len(summaries) == 1:
            row.update(json.loads(summaries[0].read_text()))
            row.update(status='ok', artifact_directory=str(summaries[0].parent))
        results.append(row)
        ranked = sorted(results, key=lambda r: (r['status'] == 'ok',
                         r['best_val_macro_f1'] if r['best_val_macro_f1'] is not None else -1), reverse=True)
        with (output / 'results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(ranked)
    if args.dry_run:
        return
    successful = [row for row in ranked if row['status'] == 'ok']
    if not successful:
        raise RuntimeError(f'All grid trials failed. See {output / "results.csv"}')
    (output / 'best_trial.json').write_text(json.dumps(successful[0], indent=2))
    print(f'Results: {output / "results.csv"}', flush=True)
    print(f'Best trial: {successful[0]["artifact_directory"]}', flush=True)
    if len(successful) != len(combinations):
        print(f'WARNING: {len(combinations) - len(successful)} trials failed; comparison is incomplete.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
