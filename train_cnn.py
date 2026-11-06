"""Train randomly initialized TrashCNN on the shared TrashNet split.

    python train_cnn.py --epochs 40 --lr 0.001 --weight-decay 0.0001

Requires GPU compute. The test set is reserved for final evaluation.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
from time import perf_counter
import uuid

from trashnet_data import DEFAULT_MANIFEST, DEFAULT_ROOT, get_dataloaders


def run_epoch(model, loader, criterion, device, optimizer=None):
    import torch
    from sklearn.metrics import f1_score

    training = optimizer is not None
    # From scratch: learn BatchNorm running statistics during training.
    model.train(training)
    total_loss = 0.0
    targets, predictions = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        'loss': total_loss / len(targets),
        'accuracy': sum(a == b for a, b in zip(targets, predictions)) / len(targets),
        'macro_f1': float(f1_score(
            targets, predictions, labels=list(range(model.classifier.out_features)),
            average='macro', zero_division=0,
        )),
    }


def save_checkpoint(checkpoint, destination):
    """Stage locally before copying to a Volume (avoids seek-write limitations)."""
    import torch

    with tempfile.TemporaryDirectory(prefix='trashnet_checkpoint_') as directory:
        local_path = Path(directory) / 'checkpoint.pt'
        torch.save(checkpoint, local_path)
        shutil.copyfile(local_path, destination)


def save_curves(history, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row['epoch'] for row in history]
    for ax, metric in zip(axes, ['loss', 'accuracy', 'macro_f1']):
        for split in ['train', 'val']:
            ax.plot(epochs, [row[f'{split}_{metric}'] for row in history],
                    marker='o', label=split)
        ax.set(xlabel='Epoch', ylabel=metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest-path', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-root', type=Path, default=Path(
        '/Volumes/workspace/default/models/cnn_experiments'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error('epochs/batch-size/lr must be positive; num-workers must be nonnegative')
    if args.weight_decay < 0:
        parser.error('weight-decay must be nonnegative')
    import numpy as np
    import torch
    from torch import nn
    from cnn_from_scratch import TrashCNN

    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available. Connect to GPU compute before running.')
    if not args.manifest_path.is_file():
        raise FileNotFoundError(
            f'Shared split missing: {args.manifest_path}. Copy your existing CSV to this '
            'server or pass --manifest-path PATH; reuse the same split for both models.'
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

    loaders, classes = get_dataloaders(
        data_root=args.data_root, manifest_path=args.manifest_path,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    model = TrashCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        architecture='TrashCNN', pretrained_weights=None, optimizer='AdamW',
        training_mode='from_scratch', checkpoint_selection='val_macro_f1', classes=classes, image_size=224,
        normalization_mean=[0.485, 0.456, 0.406],
        normalization_std=[0.229, 0.224, 0.225],
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
        split_sha256=hashlib.sha256(args.manifest_path.read_bytes()).hexdigest(),
    )
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    shutil.copyfile(args.manifest_path, output / 'split.csv')
    print(f'Classes: {classes}\nOutputs: {output}', flush=True)
    history = []
    best_macro_f1 = -1.0
    best_accuracy = None
    best_epoch = None
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        train = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val = run_epoch(model, loaders['val'], criterion, device)
        row = {'epoch': epoch, 'seconds': perf_counter() - epoch_start}
        row.update({f'train_{key}': value for key, value in train.items()})
        row.update({f'val_{key}': value for key, value in val.items()})
        history.append(row)
        with (output / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if val['macro_f1'] > best_macro_f1:
            best_macro_f1 = val['macro_f1']
            best_accuracy, best_epoch = val['accuracy'], epoch
            save_checkpoint({
                'model_state_dict': {key: value.detach().cpu()
                                     for key, value in model.state_dict().items()},
                'epoch': epoch, 'val_metrics': val, 'config': config,
            }, output / 'best.pt')
        print(
            f'Epoch {epoch}/{args.epochs} | train loss {train["loss"]:.4f} '
            f'acc {train["accuracy"]:.3f} | val loss {val["loss"]:.4f} '
            f'acc {val["accuracy"]:.3f} macro F1 {val["macro_f1"]:.3f}',
            flush=True,
        )
        save_curves(history, output / 'learning_curves.png')
    summary = dict(best_epoch=best_epoch, best_val_macro_f1=best_macro_f1,
                   val_accuracy_at_best_epoch=best_accuracy,
                   total_seconds=perf_counter() - start)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Finished. Best validation macro F1: {best_macro_f1:.3f} '
          f'(epoch {best_epoch}, accuracy {best_accuracy:.3f})')
    print(f'Artifacts: {output}')


if __name__ == '__main__':
    main()
