"""Fine-tune selected blocks of pretrained ResNet-18 on the shared TrashNet split.

After connecting to GPU compute, run from the project directory:
    python train_resnet_finetune.py --trainable-blocks layer4 --epochs 20
    python train_resnet_finetune.py --trainable-blocks layer3 layer4 --epochs 20
    python train_resnet_finetune.py --trainable-blocks fc --warmup-epochs 0 --epochs 20

Each invocation creates a separate output directory. The test set is reserved
for a later final evaluation. This script never downloads pretrained weights.
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
    # Train the selected blocks, but keep every BatchNorm running statistic fixed.
    # eval() does not disable gradients: unfrozen convolution weights still learn.
    model.train(training)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
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
            targets, predictions, labels=list(range(model.fc.out_features)),
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
    import numpy as np
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--backbone-lr', type=float, default=0.0001)
    parser.add_argument('--trainable-blocks', nargs='+', choices=['fc', 'layer3', 'layer4'],
                        default=['layer4'],
                        help='Backbone blocks to fine-tune; fc is always trained. Use fc alone for baseline.')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Initial head-only epochs, included in --epochs; use 0 to skip.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest-path', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--weights', type=Path, default=Path(
        '/Volumes/workspace/default/models/resnet_pretrained/torch_cache/'
        'checkpoints/resnet18-f37072fd.pth'))
    parser.add_argument('--output-root', type=Path, default=Path(
        '/Volumes/workspace/default/models/resnet_experiments'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error('epochs/batch-size/lr must be positive; num-workers must be nonnegative')
    if args.backbone_lr <= 0:
        parser.error('backbone-lr must be positive')
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error('warmup-epochs must be nonnegative and less than epochs')
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available. Connect to GPU compute before running.')
    if not args.weights.is_file():
        raise FileNotFoundError(f'Cached weights missing: {args.weights}. Pass --weights PATH.')
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
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    # Include backbone parameters now so Adam keeps the head's optimizer state
    # when they are unfrozen after warm-up. Frozen parameters have no gradients.
    blocks = sorted(set(args.trainable_blocks) - {'fc'})
    backbone_parameters = [parameter for block in blocks
                           for parameter in getattr(model, block).parameters()]
    parameter_groups = [{'params': model.fc.parameters(), 'lr': args.lr}]
    if backbone_parameters:
        parameter_groups.append({'params': backbone_parameters, 'lr': args.backbone_lr})
    optimizer = torch.optim.Adam(parameter_groups)
    if args.warmup_epochs == 0:
        for parameter in backbone_parameters:
            parameter.requires_grad = True
    criterion = nn.CrossEntropyLoss()

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        architecture='resnet18', pretrained_weights=ResNet18_Weights.DEFAULT.name,
        training_mode='partial_finetuning' if blocks else 'frozen_backbone',
        trainable_modules=['fc'] + blocks, batchnorm_running_stats='frozen',
        checkpoint_selection='val_macro_f1', classes=classes, image_size=224,
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
        if epoch == args.warmup_epochs + 1:
            for parameter in backbone_parameters:
                parameter.requires_grad = True
        phase = 'warmup' if epoch <= args.warmup_epochs else 'finetune'
        epoch_start = perf_counter()
        train = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val = run_epoch(model, loaders['val'], criterion, device)
        row = {'epoch': epoch, 'phase': phase, 'seconds': perf_counter() - epoch_start}
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
            f'Epoch {epoch}/{args.epochs} ({phase}) | train loss {train["loss"]:.4f} '
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
