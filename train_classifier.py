"""Run classifier experiments C1–C5 on the shared Databricks classifier export.

C5 trains on SortWaste only (class-balanced sampling like C4); it is the classifier of the
SortWaste-only pipeline in the YOLO experiments.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import uuid

from classifier import CLASSES, MODELS, PREPROCESSING, TRAINING_AUGMENTATION, build_model, image_transform
from train_cnn import save_checkpoint, save_curves


DATASETS = ('trashnet', 'realwaste', 'sortwaste')
EXPERIMENTS = {
    'C1': ('trashnet',),
    'C2': ('realwaste',),
    'C3': ('trashnet', 'realwaste'),
    'C4': DATASETS,
    'C5': ('sortwaste',),
}
BALANCED = {'C4', 'C5'}  # class-balanced sampling and SortWaste selection subset
DEFAULT_EPOCHS = {'cnn': 55, 'resnet-head': 20, 'resnet-layer4': 20}
DEFAULT_DATA = Path('/Volumes/workspace/default/waste_data/processed/classifier')
DEFAULT_WEIGHTS = Path('/Volumes/workspace/default/models/resnet_pretrained/torch_cache/'
                       'checkpoints/resnet18-f37072fd.pth')
DEFAULT_OUTPUT = Path('/Volumes/workspace/default/models/classifier_experiments')
DEFAULT_SORTWASTE_VAL_LIMIT = 500
DEFAULT_C4_SAMPLES_PER_EPOCH = 6000
DEFAULT_C4_SORTWASTE_VAL_PER_CLASS = 250


def read_index(index_path, training_datasets):
    train = {dataset: [] for dataset in training_datasets}
    validation = {dataset: [] for dataset in DATASETS}
    class_ids = {label: index for index, label in enumerate(CLASSES)}
    paths = set()
    with index_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'path', 'dataset', 'partition', 'label', 'label_id', 'use_classifier_train'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Unexpected classifier index columns: {reader.fieldnames}')
        for row in reader:
            dataset, split = row['dataset'], row['partition']
            if split == 'val' and dataset in validation:
                group = validation[dataset]
            elif split == 'train' and dataset in train and row['use_classifier_train'].lower() == 'true':
                group = train[dataset]
            else:
                continue
            path = Path(row['path'])
            if path.is_absolute() or '..' in path.parts or path.parts[:2] != (dataset, split):
                raise ValueError(f'Unexpected classifier path for {dataset}/{split}: {path}')
            if path in paths:
                raise ValueError(f'Duplicate classifier path: {path}')
            paths.add(path)
            label_id = int(row['label_id'])
            if class_ids.get(row['label']) != label_id:
                raise ValueError(f'Unexpected classifier label for {path}: {row["label"]} ({label_id})')
            group.append((path, label_id))
    for split, groups in [('train', train), ('val', validation)]:
        for dataset, rows in groups.items():
            if not rows:
                raise ValueError(f'No eligible {dataset}/{split} rows in {index_path}')
    return train, validation


def stratified_subset(rows, limit, seed):
    """Return a fixed proportional subset while retaining every represented class."""
    if limit == 0 or limit >= len(rows):
        return rows
    groups = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)
    if limit < len(groups):
        raise ValueError(f'Subset limit {limit} is smaller than the {len(groups)} represented classes.')

    total = len(rows)
    exact = {label: len(group) * limit / total for label, group in groups.items()}
    counts = {label: max(1, int(count)) for label, count in exact.items()}
    while sum(counts.values()) < limit:
        candidates = [label for label in groups if counts[label] < len(groups[label])]
        label = max(candidates, key=lambda item: (exact[item] - counts[item], -item))
        counts[label] += 1
    while sum(counts.values()) > limit:
        candidates = [label for label in groups if counts[label] > 1]
        label = min(candidates, key=lambda item: (exact[item] - counts[item], -item))
        counts[label] -= 1

    selected = []
    for label, group in sorted(groups.items()):
        selected.extend(random.Random(f'{seed}:{label}').sample(group, counts[label]))
    return sorted(selected, key=lambda row: str(row[0]))


def balanced_subset(rows, per_class_limit, seed):
    """Return up to the same fixed number of rows from every represented class."""
    groups = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)
    selected = []
    for label, group in sorted(groups.items()):
        count = min(per_class_limit, len(group))
        selected.extend(random.Random(f'{seed}:{label}').sample(group, count))
    return sorted(selected, key=lambda row: str(row[0]))


def c4_balanced_sampling(train_rows, require_all_classes=True):
    """Give every class equal mass and split that mass equally across its datasets."""
    cell_counts = {}
    datasets_per_class = {label_id: 0 for label_id in range(len(CLASSES))}
    for dataset, rows in train_rows.items():
        for _, label_id in rows:
            key = (dataset, label_id)
            cell_counts[key] = cell_counts.get(key, 0) + 1
    for _, label_id in cell_counts:
        datasets_per_class[label_id] += 1
    missing = [CLASSES[label_id] for label_id, count in datasets_per_class.items() if count == 0]
    if missing and require_all_classes:
        raise ValueError(f'C4 training data is missing classes: {missing}')

    flattened, weights = [], []
    for dataset, rows in train_rows.items():
        for row in rows:
            label_id = row[1]
            flattened.append(row)
            weights.append(1 / (datasets_per_class[label_id] * cell_counts[dataset, label_id]))
    return flattened, weights, cell_counts


class IndexedImages:
    def __init__(self, root, rows, transform):
        self.root = root
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        relative, label_id = self.rows[index]
        with Image.open(self.root / relative) as image:
            return self.transform(image.convert(PREPROCESSING['color_mode'])), label_id


def make_loader(args, rows, training=False, sample_weights=None):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    augmentation = TRAINING_AUGMENTATION if training else None
    images = IndexedImages(args.data_root, rows, image_transform(PREPROCESSING, augmentation))
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=args.c4_samples_per_epoch,
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    return DataLoader(images, batch_size=args.batch_size, shuffle=training and sampler is None,
                      sampler=sampler,
                      num_workers=args.num_workers, pin_memory=args.device.type == 'cuda',
                      generator=torch.Generator().manual_seed(
                          args.seed if sampler is None else args.seed + 1,
                      ))


def run_epoch(model, loader, criterion, device, optimizer=None):
    import torch
    from torch import nn

    training = optimizer is not None
    model.train(training)
    # Keep ImageNet BatchNorm statistics fixed while the ResNet head or layer4 adapts.
    if training and hasattr(model, 'layer4'):
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    loss_sum = 0.0
    targets, predictions = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            targets.extend(labels.tolist())
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach().item() * len(labels)
            predictions.extend(logits.detach().argmax(dim=1).tolist())
    return loss_sum, targets, predictions


def score(result):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    loss_sum, targets, predictions = result
    labels = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0,
    )
    # SortWaste has three ground-truth classes, while predictions still span all six.
    present = [i for i in labels if support[i] > 0]
    metrics = dict(loss=loss_sum / len(targets), count=len(targets),
                   accuracy=sum(a == b for a, b in zip(targets, predictions)) / len(targets),
                   macro_f1=float(f1[present].mean()), macro_f1_classes=[CLASSES[i] for i in present])
    per_class = [dict(label_id=i, label=label, accuracy=float(recall[i]),
                      precision=float(precision[i]), recall=float(recall[i]),
                      f1=float(f1[i]), support=int(support[i])) for i, label in enumerate(CLASSES)]
    return metrics, per_class, confusion_matrix(targets, predictions, labels=labels)


def dataset_average_metrics(metrics_by_dataset):
    datasets = list(metrics_by_dataset)
    return dict(
        loss=sum(metrics_by_dataset[name]['loss'] for name in datasets) / len(datasets),
        accuracy=sum(metrics_by_dataset[name]['accuracy'] for name in datasets) / len(datasets),
        macro_f1=sum(metrics_by_dataset[name]['macro_f1'] for name in datasets) / len(datasets),
        datasets=datasets,
    )


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_validation(output, evaluations, average_datasets=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output.mkdir()
    summaries = {}
    for name, evaluation in evaluations.items():
        print(f'Writing {name} validation metrics and confusion matrix...', flush=True)
        metrics, per_class, matrix = score(evaluation)
        summaries[name] = metrics
        write_csv(output / f'{name}_per_class.csv', per_class)
        with (output / f'{name}_confusion.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['actual / predicted', *CLASSES])
            writer.writerows([label, *row] for label, row in zip(CLASSES, matrix))
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, values_format='d')
        ax.set_title(f'{name} validation')
        fig.tight_layout()
        fig.savefig(output / f'{name}_confusion.png', dpi=150)
        plt.close(fig)
    if average_datasets:
        summaries['dataset_average'] = dataset_average_metrics(
            {name: summaries[name] for name in average_datasets},
        )
    (output / 'metrics.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(f'Validation reports saved to {output}', flush=True)
    return summaries


def train_model(args, architecture, epochs, train_rows, training_rows, sample_weights,
                selection_validation_rows, final_validation_rows, output, data_config):
    import numpy as np
    import torch
    from torch import nn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cnn = architecture == 'cnn'
    finetune = architecture == 'resnet-layer4'
    weights = None if cnn else args.weights
    warmup = args.warmup_epochs if finetune else 0
    learning_rate = args.cnn_lr if cnn else args.head_lr
    model = build_model(architecture, weights).to(args.device)
    groups = [{'params': model.parameters() if cnn else model.fc.parameters()}]
    if finetune:
        groups.append({'params': model.layer4.parameters(), 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(args, training_rows, training=True, sample_weights=sample_weights)
    c4 = args.experiment in BALANCED
    if c4:
        selection = 'dataset_average'
        selection_loaders = {
            name: make_loader(args, selection_validation_rows[name]) for name in train_rows
        }
    else:
        selection = next(iter(train_rows)) if len(train_rows) == 1 else 'combined'
        selection_rows = [row for dataset in train_rows for row in selection_validation_rows[dataset]]
        selection_loader = make_loader(args, selection_rows)
    checkpoint_selection = ('dataset_average_macro_f1_then_dataset_average_accuracy' if c4
                            else 'macro_f1_then_accuracy')
    config = dict(data_config, experiment=args.experiment, architecture=architecture, classes=list(CLASSES),
                  preprocessing=PREPROCESSING, training_augmentation=TRAINING_AUGMENTATION,
                  selection_validation=selection, checkpoint_selection=checkpoint_selection,
                  epochs=epochs, batch_size=args.batch_size, seed=args.seed,
                  device=str(args.device), num_workers=args.num_workers, torch_version=str(torch.__version__),
                  optimizer='AdamW', lr=learning_rate, weight_decay=args.weight_decay,
                  backbone_lr=args.backbone_lr if finetune else None, warmup_epochs=warmup,
                  pretrained_weights=str(weights) if weights else None,
                  batchnorm_running_stats='learned' if cnn else 'frozen')
    output.mkdir()
    print(f'{args.experiment} {architecture} | learning rate {learning_rate:g} → {output}', flush=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    history = []
    best = (-1.0, -1.0)
    for epoch in range(1, epochs + 1):
        if finetune and epoch == warmup + 1:
            model.layer4.requires_grad_(True)
        train_metrics = score(run_epoch(model, train_loader, criterion, args.device, optimizer))[0]
        val_metrics_by_dataset = None
        if c4:
            val_metrics_by_dataset = {
                name: score(run_epoch(model, loader, criterion, args.device))[0]
                for name, loader in selection_loaders.items()
            }
            val_metrics = dataset_average_metrics(val_metrics_by_dataset)
        else:
            val_metrics = score(run_epoch(model, selection_loader, criterion, args.device))[0]
        history_row = dict(epoch=epoch, **{
            f'{split}_{key}': metrics[key]
            for split, metrics in [('train', train_metrics), ('val', val_metrics)]
            for key in ('loss', 'accuracy', 'macro_f1')
        })
        if c4:
            history_row.update({
                f'val_{dataset}_{key}': metrics[key]
                for dataset, metrics in val_metrics_by_dataset.items()
                for key in ('loss', 'accuracy', 'macro_f1')
            })
        history.append(history_row)
        rank = (val_metrics['macro_f1'], val_metrics['accuracy'])
        if rank > best:
            best = rank
            checkpoint = dict(model_state_dict={name: tensor.detach().cpu()
                                                for name, tensor in model.state_dict().items()},
                              epoch=epoch, val_metrics=val_metrics, config=config)
            if c4:
                checkpoint['val_metrics_by_dataset'] = val_metrics_by_dataset
            save_checkpoint(checkpoint, output / 'best.pt')
        write_csv(output / 'history.csv', history)
        print(f'Epoch {epoch}/{epochs} | train loss {train_metrics["loss"]:.4f} | '
              f'{selection} val macro F1 {rank[0]:.4f}, accuracy {rank[1]:.4f}', flush=True)
    print(f'{architecture}: training complete. Saving learning curves...', flush=True)
    save_curves(history, output / 'learning_curves.png')

    print(f'{architecture}: loading best checkpoint for final validation...', flush=True)
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    evaluations = {}
    for name, rows in final_validation_rows.items():
        print(f'{architecture}: validating on {name} ({len(rows)} images)...', flush=True)
        evaluations[name] = run_epoch(model, make_loader(args, rows), criterion, args.device)
        metrics = score(evaluations[name])[0]
        print(f'{architecture}: {name} validation complete | macro F1 '
              f'{metrics["macro_f1"]:.4f}, accuracy {metrics["accuracy"]:.4f}', flush=True)
    if selection == 'combined':
        print(f'{architecture}: combining training-dataset validation results...', flush=True)
        parts = [evaluations[name] for name in train_rows]
        evaluations['combined'] = (sum(loss for loss, _, _ in parts),
                                   [label for _, targets, _ in parts for label in targets],
                                   [label for _, _, predictions in parts for label in predictions])
    print(f'{architecture}: generating validation reports...', flush=True)
    summary = save_validation(
        output / 'validation', evaluations,
        average_datasets=tuple(train_rows) if c4 else (),
    )
    if c4:
        selection_metrics = dict(
            epoch=checkpoint['epoch'],
            dataset_average=checkpoint['val_metrics'],
            by_dataset=checkpoint['val_metrics_by_dataset'],
        )
        (output / 'validation' / 'selection_metrics.json').write_text(
            json.dumps(selection_metrics, indent=2) + '\n',
        )
    result = dict(architecture=architecture, best_epoch=checkpoint['epoch'], selection_validation=selection,
                  **{f'{name}_{metric}': metrics[metric] for name, metrics in summary.items()
                     for metric in ('macro_f1', 'accuracy')}, checkpoint=str(output / 'best.pt'))
    if c4:
        for metric in ('macro_f1', 'accuracy'):
            result[f'final_dataset_average_{metric}'] = result.pop(f'dataset_average_{metric}')
            result[f'dataset_average_{metric}'] = checkpoint['val_metrics'][metric]
    print(f'{architecture}: finished. Best checkpoint came from epoch {checkpoint["epoch"]}.', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=EXPERIMENTS)
    parser.add_argument('--model', choices=(*MODELS, 'all'),
                        help='Defaults to all for C1, C3, and C4. For C2, pass the architecture selected in C1.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--index', type=Path, help='Defaults to index.csv under --data-root.')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--epochs', type=int,
                        help='Override the epoch budget for every selected model (CNN: 55; ResNet: 20).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cnn-lr', type=float, default=0.0003,
                        help='Learning rate for every custom CNN parameter.')
    parser.add_argument('--head-lr', type=float, default=0.001,
                        help='Learning rate for the ResNet classification head.')
    parser.add_argument('--backbone-lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sortwaste-val-limit', type=int, default=DEFAULT_SORTWASTE_VAL_LIMIT,
                        help='C1–C3 proportional SortWaste validation subset; use 0 for all images.')
    parser.add_argument('--c4-samples-per-epoch', type=int, default=DEFAULT_C4_SAMPLES_PER_EPOCH,
                        help='Weighted draws per C4 epoch (default: 6000; ignored by C1–C3).')
    parser.add_argument('--c4-sortwaste-val-per-class', type=int,
                        default=DEFAULT_C4_SORTWASTE_VAL_PER_CLASS,
                        help='SortWaste selection-validation examples per available class in C4 (default: 250).')
    parser.add_argument('--dry-run', action='store_true', help='Validate the index and print counts without training.')
    args = parser.parse_args(argv)
    if args.experiment == 'C2' and args.model in (None, 'all'):
        parser.error('C2 requires --model with one architecture selected from C1.')
    models = (args.model,) if args.model in MODELS else MODELS
    epochs_by_model = {model: DEFAULT_EPOCHS[model] if args.epochs is None else args.epochs
                       for model in models}
    if min(*epochs_by_model.values(), args.batch_size, args.cnn_lr,
           args.head_lr, args.backbone_lr) <= 0:
        parser.error('epochs, batch size and learning rates must be positive.')
    if min(args.num_workers, args.weight_decay, args.warmup_epochs) < 0:
        parser.error('workers, weight decay and warm-up must be nonnegative.')
    if args.sortwaste_val_limit < 0:
        parser.error('sortwaste-val-limit must be nonnegative.')
    if args.c4_samples_per_epoch <= 0 or args.c4_sortwaste_val_per_class <= 0:
        parser.error('C4 samples per epoch and SortWaste validation rows per class must be positive.')
    if 'resnet-layer4' in models and args.warmup_epochs >= epochs_by_model['resnet-layer4']:
        parser.error('ResNet layer4 warm-up must be shorter than the epoch budget.')

    index_path = args.index or args.data_root / 'index.csv'
    train_rows, validation_rows = read_index(index_path, EXPERIMENTS[args.experiment])
    training_rows = [row for rows in train_rows.values() for row in rows]
    sample_weights = None
    c4_sampling = None
    if args.experiment in BALANCED:
        training_rows, sample_weights, cell_counts = c4_balanced_sampling(
            train_rows, require_all_classes=args.experiment == 'C4')
        c4_sampling = {
            'strategy': 'weighted_random_sampler',
            'replacement': True,
            'samples_per_epoch': args.c4_samples_per_epoch,
            'weight_formula': '1 / (datasets_containing_class * examples_in_dataset_and_class)',
            'dataset_class_counts': {
                dataset: {CLASSES[label_id]: cell_counts.get((dataset, label_id), 0)
                          for label_id in range(len(CLASSES))}
                for dataset in train_rows
            },
        }
    sortwaste_val_total = len(validation_rows['sortwaste'])
    if args.experiment in BALANCED:
        selection_validation_rows = dict(validation_rows)
        selection_validation_rows['sortwaste'] = balanced_subset(
            validation_rows['sortwaste'], args.c4_sortwaste_val_per_class, args.seed,
        )
        final_validation_rows = validation_rows
    else:
        validation_rows['sortwaste'] = stratified_subset(
            validation_rows['sortwaste'], args.sortwaste_val_limit, args.seed,
        )
        selection_validation_rows = validation_rows
        final_validation_rows = validation_rows
    data_config = dict(data_root=str(args.data_root), index=str(index_path),
                       index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                       training_datasets=list(train_rows),
                       train_counts={name: len(rows) for name, rows in train_rows.items()},
                       validation_counts={name: len(rows) for name, rows in final_validation_rows.items()},
                       selection_validation_counts={
                           name: len(rows) for name, rows in selection_validation_rows.items()
                       },
                       selection_validation_class_counts={
                           name: {
                               CLASSES[label_id]: sum(row[1] == label_id for row in rows)
                               for label_id in range(len(CLASSES))
                           }
                           for name, rows in selection_validation_rows.items()
                       },
                       sortwaste_validation_total=sortwaste_val_total,
                       sortwaste_validation_subset_seed=args.seed,
                       c4_sortwaste_validation_per_class=(
                           args.c4_sortwaste_val_per_class if args.experiment in BALANCED else None
                       ),
                       training_sampling=c4_sampling or {'strategy': 'shuffle_without_replacement'})
    learning_rates = {'cnn': args.cnn_lr, 'resnet-head': args.head_lr,
                      'resnet-layer4': {'head': args.head_lr, 'layer4': args.backbone_lr}}
    selected_learning_rates = {model: learning_rates[model] for model in models}
    print(f'{args.experiment} | epochs {epochs_by_model} | learning rates {selected_learning_rates} | '
          f'train {data_config["train_counts"]} | '
          f'selection validation {data_config["selection_validation_counts"]}', flush=True)
    if c4_sampling:
        print(f'C4 balanced sampling | {args.c4_samples_per_epoch} draws per epoch | '
              f'{c4_sampling["weight_formula"]}', flush=True)
        print(f'C4 final validation {data_config["validation_counts"]}', flush=True)
    if args.dry_run:
        return
    import torch

    args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use GPU compute or pass --device cpu.')
    if any(model != 'cnn' for model in models) and not args.weights.is_file():
        parser.error(f'Pretrained ResNet-18 weights missing: {args.weights}. Pass --weights PATH.')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = args.output_root / args.experiment / run_id
    output.mkdir(parents=True)
    results = []
    total_models = len(epochs_by_model)
    for model_number, (model, epochs) in enumerate(epochs_by_model.items(), start=1):
        print(f'\nStarting model {model_number}/{total_models}: {model} ({epochs} epochs)', flush=True)
        result = train_model(args, model, epochs, train_rows, training_rows, sample_weights,
                             selection_validation_rows, final_validation_rows,
                             output / model, data_config)
        selection = result['selection_validation']
        results.append(result)
        results.sort(key=lambda row: (row[f'{selection}_macro_f1'], row[f'{selection}_accuracy']), reverse=True)
        write_csv(output / 'comparison.csv', results)
        print(f'Model {model_number}/{total_models} complete: {model}. '
              f'Updated comparison saved to {output / "comparison.csv"}', flush=True)
    print(f'Top architecture by {selection} validation: {results[0]["architecture"]}', flush=True)
    print(f'Comparison: {output / "comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
