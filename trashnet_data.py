"""Shared TrashNet splits and loaders for CNN and ResNet experiments.

Run once: python trashnet_data.py
In a training script:
    from trashnet_data import get_dataloaders
    loaders, classes = get_dataloaders()
    for images, labels in loaders['train']:
        ...

The CSV stores archive member paths, class names, labels, and split membership.
Keep this file for all experiments. Pass manifest_path pointing to a writable
Volume location if the split needs to be shared across Databricks jobs.
"""

import csv
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

DEFAULT_ROOT = Path('/Volumes/workspace/default/trashnet_data')
DEFAULT_MANIFEST = Path(__file__).with_name('trashnet_split.csv')
FIELDS = ['path', 'class_name', 'label', 'split']


def create_split(data_root=DEFAULT_ROOT, manifest_path=DEFAULT_MANIFEST, seed=42):
    """Create a fixed stratified 70/15/15 split; reuse an existing CSV unchanged."""
    from sklearn.model_selection import train_test_split

    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        return manifest_path
    with ZipFile(Path(data_root) / 'dataset-resized.zip') as archive:
        paths = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            and '__MACOSX' not in name
            and not Path(name).name.startswith('.')
        )
    if not paths:
        raise ValueError('No images found in dataset-resized.zip')
    classes = sorted({Path(name).parent.name for name in paths})
    labels = [classes.index(Path(name).parent.name) for name in paths]
    train, held_out = train_test_split(
        list(range(len(paths))), test_size=0.30, stratify=labels, random_state=seed
    )
    val, test = train_test_split(
        held_out, test_size=0.5, stratify=[labels[i] for i in held_out],
        random_state=seed,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for split, indices in [('train', train), ('val', val), ('test', test)]:
            for i in sorted(indices):
                writer.writerow(dict(path=paths[i], class_name=classes[labels[i]],
                                     label=labels[i], split=split))
    return manifest_path


class TrashNetDataset:
    """Map-style PyTorch dataset. Opens a separate ZIP handle per image/worker."""

    def __init__(self, data_root, rows, transform=None):
        self.archive_path = Path(data_root) / 'dataset-resized.zip'
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        with ZipFile(self.archive_path) as archive:
            with archive.open(row['path']) as handle:
                image = Image.open(handle).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row['label'])


def get_dataloaders(data_root=DEFAULT_ROOT, manifest_path=DEFAULT_MANIFEST,
                    batch_size=32, image_size=224, num_workers=0, seed=42,
                    train_transform=None, eval_transform=None):
    """Return ({'train': loader, 'val': loader, 'test': loader}, class_names).

    Defaults use ImageNet normalization for both models and augmentation only
    for training. Supply custom transforms when an experiment requires them.
    seed controls new splits and training shuffling; existing splits are reused.
    num_workers=0 is a portable starting point for Databricks notebooks.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    manifest = create_split(data_root, manifest_path, seed)
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Split manifest must contain unique image paths.')
    if {row['split'] for row in rows} != {'train', 'val', 'test'}:
        raise ValueError('Split manifest must contain train, val, and test.')
    mapping = {int(row['label']): row['class_name'] for row in rows}
    if sorted(mapping) != list(range(len(mapping))) or any(
        mapping[int(row['label'])] != row['class_name'] for row in rows
    ):
        raise ValueError('Inconsistent class labels in split manifest.')
    classes = [mapping[i] for i in range(len(mapping))]
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
        ])
    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), normalize,
        ])
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = TrashNetDataset(
            data_root, [row for row in rows if row['split'] == split],
            train_transform if split == 'train' else eval_transform,
        )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders, classes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest-path', type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = create_split(args.data_root, args.manifest_path)
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    print(f'Saved split: {manifest}')
    for split in ['train', 'val', 'test']:
        counts = Counter(row['class_name'] for row in rows if row['split'] == split)
        print(f'{split}: {sum(counts.values())} images — {dict(sorted(counts.items()))}')
