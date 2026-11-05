"""Preview the shared loader: python tests/preview_trashnet.py

Saves a labeled image grid and displays it when a graphical backend is available.
"""

import argparse
from pathlib import Path
import sys

# Allow running this file directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trashnet_data import get_dataloaders


def main():
    import matplotlib.pyplot as plt
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'reports' / 'trashnet_preview.png')
    parser.add_argument('--no-show', action='store_true', help='Save without opening a window.')
    args = parser.parse_args()

    loaders, classes = get_dataloaders(batch_size=12)
    images, labels = next(iter(loaders[args.split]))
    print(f'Classes: {classes}')
    print(f'{args.split} batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}')

    # Reverse the loader's ImageNet normalization for display.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_images = (images.cpu() * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for index, ax in enumerate(axes.flat):
        ax.axis('off')
        if index < len(images):
            ax.imshow(display_images[index].permute(1, 2, 0).numpy())
            ax.set_title(classes[labels[index].item()])
    fig.suptitle(f'TrashNet — {args.split} split (ground-truth labels)')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved preview: {args.output}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
