"""Export ready-to-train classifier crops and single-class YOLO data to data/processed/.

Reads data/manifests/unified_manifest.csv (run scripts/build_manifest.py first) and the raw
images. The output folder is self-contained, so it can be zipped and shared:

    python scripts/export_datasets.py --workers 8
    python scripts/export_datasets.py --only classifier --datasets trashnet realwaste
    python scripts/export_datasets.py --only yolo_classes   # end-to-end YOLO baselines E1/E2

Data elsewhere (e.g. Databricks volumes): pass the same --raw-dir / --taco-dir / --manifest-dir
as scripts/build_manifest.py, and --out.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from build_manifest import add_path_arguments  # noqa: E402
from data_preprocessing import DataPaths, load_config, read_manifest, validate_manifest  # noqa: E402
from data_preprocessing.export import (CLASSIFIER_MAX_SIDE, DETECTOR_MAX_SIDE,  # noqa: E402
                                       export_classifier, export_detector, label_path, yolo_class_names)

README = """# Processed waste datasets

Built from `unified_manifest.csv` by `scripts/export_datasets.py`. Labels: {labels}.

## classifier/ (material classifier, ground-truth crops)

`<dataset>/<partition>/<label>/<name>.jpg`, longest side at most {cls_side} px (whole images for
RealWaste/TrashNet, exact box crops without padding for SortWaste/TACO).

{cls_table}

- Use `index.csv` (`path`, `label_id`, ...) rather than torchvision `ImageFolder`: SortWaste has no
  glass/paper/trash folders, so `ImageFolder` would assign it different class ids.
- `label_id` is fixed: {label_ids}.
- **TACO is never used for training** (`use_classifier_train` is False); `fold0..2` together are the
  external test set. Report TACO trash with and without `original_label == "Cigarette"`.
- Combinations: C1 = trashnet, C2 = realwaste, C3 = both, C4 = both + sortwaste. Select models on
  the `val` partitions only.

## detector/ (single class `waste_object`, YOLO format)

Images are upright JPEGs, longest side at most {det_side} px; `labels/` mirrors `images/`.

{det_table}

- `d1_sortwaste.yaml`: D1, train/val/test = SortWaste official split.
- `d2_taco_fold<k>.yaml`: D2, train = SortWaste train + the two other TACO folds, val = SortWaste
  val, test = TACO fold k. Run all three folds and average.
- D1 on TACO fold k: `model.val(data="d2_taco_fold<k>.yaml", split="test")`, the same test images
  as D2 fold k (D1 never sees TACO). `taco_all.txt` lists every TACO image.
- The YAMLs have no `path`; Ultralytics resolves the lists next to the YAML, so keep this folder intact.
- Train with `single_cls=True` (labels are already all class 0; this just makes it explicit).

## yolo_classes/ (end-to-end YOLO baselines, class-aware labels)

Same images and lists as `detector/`, but each box keeps its class: {yolo_ids}. Beverage cartons
(SortWaste `ecal`, TACO `Drink carton`) get their own `carton` class instead of cardboard.

{yolo_table}

- `e1_sortwaste.yaml`: E1, the SortWaste-only end-to-end YOLO (it can only learn cardboard, metal,
  plastic and carton). Test E1 on TACO fold k with `model.val(data="e2_taco_fold<k>.yaml", split="test")`.
- `e2_taco_fold<k>.yaml`: E2, SortWaste + the two other TACO folds, all 7 classes.
- Compare with the detector + classifier pipelines on the shared classes (cardboard, metal,
  plastic): average their per-class AP; ignore `carton` and the classes E1 never saw.

## Using it

On Databricks, `notebooks/databricks_prepare_data.py` builds this folder from the raw data in the
volumes (output defaults to `/tmp/processed` on the cluster's local disk).

Minimal PyTorch dataset over `index.csv`:

```python
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class WasteCrops(Dataset):
    def __init__(self, root, datasets, partition, transform=None):
        index = pd.read_csv(f"{{root}}/index.csv")
        self.root, self.transform = root, transform
        self.items = index[index["dataset"].isin(datasets) & (index["partition"] == partition)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row = self.items.iloc[i]
        image = Image.open(f"{{self.root}}/{{row.path}}").convert("RGB")
        return (self.transform(image) if self.transform else image), int(row.label_id)

train = WasteCrops("/tmp/processed/classifier", ["trashnet", "realwaste"], "train")   # C3
```

Detector: `YOLO("yolo26m.pt").train(data="/tmp/processed/detector/d1_sortwaste.yaml", single_cls=True, ...)`.
"""


PARTS = ["classifier", "detector", "yolo_classes"]
DONE_MARKER = "_SUCCESS"  # lists the datasets a finished part contains


def markdown_table(df) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(i)] + [f"{v:,}" for v in row]) + " |" for i, row in df.iterrows()]
    return "\n".join(lines)


def classifier_table(out: Path, labels: list[str]):
    """Files per dataset/partition and label, read back from classifier/index.csv."""
    if not (out / "classifier/index.csv").exists():
        return None
    index = pd.read_csv(out / "classifier/index.csv")
    table = index.pivot_table(index=["dataset", "partition"], columns="label", values="path",
                              aggfunc="size", fill_value=0).reindex(columns=labels, fill_value=0)
    table.index = [f"{d} / {p}" for d, p in table.index]
    table.index.name = "dataset / partition"
    return table


def detector_table(out: Path, part: str = "detector", class_names: list[str] | None = None):
    """Images and boxes per image list (and per class), read back from the list and label files."""
    det = out / part
    lists = sorted(f for f in det.glob("*.txt") if f.stem != "taco_all") if det.exists() else []
    if not lists:
        return None
    rows = []
    for f in lists:
        images = f.read_text().splitlines()
        ids = [line.split()[0] for p in images for line in (det / label_path(p[2:])).read_text().splitlines()]
        row = {"list": f.stem, "images": len(images), "boxes": len(ids)}
        for i, name in enumerate(class_names or []):
            row[name] = ids.count(str(i))
        rows.append(row)
    return pd.DataFrame(rows).set_index("list")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--only", choices=PARTS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--datasets", nargs="+", choices=["realwaste", "trashnet", "sortwaste", "taco"],
                        help="export only these datasets (default: all)")
    parser.add_argument("--force", action="store_true", help="delete an existing output folder first")
    add_path_arguments(parser)
    args = parser.parse_args(argv)

    paths = DataPaths(REPO_ROOT, args.raw_dir, args.taco_dir, args.manifest_dir)
    config = load_config(paths.config)
    manifest_file = paths.manifests / "unified_manifest.csv"
    if not manifest_file.exists():
        parser.error(f"{manifest_file} not found; run scripts/build_manifest.py first")
    manifest = read_manifest(manifest_file)
    validate_manifest(manifest, config)
    if args.datasets:
        manifest = manifest[manifest["dataset"].isin(args.datasets)]

    parts = [args.only] if args.only else PARTS
    for part in parts:
        target = args.out / part
        if target.exists():
            if not args.force:
                parser.error(f"{target} exists (use --force to rebuild it)")
            shutil.rmtree(target)

    labels = config["common_labels"]
    exported = sorted(manifest["dataset"].unique())

    def mark_done(part: str) -> None:
        # written last, so a crashed export (no marker) is rebuilt instead of reused
        (args.out / part / DONE_MARKER).write_text(",".join(exported) + "\n")

    if "classifier" in parts:
        index = export_classifier(manifest, paths, args.out / "classifier", args.workers)
        mark_done("classifier")
        print(f"classifier: {len(index):,} files\n{classifier_table(args.out, labels).to_string()}")
    if "detector" in parts:
        det = export_detector(manifest, paths, args.out / "detector", config["splits"]["taco_n_folds"],
                              args.workers)
        mark_done("detector")
        print(f"detector: {len(det):,} images\n{detector_table(args.out).to_string()}")
    yolo_names = yolo_class_names(labels)
    if "yolo_classes" in parts:
        det = export_detector(manifest, paths, args.out / "yolo_classes", config["splits"]["taco_n_folds"],
                              args.workers, class_names=yolo_names)
        mark_done("yolo_classes")
        print(f"yolo_classes: {len(det):,} images\n{detector_table(args.out, 'yolo_classes', yolo_names).to_string()}")

    # describe everything in the folder, including parts exported by an earlier run
    tables = {}
    for name, make in [("cls_table", lambda: classifier_table(args.out, labels)),
                       ("det_table", lambda: detector_table(args.out)),
                       ("yolo_table", lambda: detector_table(args.out, "yolo_classes", yolo_names))]:
        table = make()
        tables[name] = markdown_table(table) if table is not None else "(not exported yet)"
    (args.out / "README.md").write_text(README.format(
        labels=", ".join(labels), cls_side=CLASSIFIER_MAX_SIDE, det_side=DETECTOR_MAX_SIDE,
        label_ids=", ".join(f"{i} = {name}" for i, name in enumerate(labels)),
        yolo_ids=", ".join(f"{i} = {name}" for i, name in enumerate(yolo_names)), **tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
