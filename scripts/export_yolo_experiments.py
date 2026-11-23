"""Add the YOLO experiment data to an existing processed/yolo_classes/ folder.

Writes, next to the SortWaste and TACO data already there (which it does not touch):
  images/ and labels/ for TrashNet and RealWaste: each photo is one box covering the whole photo
  trashnet_<split>.txt, realwaste_<split>.txt   image lists
  taco_heldout.txt                              the 177 TACO test photos of the TACO-trained YOLO
  yolo_sortwaste.yaml                           train/val on SortWaste
  yolo_sortwaste_trashnet_realwaste.yaml        train/val on SortWaste + TrashNet + RealWaste
Every image count is checked against the expected totals.

    python scripts/export_yolo_experiments.py --out data/processed
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from build_manifest import add_path_arguments  # noqa: E402
from data_preprocessing import DataPaths, load_config, read_manifest, validate_manifest  # noqa: E402
from data_preprocessing.export import (TACO_HELDOUT_LIST, WHOLE_IMAGE_DATASETS,  # noqa: E402
                                       experiment_yamls, export_whole_images, taco_heldout_paths,
                                       yolo_class_names)

DONE_MARKER = "_SUCCESS_experiments"
BASE_MARKER = "_SUCCESS"  # written by export_datasets.py when yolo_classes/ is complete
EXPECTED_IMAGES = {
    "sortwaste_train": 3705, "sortwaste_val": 780, "sortwaste_test": 776,
    "trashnet_train": 2018, "trashnet_val": 251, "trashnet_test": 252,
    "realwaste_train": 2870, "realwaste_val": 359, "realwaste_test": 358,
    "taco_all": 1500, "taco_heldout": 177,
}


def count_lists(folder: Path) -> dict[str, int]:
    return {name: len((folder / f"{name}.txt").read_text().splitlines())
            for name in EXPECTED_IMAGES if (folder / f"{name}.txt").exists()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    add_path_arguments(parser)
    args = parser.parse_args(argv)

    folder = args.out / "yolo_classes"
    base = folder / BASE_MARKER
    have = set(base.read_text().strip().split(",")) if base.exists() else set()
    if not {"sortwaste", "taco"} <= have:
        parser.error(f"{folder} needs SortWaste and TACO first (run export_datasets.py --only yolo_classes)")

    paths = DataPaths(REPO_ROOT, args.raw_dir, args.taco_dir, args.manifest_dir)
    config = load_config(paths.config)
    manifest = read_manifest(paths.manifests / "unified_manifest.csv")
    validate_manifest(manifest, config)

    (folder / DONE_MARKER).unlink(missing_ok=True)
    for dataset in WHOLE_IMAGE_DATASETS:  # rebuild these two cleanly; SortWaste/TACO stay as they are
        for kind in ("images", "labels"):
            shutil.rmtree(folder / kind / dataset, ignore_errors=True)
    index = export_whole_images(manifest, paths, folder, args.workers)
    print(f"whole-photo boxes: {len(index):,} photos")

    heldout = taco_heldout_paths((folder / "taco_all.txt").read_text())
    (folder / TACO_HELDOUT_LIST).write_text("".join(f"{line}\n" for line in heldout))
    for name, text in experiment_yamls(yolo_class_names(config["common_labels"])).items():
        (folder / name).write_text(text)

    counts = count_lists(folder)
    wrong = {k: (counts.get(k), n) for k, n in EXPECTED_IMAGES.items() if counts.get(k) != n}
    if wrong:
        raise RuntimeError(f"image counts differ from expected (found, expected): {wrong}")
    (folder / DONE_MARKER).write_text("".join(f"{k}={v}\n" for k, v in counts.items()))
    print("\n".join(f"  {k:16s} {v:6,}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
