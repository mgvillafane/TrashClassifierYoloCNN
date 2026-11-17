"""Build and validate data/manifests/unified_manifest.csv from the local raw datasets.

Run from anywhere inside the repository:

    python scripts/build_manifest.py                   # manifest + summary + figures
    python scripts/build_manifest.py --make-taco-folds # one-time: write configs/taco_folds.csv first

Data elsewhere (e.g. Databricks volumes): pass --raw-dir / --taco-dir and --manifest-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_preprocessing import (DataPaths, build_manifest, class_table, exclusion_table,  # noqa: E402
                                inventory, load_config, summarize, validate_manifest)
from data_preprocessing.splits import make_taco_folds, save_taco_folds, taco_batch_counts  # noqa: E402


def add_path_arguments(parser) -> None:
    parser.add_argument("--raw-dir", type=Path, help="folder with realwaste/, trashnet/, sortwaste/ (default data/raw)")
    parser.add_argument("--taco-dir", type=Path, help="folder with TACO annotations.json and batch_* (default data/raw/taco/data)")
    parser.add_argument("--manifest-dir", type=Path, help="where the manifest is written/read (default data/manifests)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-taco-folds", action="store_true",
                        help="search balanced TACO batch folds and write configs/taco_folds.csv")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing taco_folds.csv")
    parser.add_argument("--no-figures", action="store_true")
    add_path_arguments(parser)
    args = parser.parse_args(argv)

    paths = DataPaths(REPO_ROOT, args.raw_dir, args.taco_dir, args.manifest_dir)
    missing = paths.missing()
    if missing:
        parser.error("missing inputs:\n  " + "\n  ".join(map(str, missing)))
    config = load_config(paths.config)

    folds_file = paths.root / config["splits"]["taco_folds_file"]
    if args.make_taco_folds:
        if folds_file.exists() and not args.force:
            parser.error(f"{folds_file} exists; folds must stay fixed once results exist (use --force to replace)")
        draft = build_manifest(paths, config)
        counts = taco_batch_counts(draft, config["common_labels"])
        folds, score = make_taco_folds(counts, config["splits"]["taco_n_folds"])
        save_taco_folds(folds, folds_file)
        print(f"wrote {folds_file.relative_to(paths.root)} (worst relative imbalance {score:.1%})")

    manifest = build_manifest(paths, config)
    validate_manifest(manifest, config)

    paths.manifests.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths.manifests / "unified_manifest.csv", index=False)
    summarize(manifest).to_csv(paths.manifests / "summary.csv", index=False)
    table = class_table(manifest, config)
    table.to_csv(paths.manifests / "class_table.csv")
    exclusion_table(manifest).to_csv(paths.manifests / "exclusions.csv", index=False)

    print(f"{len(manifest):,} rows -> {paths.manifests / 'unified_manifest.csv'}")
    print(inventory(manifest, config).to_string())
    print("\nclassifier-eligible examples:\n" + table.to_string())
    if manifest["taco_fold"].isna().all():
        print(f"\nnote: {folds_file.relative_to(paths.root)} not found; TACO rows have no fold yet")

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")  # files only, no GUI window
        from data_preprocessing.report import save_figures
        for path in save_figures(manifest, table, paths.manifests / "figures", paths, config["common_labels"]):
            print("figure:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
