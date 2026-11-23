"""Deterministic splits: stratified 80/10/10 for whole-image datasets, balanced batch folds for TACO."""

from __future__ import annotations

import csv
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def assign_whole_image_splits(df: pd.DataFrame, seed: int, ratios: dict[str, float]) -> pd.Series:
    """Stratify by (dataset, original_label); byte-identical images share a split.

    Random is seeded with a string, which is hashed with SHA-512 and therefore stable
    across processes, unlike the built-in hash().
    """
    names = list(ratios)
    fractions = np.array([ratios[n] for n in names], dtype=float)
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {fractions.sum()}")

    key = df["content_md5"] if "content_md5" in df else df["image_path"]
    split = pd.Series(index=df.index, dtype=object)
    for (dataset, label), group in df.groupby(["dataset", "original_label"], sort=True):
        units = sorted(key[group.index].unique())
        random.Random(f"{seed}/{dataset}/{label}").shuffle(units)
        bounds = np.round(np.cumsum(fractions) * len(units)).astype(int)
        unit_split = {}
        start = 0
        for name, end in zip(names, bounds):
            unit_split.update(dict.fromkeys(units[start:end], name))
            start = end
        split[group.index] = key[group.index].map(unit_split)
    return split


def taco_batch_counts(manifest: pd.DataFrame, common_labels: list[str]) -> pd.DataFrame:
    """Per-batch quantities the folds should balance: images, detector boxes, classifier crops per class."""
    taco = manifest[manifest["dataset"] == "taco"]
    counts = pd.DataFrame({
        "images": taco.groupby("taco_batch")["source_image_id"].nunique(),
        "boxes": taco[taco["use_detector_eval"]].groupby("taco_batch").size(),
    })
    eligible = taco[taco["use_classifier_eval"]]
    per_class = pd.crosstab(eligible["taco_batch"], eligible["common_label"]).reindex(columns=common_labels)
    return counts.join(per_class).fillna(0).astype(int).sort_index()


def _equal_partitions(items: tuple, size: int):
    """All ways to cut items into unordered groups of `size` (each group anchored on its smallest item)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for others in combinations(rest, size - 1):
        group = (first, *others)
        remaining = tuple(i for i in rest if i not in others)
        for tail in _equal_partitions(remaining, size):
            yield [group, *tail]


def make_taco_folds(counts: pd.DataFrame, n_folds: int = 3) -> tuple[dict[int, int], float]:
    """Exhaustively search equal-size batch groupings; minimize the worst relative deviation
    of any balanced quantity from its per-fold target. Returns ({batch: fold}, score)."""
    batches = tuple(sorted(counts.index))
    if len(batches) % n_folds:
        raise ValueError(f"{len(batches)} batches cannot form {n_folds} equal folds")
    values = counts.loc[list(batches)].to_numpy(dtype=float)
    target = values.sum(axis=0) / n_folds
    target[target == 0] = 1.0
    row = {b: i for i, b in enumerate(batches)}

    best, best_score = None, np.inf
    for partition in _equal_partitions(batches, len(batches) // n_folds):
        sums = np.stack([values[[row[b] for b in group]].sum(axis=0) for group in partition])
        score = float(np.max(np.abs(sums - target) / target))
        if score < best_score:
            best, best_score = partition, score
    return {b: fold for fold, group in enumerate(best) for b in group}, best_score


def save_taco_folds(folds: dict[int, int], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "fold"])
        for batch in sorted(folds):
            writer.writerow([batch, folds[batch]])


def load_taco_folds(path: Path) -> dict[int, int]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {int(r["batch"]): int(r["fold"]) for r in csv.DictReader(f)}
