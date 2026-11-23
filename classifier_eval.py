"""Score trained C1–C4 checkpoints on data never used for training or checkpoint selection.

Evaluation sets come from the shared classifier/index.csv:
  <dataset>_test      in-domain test split of TrashNet, RealWaste and SortWaste
  taco                all TACO crops (external test; never used for training)
  taco_no_cigarettes  TACO without 'Cigarette' (77% of TACO trash crops)
  taco_shared3        TACO crops whose true class is cardboard, metal or plastic (the classes
                      SortWaste has; used to compare with end-to-end YOLO)
  taco_fold<k>        TACO fold k, for comparing with the detector pipeline per fold
  <dataset>_val       TrashNet/RealWaste validation, only to check that loading reproduces the
                      validation scores stored in the checkpoints

Macro-F1 averages over the classes present in the evaluated set (as in train_classifier.py),
so SortWaste and taco_shared3 average over 3 classes while predictions still span all 6.
Torch-free so the bookkeeping can be tested without a GPU; the notebook does the inference.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from classifier import CLASSES

IN_DOMAIN = ("trashnet", "realwaste", "sortwaste")
SHARED3 = ("cardboard", "metal", "plastic")
VAL_CHECK = ("trashnet", "realwaste")


def check_index(index: pd.DataFrame) -> None:
    ids = dict(zip(index["label"], index["label_id"]))
    expected = {label: i for i, label in enumerate(CLASSES)}
    wrong = {label: i for label, i in ids.items() if expected.get(label) != i}
    if wrong:
        raise ValueError(f"index.csv label ids differ from the checkpoints' class order {CLASSES}: {wrong}")


def eval_sets(index: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over index rows, one per evaluation set."""
    taco = index["dataset"] == "taco"
    sets = {f"{d}_test": (index["dataset"] == d) & (index["partition"] == "test") for d in IN_DOMAIN}
    sets["taco"] = taco
    sets["taco_no_cigarettes"] = taco & (index["original_label"] != "Cigarette")
    sets["taco_shared3"] = taco & index["label"].isin(SHARED3)
    for fold in sorted(index.loc[taco, "partition"].unique()):
        sets[f"taco_{fold}"] = taco & (index["partition"] == fold)
    sets |= {f"{d}_val": (index["dataset"] == d) & (index["partition"] == "val") for d in VAL_CHECK}
    return {name: mask.to_numpy() for name, mask in sets.items() if mask.any()}


def eval_rows(index: pd.DataFrame) -> pd.DataFrame:
    """Rows in at least one evaluation set; every model is run once over these."""
    keep = np.logical_or.reduce(list(eval_sets(index).values()))
    return index[keep].reset_index(drop=True)


def load_images(root: Path, paths, size, resample, workers: int = 32) -> np.ndarray:
    """Decode and resize every crop once (N x H x W x 3 uint8).

    Same result as the deterministic torchvision Resize((H, W)) used for validation in
    classifier.image_transform, which calls PIL's resize with the same filter.
    """
    from PIL import Image

    height, width = size

    def load(relative):
        with Image.open(Path(root) / relative) as image:
            return np.asarray(image.convert("RGB").resize((width, height), resample))

    with ThreadPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(load, paths)))


def score(targets: np.ndarray, predictions: np.ndarray):
    """Same definitions as sklearn's precision_recall_fscore_support with zero_division=0."""
    k = len(CLASSES)
    matrix = np.bincount(targets * k + predictions, minlength=k * k).reshape(k, k)  # rows = true
    hits, support, predicted = np.diag(matrix), matrix.sum(axis=1), matrix.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, hits / predicted, 0.0)
        recall = np.where(support > 0, hits / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    summary = dict(n=int(len(targets)), accuracy=float(np.mean(targets == predictions)),
                   macro_f1=float(f1[present].mean()), balanced_accuracy=float(recall[present].mean()))
    per_class = pd.DataFrame(dict(label=CLASSES, precision=precision, recall=recall, f1=f1, support=support))
    return summary, per_class, matrix


def score_model(rows: pd.DataFrame, predictions: np.ndarray, model: dict):
    """Metrics, per-class and long-form confusion tables for one model on every evaluation set."""
    metrics, per_class, confusion = [], [], []
    targets = rows["label_id"].to_numpy()
    for name, mask in eval_sets(rows).items():
        summary, classes, matrix = score(targets[mask], predictions[mask])
        metrics.append({**model, "eval_set": name, **summary})
        per_class.append(classes.assign(**model, eval_set=name))
        true, pred = np.nonzero(matrix)
        confusion.append(pd.DataFrame({**model, "eval_set": name, "true": np.array(CLASSES)[true],
                                       "pred": np.array(CLASSES)[pred], "count": matrix[true, pred]}))
    return pd.DataFrame(metrics), pd.concat(per_class), pd.concat(confusion)


def mark_selected(models: pd.DataFrame) -> pd.Series:
    """Per experiment and architecture, the run with the best stored validation score.

    Uses only the validation metrics saved in each checkpoint (macro-F1, then accuracy), the
    same rule train_classifier.py used, so test results never influence which model is reported.
    """
    ranked = models.sort_values(["val_macro_f1", "val_accuracy"], ascending=False)
    best = ranked.groupby(["experiment", "architecture"]).head(1).index
    return models.index.isin(best)


KEY = ["experiment", "architecture"]


def paper_tables(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sel = metrics[metrics["selected"]]

    def wide(sets, value):
        return (sel[sel["eval_set"].isin(sets)].pivot_table(index=KEY, columns="eval_set", values=value)
                .reindex(columns=[s for s in sets if s in set(sel["eval_set"])]))

    in_domain = [f"{d}_test" for d in IN_DOMAIN]
    test_f1 = wide(in_domain, "macro_f1").add_suffix(" F1")
    test_acc = wide(in_domain, "accuracy").add_suffix(" acc")
    taco = wide(["taco", "taco_no_cigarettes", "taco_shared3"], "macro_f1").add_suffix(" F1")
    taco["taco acc"] = wide(["taco"], "accuracy")["taco"]
    folds = sel[sel["eval_set"].str.fullmatch(r"taco_fold\d+")].groupby(KEY)["macro_f1"]
    taco["fold F1 mean"], taco["fold F1 std"] = folds.mean(), folds.std()
    check = sel[sel["eval_set"].str.endswith("_val")][KEY + ["eval_set", "macro_f1", "selection_validation",
                                                          "val_macro_f1"]]
    check = check[check["eval_set"] == check["selection_validation"] + "_val"]
    return {"in_domain_test": pd.concat([test_f1, test_acc], axis=1).round(3),
            "taco": taco.round(3),
            "val_reproduction": check.round(4)}


def per_class_table(per_class: pd.DataFrame, eval_set: str = "taco") -> pd.DataFrame:
    """F1 per class (columns) for the selected models on one evaluation set."""
    rows = per_class[per_class["selected"] & (per_class["eval_set"] == eval_set)]
    table = rows.pivot_table(index=KEY, columns="label", values="f1").reindex(columns=list(CLASSES))
    support = rows.drop_duplicates("label").set_index("label")["support"].reindex(list(CLASSES))
    table.loc[("support", ""), :] = support.to_numpy()
    return table.round(3)
