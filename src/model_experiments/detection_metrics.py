"""Score detections that carry a material label: end-to-end YOLO and the same boxes re-labelled.

Rules (fixed before any test result was seen):
  * A prediction is a true positive when it overlaps an unmatched true box of the SAME class by
    IoU >= 0.5; predictions of a class are matched greedily from the highest score (COCO/VOC style).
  * Drink cartons (class 6) are not scored: a prediction that is not a true positive but overlaps a
    carton box by IoU >= 0.5 is ignored (neither right nor wrong), like COCO's crowd regions.
  * Average precision per class uses COCO's 101-point interpolation at IoU 0.5.
  * Precision / recall / F1 use the predictions whose detector confidence is >= 0.25.
  * Class averages weight every class equally, over the classes present in the test set.
  * Confidence intervals resample test photos with replacement (detections stay with their photo).
Boxes are normalized [x1, y1, x2, y2].
"""

from __future__ import annotations

import numpy as np

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash", "carton")
CARTON = 6
SHARED3 = (0, 2, 4)  # cardboard, metal, plastic: the materials SortWaste has
ALL6 = (0, 1, 2, 3, 4, 5)
IOU_MATCH = 0.5
F1_CONFIDENCE = 0.25


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def agnostic_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Indices kept after class-agnostic NMS, so every object has one box whatever its label."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ious = iou_matrix(boxes, boxes)
    keep, removed = [], np.zeros(len(order), bool)
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        removed |= ious[i] > threshold
    return np.array(keep, int)


def match(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels) -> np.ndarray:
    """Per prediction: 1 = true positive, 0 = false positive, -1 = ignored (overlaps a carton)."""
    pred_labels, pred_scores, gt_labels = (np.asarray(x) for x in (pred_labels, pred_scores, gt_labels))
    flags = np.zeros(len(pred_labels), int)
    if not len(pred_labels):
        return flags
    ious = iou_matrix(pred_boxes, gt_boxes)
    carton = gt_labels == CARTON
    for c in np.unique(pred_labels):
        candidates = np.flatnonzero(gt_labels == c)
        used = np.zeros(len(gt_labels), bool)
        for i in np.flatnonzero(pred_labels == c)[np.argsort(-pred_scores[pred_labels == c], kind="stable")]:
            free = candidates[~used[candidates]]
            if len(free) and ious[i, free].max() >= IOU_MATCH:
                used[free[ious[i, free].argmax()]] = True
                flags[i] = 1
            elif carton.any() and ious[i, carton].max() >= IOU_MATCH:
                flags[i] = -1
    return flags


def average_precision(scores, flags, n_gt, weights=None) -> float:
    """COCO 101-point AP from scored predictions (flags 1/0/-1) and the (weighted) number of true boxes."""
    scores, flags = np.asarray(scores, float), np.asarray(flags)
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights, float)
    if n_gt <= 0:
        return float("nan")
    keep = (flags >= 0) & (weights > 0)
    order = np.argsort(-scores[keep], kind="stable")
    tp = np.cumsum((flags[keep] == 1)[order] * weights[keep][order])
    fp = np.cumsum((flags[keep] == 0)[order] * weights[keep][order])
    if not len(tp):
        return 0.0
    recall, precision = tp / n_gt, tp / np.maximum(tp + fp, np.finfo(float).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0).mean())


def prf(flags, n_gt, weights=None) -> tuple[float, float, float]:
    flags = np.asarray(flags)
    weights = np.ones(len(flags)) if weights is None else np.asarray(weights, float)
    tp, fp = float(((flags == 1) * weights).sum()), float(((flags == 0) * weights).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if n_gt and precision + recall else 0.0
    return precision, recall, f1


class Scored:
    """One system on one test set: matched once, then scored for any resampling of the photos."""

    def __init__(self, images: list[dict], labels_key: str, score_key: str):
        """images: dicts with gt_boxes, gt_labels, boxes, det_conf and the label/score arrays named."""
        self.n_images = len(images)
        image_ids, labels, scores, flags, f1_flags = [], [], [], [], []
        self.gt_counts = np.zeros((len(images), len(CLASSES)))
        for k, im in enumerate(images):
            self.gt_counts[k] = np.bincount(np.asarray(im["gt_labels"], int), minlength=len(CLASSES))
            lab, sc = np.asarray(im[labels_key], int), np.asarray(im[score_key], float)
            image_ids.append(np.full(len(lab), k))
            labels.append(lab)
            scores.append(sc)
            flags.append(match(im["boxes"], lab, sc, im["gt_boxes"], im["gt_labels"]))
            confident = np.asarray(im["det_conf"]) >= F1_CONFIDENCE  # same boxes for every labeller
            f = np.full(len(lab), -2)  # -2 = below the confidence cut, not counted
            f[confident] = match(np.asarray(im["boxes"]).reshape(-1, 4)[confident], lab[confident], sc[confident],
                                 im["gt_boxes"], im["gt_labels"])
            f1_flags.append(f)
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
        self.image_ids, self.labels, self.scores = cat(image_ids).astype(int), cat(labels).astype(int), cat(scores)
        self.flags, self.f1_flags = cat(flags).astype(int), cat(f1_flags).astype(int)
        # per class, predictions sorted by score once (resampling only changes weights)
        self._ranked = {c: np.flatnonzero(self.labels == c)[np.argsort(-self.scores[self.labels == c], kind="stable")]
                        for c in range(len(CLASSES))}

    def per_class(self, image_weights=None) -> dict[int, dict]:
        w_img = np.ones(self.n_images) if image_weights is None else np.asarray(image_weights, float)
        w = w_img[self.image_ids]
        n_gt = (self.gt_counts * w_img[:, None]).sum(axis=0)
        out = {}
        for c in range(len(CLASSES)):
            ranked = self._ranked[c]
            counted = ranked[self.f1_flags[ranked] != -2]
            precision, recall, f1 = prf(self.f1_flags[counted], n_gt[c], w[counted])
            ap = average_precision(-np.arange(len(ranked), dtype=float), self.flags[ranked], n_gt[c], w[ranked])
            out[c] = dict(n_gt=float(n_gt[c]), ap50=ap, precision=precision, recall=recall, f1=f1)
        return out

    def summary(self, classes, image_weights=None) -> dict[str, float]:
        """Equal-weight class means over the classes present (n_gt > 0)."""
        rows = self.per_class(image_weights)
        present = [c for c in classes if rows[c]["n_gt"] > 0]
        mean = lambda key: float(np.mean([rows[c][key] for c in present])) if present else float("nan")  # noqa: E731
        return {"map50": mean("ap50"), "macro_f1": mean("f1"), "macro_precision": mean("precision"),
                "macro_recall": mean("recall"), "classes": len(present)}


def bootstrap_weights(n_images: int, repeats: int = 1000, seed: int = 42) -> np.ndarray:
    """Photo multiplicities for each resample (shared by systems, so differences are paired)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=repeats).astype(float)


def interval(values) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if len(values) else (np.nan, np.nan)
