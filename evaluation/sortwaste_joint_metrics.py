"""Detection metrics for the original three-material SortWaste comparison."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

LABELS = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'carton')
SHARED = (0, 2, 4)


def overlaps(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0)
    box = np.asarray(box)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(1)
    union = np.maximum(0, box[2:] - box[:2]).prod() + np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1) - intersection
    return intersection / np.maximum(union, 1e-12)


def prepare(images, system):
    prepared = []
    for label in SHARED:
        support, detections = [], []
        for photo_index, image in enumerate(images):
            truth = [t['box'] for t in image['targets'] if t['class_id'] == label]
            support.append(len(truth))
            candidates = sorted((p for p in image[system] if p['class_id'] == label),
                                key=lambda p: p['confidence'], reverse=True)
            used = set()
            for prediction in candidates:
                ious = overlaps(prediction['box'], truth)
                if used:
                    ious[list(used)] = -1
                target = int(ious.argmax()) if len(ious) else -1
                correct = target >= 0 and ious[target] >= .5
                if correct:
                    used.add(target)
                detections.append((prediction['confidence'], correct, photo_index))
        detections.sort(key=lambda r: r[0], reverse=True)
        prepared.append((np.asarray(support), np.asarray([d[0] for d in detections]),
                         np.asarray([d[1] for d in detections], dtype=float),
                         np.asarray([d[2] for d in detections], dtype=int)))
    return prepared


def ap_value(recall, precision, method='coco101'):
    if not len(recall):
        return 0.0
    if method == 'coco101':
        return float(np.mean([precision[recall >= threshold].max(initial=0.0)
                              for threshold in np.linspace(0, 1, 101)]))
    from ultralytics.utils.metrics import compute_ap
    if method == 'ultralytics':
        return float(compute_ap(recall, precision)[0])
    if method == 'voc':
        r = np.concatenate(([0.], recall, [1.]))
        p = np.concatenate(([0.], precision, [0.]))
        p = np.maximum.accumulate(p[::-1])[::-1]
        indices = np.where(r[1:] != r[:-1])[0]
        return float(np.sum((r[indices+1] - r[indices]) * p[indices+1]))
    raise ValueError(method)


def score(prepared, weights=None, ap_method='coco101'):
    if weights is None:
        weights = np.ones(len(prepared[0][0]), dtype=int)
    rows = []
    for label, (support, confidence, truth, photo_index) in zip(SHARED, prepared):
        count = int(weights @ support)
        importance = weights[photo_index]
        keep = importance > 0
        true_positive = np.cumsum((truth * importance)[keep])
        false_positive = np.cumsum(((1 - truth) * importance)[keep])
        recall = true_positive / max(count, 1)
        precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
        selected = confidence >= .25
        correct = float((truth * importance)[selected].sum())
        predicted = int(importance[selected].sum())
        rows.append({'class': LABELS[label], 'n_gt': count,
                     'ap50': ap_value(recall, precision, ap_method) if count else 0.,
                     'precision': correct / predicted if predicted else 0.,
                     'recall': correct / count if count else 0.,
                     'f1': 2 * correct / (predicted + count) if predicted + count else 0.,
                     'true_positives': int(correct), 'predicted_boxes': predicted})
    return {'map50': float(np.mean([r['ap50'] for r in rows])),
            'macro_f1': float(np.mean([r['f1'] for r in rows])),
            'macro_precision': float(np.mean([r['precision'] for r in rows])),
            'macro_recall': float(np.mean([r['recall'] for r in rows])), 'per_class': rows}


def summarize(images, systems=('yolo', 'pipeline'), ap_method='coco101', repetitions=0):
    prepared = {system: prepare(images, system) for system in systems}
    results = {system: score(value, ap_method=ap_method) for system, value in prepared.items()}
    if repetitions:
        rng = np.random.default_rng(42)
        samples = {s: {m: [] for m in ('map50', 'macro_f1', 'macro_precision', 'macro_recall')} for s in systems}
        for _ in range(repetitions):
            weights = rng.multinomial(len(images), np.full(len(images), 1/len(images)))
            for system in systems:
                values = score(prepared[system], weights, ap_method)
                for metric in samples[system]:
                    samples[system][metric].append(values[metric])
        for system in systems:
            results[system]['intervals_95'] = {metric: np.quantile(values, [.025, .975]).tolist()
                                               for metric, values in samples[system].items()}
    return {'systems': results, 'test_images': len(images), 'ap_method': ap_method,
            'protocol': {'classes': [LABELS[i] for i in SHARED], 'iou_match': .5,
                         'f1_confidence': .25, 'predict_confidence': .01, 'max_boxes_per_photo': 100,
                         'image_size': 640, 'agnostic_nms_iou': .7,
                         'hybrid_score': 'YOLO confidence multiplied by ResNet max softmax',
                         'bootstrap': repetitions}}


def baseline_reproduction(images, prediction_file, summary_file, per_class_file):
    """Verify the scoring implementation against the unmodified saved baseline."""
    by_photo = {image['image']: {'image': image['image'], 'targets': image['targets'],
                                'yolo': [], 'pipeline': []} for image in images}
    with gzip.open(prediction_file, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['photo'] not in by_photo:
                continue
            target = by_photo[row['photo']]
            box = [float(row[k]) for k in ['x1', 'y1', 'x2', 'y2']]
            target['yolo'].append({'box': box, 'class_id': int(row['yolo_label']), 'confidence': float(row['yolo_score'])})
            target['pipeline'].append({'box': box, 'class_id': int(row['classifier_all_label']),
                                       'confidence': float(row['classifier_all_score'])})
    with Path(summary_file).open(newline='', encoding='utf-8-sig') as stream:
        expected = {row['system']: row for row in csv.DictReader(stream)
                    if row['test_set'] == 'sortwaste_test' and row['classes_scored'] == 'shared3'}
    with Path(per_class_file).open(newline='', encoding='utf-8-sig') as stream:
        expected_classes = {(row['system'], row['class']): row for row in csv.DictReader(stream)
                            if row['test_set'] == 'sortwaste_test' and row['class'] in [LABELS[i] for i in SHARED]}
    prepared = {system: prepare(list(by_photo.values()), system) for system in ['yolo','pipeline']}
    methods = {}
    for method in ['coco101', 'ultralytics', 'voc']:
        measured = {s: score(value, ap_method=method) for s, value in prepared.items()}
        deviations = {}
        for system, original in [('yolo','e2e_all'), ('pipeline','pipeline_all')]:
            for metric in ['map50','macro_f1','macro_precision','macro_recall']:
                deviations[system + '/' + metric] = measured[system][metric] - float(expected[original][metric])
            for row in measured[system]['per_class']:
                for metric in ['ap50','precision','recall','f1']:
                    deviations[system + '/' + row['class'] + '/' + metric] = row[metric] - float(expected_classes[(original,row['class'])][metric])
        methods[method] = {'max_absolute_deviation': max(map(abs, deviations.values())),
                           'deviations': deviations, 'measured': measured}
    chosen = min(methods, key=lambda name: methods[name]['max_absolute_deviation'])
    return {'passed': methods[chosen]['max_absolute_deviation'] < 1e-8,
            'selected_ap_method': chosen, 'methods': methods,
            'description': 'Scoring verified against archived baseline predictions, without fitting or tuning a model.'}


def save_predictions(path, images):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(images, stream)
