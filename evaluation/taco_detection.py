"""Six-class, end-to-end detection scores for the frozen-YOLO C4 comparison."""
from __future__ import annotations

from pathlib import Path
import math
import shutil
import tempfile

if __package__:
    from . import sortwaste as base
else:
    import sortwaste as base

SYSTEMS = ("yolo", "yolo_resnet", "yolo_post_resnet")


def class_matches(predictions, targets, label):
    """Match score-ranked predictions to unused ground truth of the same class."""
    truth = [item["box"] for item in targets if item["class_id"] == label]
    candidates = sorted((p for p in predictions if p["class_id"] == label),
                        key=lambda p: p["confidence"], reverse=True)
    used, results = set(), []
    for prediction in candidates:
        choices = [(base.iou(prediction["box"], box), index) for index, box in enumerate(truth) if index not in used]
        overlap, index = max(choices, default=(0, -1))
        correct = overlap >= .5
        if correct:
            used.add(index)
        results.append((prediction["confidence"], correct))
    return len(truth), results


def summarize(images, repetitions=1000):
    import numpy as np
    from ultralytics.utils.metrics import compute_ap

    prepared = {}
    for system in SYSTEMS:
        classes = []
        for label in range(6):
            support, values = [], []
            for position, image in enumerate(images):
                count, matched = class_matches(image[system], image["targets"], label)
                support.append(count)
                values.extend((confidence, correct, position) for confidence, correct in matched)
            values.sort(key=lambda row: row[0], reverse=True)
            classes.append((np.asarray(support), np.asarray([v[0] for v in values]),
                            np.asarray([v[1] for v in values], dtype=float),
                            np.asarray([v[2] for v in values], dtype=int)))
        prepared[system] = classes

    def scores(system, weights):
        rows = []
        for label, (support, confidence, tp, image_index) in enumerate(prepared[system]):
            count = int(weights @ support)
            pred_weights = weights[image_index]
            keep = pred_weights > 0
            tpc = np.cumsum((tp * pred_weights)[keep])
            fpc = np.cumsum(((1 - tp) * pred_weights)[keep])
            if count and len(tpc):
                ap = float(compute_ap(tpc / count, tpc / np.maximum(tpc + fpc, 1e-16))[0])
            else:
                ap = 0.0
            selected = confidence >= .25
            correct = float((tp * pred_weights)[selected].sum())
            predicted = int(pred_weights[selected].sum())
            precision = correct / predicted if predicted else 0.0
            recall = correct / count if count else 0.0
            f1 = 2 * correct / (predicted + count) if predicted + count else 0.0
            rows.append(dict(label=base.RESNET_CLASSES[label], support=count, ap50=ap,
                             precision=precision, recall=recall, f1=f1,
                             true_positives=int(correct), predicted_boxes=predicted))
        return {"map50": sum(r["ap50"] for r in rows) / 6,
                "macro_f1": sum(r["f1"] for r in rows) / 6, "per_class": rows}

    results = {system: scores(system, np.ones(len(images), dtype=int)) for system in SYSTEMS}
    samples = {system: {metric: [] for metric in ("map50", "macro_f1")} for system in SYSTEMS}
    rng = np.random.default_rng(42)
    for _ in range(repetitions):
        weights = rng.multinomial(len(images), np.full(len(images), 1 / len(images)))
        for system in SYSTEMS:
            metrics = scores(system, weights)
            for key in samples[system]:
                samples[system][key].append(metrics[key])
    differences = {}
    for system in SYSTEMS:
        results[system]["intervals_95"] = {
            metric: np.quantile(values, [.025, .975]).tolist() for metric, values in samples[system].items()}
        if system != "yolo":
            differences[system + " minus yolo"] = {
                metric: {"difference": results[system][metric] - results['yolo'][metric],
                         "interval_95": np.quantile(np.asarray(values) - samples['yolo'][metric], [.025, .975]).tolist()}
                for metric, values in samples[system].items()}
    return {"systems": results, "paired_differences": differences,
            "protocol": {"classes": list(base.RESNET_CLASSES), "iou_match": .5, "f1_confidence": .25,
                         "predict_confidence": .01, "image_size": 640, "max_detections": 100,
                         "agnostic_nms": True, "nms_iou": .7,
                         "hybrid_score": "YOLO object confidence multiplied by ResNet maximum softmax probability",
                         "ap_implementation": "Ultralytics compute_ap; confidence-ranked same-class greedy matching",
                         "bootstrap": "paired source-image percentile bootstrap", "repetitions": repetitions,
                         "seed": 42, "confidence_level": .95, "test_images": len(images)}}


def evaluate(root, yolo_path, resnet_path, adapted_path, output):
    from PIL import Image
    import torch
    from ultralytics import YOLO

    root, output = Path(root), Path(output)
    splits = base.read_splits(root, 'taco')
    images = []
    with tempfile.TemporaryDirectory(prefix='taco_detection_') as directory:
        local = Path(directory)
        for source, name in [(yolo_path, 'yolo.pt'), (resnet_path, 'c4.pt'), (adapted_path, 'adapted.pt')]:
            shutil.copyfile(source, local / name)
        yolo = YOLO(str(local / 'yolo.pt'))
        classifiers = [base.ResNetClassifier(local / name, 'cuda:0') for name in ['c4.pt', 'adapted.pt']]
        for number, relative in enumerate(splits['test'], 1):
            shutil.copyfile(root / relative, local / relative.name)
            with Image.open(local / relative.name) as opened:
                image = opened.convert('RGB')
            width, height = image.size
            targets = base.read_labels(root / 'labels' / relative.relative_to('images').with_suffix('.txt'))
            prediction = yolo.predict(image, imgsz=640, conf=.01, max_det=100,
                                      agnostic_nms=True, iou=.7, device=0, verbose=False)[0]
            boxes = prediction.boxes.xyxy.cpu().tolist()
            labels = prediction.boxes.cls.cpu().int().tolist()
            confidences = prediction.boxes.conf.cpu().tolist()
            normalized = [[x1 / width, y1 / height, x2 / width, y2 / height] for x1,y1,x2,y2 in boxes]
            crops = [image.crop((max(0, math.floor(x1)), max(0, math.floor(y1)),
                                min(width, max(math.floor(x1)+1, math.ceil(x2))),
                                min(height, max(math.floor(y1)+1, math.ceil(y2))))) for x1,y1,x2,y2 in boxes]
            row = {"image": str(relative), "targets": targets,
                   "yolo": [dict(box=box, class_id=label, confidence=confidence)
                            for box, label, confidence in zip(normalized, labels, confidences)]}
            for system, classifier in zip(SYSTEMS[1:], classifiers):
                predictions = classifier.predict(crops, batch_size=32)
                row[system] = [dict(box=box, class_id=label, confidence=detector_score * probability)
                               for box, detector_score, (label, probability) in zip(normalized, confidences, predictions)]
            images.append(row)
            if number % 25 == 0 or number == len(splits['test']):
                base.write_json(output / 'detection_status.json', dict(state='evaluating', images_complete=number,
                                                                     test_images=len(splits['test'])))
                print(f'End-to-end detection: {number}/{len(splits["test"])}', flush=True)
        del yolo, classifiers
        torch.cuda.empty_cache()
    base.write_json(output / 'detection_predictions.json', images)
    metrics = summarize(images)
    metrics['checkpoint_sha256'] = {key: base.sha256(path) for key, path in
                                    [('yolo', yolo_path), ('c4', resnet_path), ('adapted', adapted_path)]}
    metrics['split_manifest_sha256'] = base.sha256(root / 'split_manifest.json')
    base.write_json(output / 'detection_metrics.json', metrics)
    base.write_json(output / 'detection_status.json', dict(state='success', test_images=len(images)))
    return metrics
