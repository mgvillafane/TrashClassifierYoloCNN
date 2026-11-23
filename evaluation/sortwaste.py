"""Paired classification evaluation on existing SortWaste or six-class TACO test splits.

YOLO boxes are generated once. Both systems use exactly those boxes and the same
class-independent, one-to-one IoU matching. ResNet replaces every predicted label.
No model fitting, calibration, threshold search, or test-set subsampling is done.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from time import perf_counter


CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash", "carton")
RESNET_CLASSES = CLASSES[:6]
DEFAULT_YOLO = "/Volumes/workspace/default/tacotrashdata/reports/sortwaste_yolo/sortwaste_yolo26m_40epochs_20261121T213604Z_1bf985/weights/best.pt"
DEFAULT_RESNET = "/Volumes/workspace/default/models/classifier_experiments/C4/20261120T034908Z_9bfaa9fb/resnet-layer4/best.pt"
DEFAULT_DATA = "/Volumes/workspace/default/waste_data/processed/yolo_classes"
DEFAULT_INDEX = "/Volumes/workspace/default/waste_data/processed/classifier/index.csv"
TACO_YOLO = "/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_trashnet_yolo26m_40epochs_2184af/weights/best.pt"
TACO_DATA = "/Volumes/workspace/default/tacotrashdata/yolo/taco_trashnet"


def detector_classes(names):
    labels = tuple(names[key] for key in sorted(names)) if isinstance(names, dict) else tuple(names)
    if labels not in (RESNET_CLASSES, CLASSES):
        raise ValueError("Expected the six common YOLO classes, optionally followed by carton")
    return labels


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_splits(root, dataset="sortwaste"):
    """Check every split without opening any training or validation images."""
    root = Path(root)
    splits = {}
    manifest = None
    if dataset == "taco":
        manifest = json.loads((root / "split_manifest.json").read_text())
        if detector_classes({int(key): value for key, value in manifest["class_names"].items()}) != RESNET_CLASSES:
            raise ValueError("Expected TACO mapped to the six common material classes")
    for split in ("train", "val", "test"):
        if manifest:
            entries = [Path(row["image"]) for row in manifest["images"] if row["split"] == split]
            expected = Path("images") / split
            if len(entries) != manifest["image_counts"][split]:
                raise ValueError(f"TACO {split} membership disagrees with its manifest")
        else:
            entries = [Path(line.strip()) for line in (root / f"sortwaste_{split}.txt").read_text().splitlines() if line.strip()]
            expected = Path("images/sortwaste") / split
        if not entries or any(entry.is_absolute() or ".." in entry.parts or not entry.is_relative_to(expected) for entry in entries):
            raise ValueError(f"Invalid or empty {split} image list")
        stems = [entry.stem for entry in entries]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate source images in {split}")
        splits[split] = entries
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if {p.stem for p in splits[left]} & {p.stem for p in splits[right]}:
            raise ValueError(f"Source-image overlap between {left} and {right}")
        if manifest:
            sources = {split: {row["source_file"] for row in manifest["images"] if row["split"] == split}
                       for split in (left, right)}
            if sources[left] & sources[right]:
                raise ValueError(f"TACO source-image overlap between {left} and {right}")
    return splits


def taco_eligibility(root, entries):
    """Use every annotated TACO object; open labels only for the requested split."""
    eligible = {}
    for relative in entries:
        label_path = Path(root) / "labels" / relative.relative_to("images").with_suffix(".txt")
        for index, target in enumerate(read_labels(label_path)):
            if target["class_id"] >= len(RESNET_CLASSES):
                raise ValueError(f"Unexpected seventh class in six-class TACO: {label_path}")
            eligible[(relative.stem, index)] = RESNET_CLASSES[target["class_id"]]
    if not eligible:
        raise ValueError("No annotated TACO objects in requested split")
    return eligible


def split_hash(root, entries, dataset, partition):
    if dataset == "sortwaste":
        return sha256(Path(root) / f"sortwaste_{partition}.txt")
    return hashlib.sha256(("\n".join(str(path) for path in entries) + "\n").encode()).hexdigest()


def classifier_eligibility(index, splits, partition="test"):
    """Use original exported crop IDs, preserving its crop-size exclusions."""
    allowed = {split: {path.stem for path in paths} for split, paths in splits.items()}
    eligible = {}
    with Path(index).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "sortwaste":
                continue
            split = row["partition"]
            source = Path(row["source_image_id"]).stem
            if split not in allowed or source not in allowed[split]:
                raise ValueError(f"Classifier/detector source split mismatch: {source}/{split}")
            if split != partition:
                continue
            crop_source, suffix = Path(row["path"]).stem.rsplit("_", 1)
            if crop_source != source:
                raise ValueError(f"Unexpected crop/source ID: {row['path']}")
            key = (source, int(suffix))
            if key in eligible or row["label"] not in RESNET_CLASSES:
                raise ValueError(f"Invalid classifier {partition} crop: {key}")
            eligible[key] = row["label"]
    if not eligible:
        raise ValueError(f"No eligible SortWaste classifier {partition} crops")
    return eligible


def read_labels(path):
    objects = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid label in {path}")
        label, x, y, width, height = values
        if label != int(label) or not 0 <= label < len(CLASSES):
            raise ValueError(f"Invalid class in {path}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid box in {path}")
        objects.append({"class_id": int(label), "box": [max(0, x - width / 2), max(0, y - height / 2),
                                                        min(1, x + width / 2), min(1, y + height / 2)]})
    return objects


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions, targets, threshold=0.5):
    """Greedy descending-IoU assignment, independent of either system's labels."""
    candidates = []
    for prediction, box in enumerate(predictions):
        for target, truth in enumerate(targets):
            overlap = iou(box, truth)
            if overlap >= threshold:
                candidates.append((-overlap, prediction, target))
    used_predictions, used_targets, matches = set(), set(), []
    for negative_overlap, prediction, target in sorted(candidates):
        if prediction not in used_predictions and target not in used_targets:
            used_predictions.add(prediction)
            used_targets.add(target)
            matches.append((prediction, target, -negative_overlap))
    return matches


def classification_metrics(pairs, scope):
    """Retain predictions outside the target scope as errors; average supported labels."""
    pairs = [(truth, prediction) for truth, prediction in pairs if truth in scope]
    per_class = []
    for label in CLASSES:
        tp = sum(truth == label and prediction == label for truth, prediction in pairs)
        fp = sum(truth != label and prediction == label for truth, prediction in pairs)
        fn = sum(truth == label and prediction != label for truth, prediction in pairs)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append(dict(label=label, precision=precision, recall=recall, f1=f1, support=support))
    supported = [row for row in per_class if row["support"]]
    count = len(pairs)
    return {
        "count": count, "correct": sum(truth == prediction for truth, prediction in pairs),
        "accuracy": sum(truth == prediction for truth, prediction in pairs) / count if count else None,
        "macro_f1": sum(row["f1"] for row in supported) / len(supported) if supported else None,
        "weighted_f1": sum(row["f1"] * row["support"] for row in supported) / count if count else None,
        "macro_f1_classes": [row["label"] for row in supported], "per_class": per_class,
    }


def summarize(records):
    """Paired rows include matched targets, missed targets and unmatched detections."""
    summary = {"systems": {}}
    target_rows = [row for row in records if row["gt_class"] is not None]
    matched = [row for row in target_rows if row["prediction_index"] is not None]
    primary = [row for row in matched if row["classifier_eligible"]]
    predictions = [row for row in records if row["prediction_index"] is not None]
    cartons = [row for row in target_rows if row["gt_class"] == "carton"]
    summary["counts"] = dict(ground_truth_objects=len(target_rows), predicted_boxes=len(predictions),
        matched_objects=len(matched), missed_objects=len(target_rows) - len(matched),
        unmatched_predictions=len(predictions) - len(matched), classifier_eligible_matched=len(primary),
        classifier_eligible_ground_truth=sum(row["classifier_eligible"] for row in target_rows),
        carton_ground_truth=len(cartons), carton_matched=sum(row["prediction_index"] is not None for row in cartons))
    summary["localization_recall"] = len(matched) / len(target_rows) if target_rows else None
    systems = ["yolo", "yolo_resnet"]
    if any("yolo_post_resnet_class" in row for row in records):
        systems.append("yolo_post_resnet")
    for system in systems:
        key = system + "_class"
        pairs = [(row["gt_class"], row[key]) for row in matched]
        eligible_pairs = [(row["gt_class"], row[key]) for row in primary]
        correct = sum(row["gt_class"] == row[key] for row in matched)
        precision = correct / len(predictions) if predictions else 0.0
        recall = correct / len(target_rows) if target_rows else 0.0
        summary["systems"][system] = {
            "primary_classifier_eligible": classification_metrics(eligible_pairs, RESNET_CLASSES),
            "all_shared_class_matched": classification_metrics(pairs, RESNET_CLASSES),
            "all_seven_class_matched": classification_metrics(pairs, CLASSES),
            "carton_matched": classification_metrics(pairs, ("carton",)),
            "end_to_end_all_seven": {"correctly_localized_and_classified": correct,
                "precision": precision, "recall": recall,
                "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0},
        }
    outcomes = Counter()
    for row in primary:
        before = row["yolo_class"] == row["gt_class"]
        after = row["yolo_resnet_class"] == row["gt_class"]
        outcomes["both_correct" if before and after else "resnet_corrected_yolo" if after else
                 "resnet_broke_correct_yolo" if before else "both_wrong"] += 1
    summary["paired_primary_outcomes"] = dict(outcomes)
    if "yolo_post_resnet" in systems:
        outcomes = Counter()
        for row in primary:
            before = row["yolo_class"] == row["gt_class"]
            after = row["yolo_post_resnet_class"] == row["gt_class"]
            outcomes["both_correct" if before and after else "resnet_corrected_yolo" if after else
                     "resnet_broke_correct_yolo" if before else "both_wrong"] += 1
        summary["post_training_primary_outcomes"] = dict(outcomes)
    return summary


def paired_bootstrap(records, repetitions=2000, seed=42, system="yolo_resnet"):
    """Resample source images, preserving within-image correlation and paired predictions."""
    import numpy as np

    images = sorted({row["image"] for row in records})
    positions = {name: index for index, name in enumerate(images)}
    matrices = np.zeros((len(images), 2, len(CLASSES), len(CLASSES)), dtype=np.int64)
    for row in records:
        if not row["classifier_eligible"] or row["prediction_index"] is None:
            continue
        truth = CLASSES.index(row["gt_class"])
        for number, key in enumerate(("yolo_class", system + "_class")):
            matrices[positions[row["image"]], number, truth, CLASSES.index(row[key])] += 1
    weights = np.random.default_rng(seed).multinomial(len(images), [1 / len(images)] * len(images), size=repetitions)
    samples = (weights @ matrices.reshape(len(images), -1)).reshape(repetitions, 2, len(CLASSES), len(CLASSES))
    diagonal = np.diagonal(samples, axis1=2, axis2=3)
    support = samples.sum(axis=3)
    predicted = samples.sum(axis=2)
    supported = matrices.sum(axis=(0, 1, 3)) > 0
    denominators = support + predicted
    f1 = np.divide(2 * diagonal, denominators, out=np.zeros_like(diagonal, dtype=float), where=denominators > 0)
    total = samples.sum(axis=(2, 3))
    if not total.any():
        return {"method": "paired source-image bootstrap", "repetitions": repetitions, "available": False}
    accuracy = np.divide(diagonal.sum(axis=2), total, out=np.zeros_like(total, dtype=float), where=total > 0)
    differences = {"accuracy_delta": accuracy[:, 1] - accuracy[:, 0],
                   "macro_f1_delta": f1[:, 1, supported].mean(axis=1) - f1[:, 0, supported].mean(axis=1)}
    return {"method": "paired source-image bootstrap", "repetitions": repetitions, "seed": seed,
            "delta_direction": system + " minus YOLO", "confidence_level": 0.95,
            **{key + "_95ci": np.quantile(value, [0.025, 0.975]).tolist() for key, value in differences.items()}}


class ResNetClassifier:
    """Load a ResNet-layer4 checkpoint using its saved preprocessing and label order."""

    def __init__(self, path, device):
        import torch
        from torch import nn
        from torchvision.models import resnet18
        from torchvision import transforms

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config["architecture"] != "resnet-layer4" or tuple(config["classes"]) != RESNET_CLASSES:
            raise ValueError("Expected a six-class resnet-layer4 checkpoint")
        self.config = config
        self.epoch = checkpoint["epoch"]
        settings = config["preprocessing"]
        self.color_mode = settings["color_mode"]
        self.transform = transforms.Compose([
            transforms.Resize(tuple(settings["image_size"]), interpolation=transforms.InterpolationMode(settings["interpolation"])),
            transforms.ToTensor(), transforms.Normalize(settings["mean"], settings["std"]),
        ])
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(RESNET_CLASSES))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = device
        self.model.to(device).eval()

    def predict(self, crops, batch_size=128):
        import torch

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                inputs = torch.stack([self.transform(crop.convert(self.color_mode)) for crop in crops[start:start + batch_size]])
                scores, identifiers = self.model(inputs.to(self.device)).softmax(dim=1).max(dim=1)
                predictions.extend((int(identifier), float(score)) for identifier, score in zip(identifiers.cpu(), scores.cpu()))
        return predictions


def save_outputs(output, records, summary):
    with (output / "objects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "metrics.json", summary)


def run_evaluation(args):
    import numpy as np
    from PIL import Image
    import torch
    import yaml
    from ultralytics import YOLO, settings

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "preparing"})
    root = Path(args.data_root)
    splits = read_splits(root, args.dataset)
    eligible = (taco_eligibility(root, splits["test"]) if args.dataset == "taco"
                else classifier_eligibility(args.classifier_index, splits))
    names = yaml.safe_load((root / ("dataset.yaml" if args.dataset == "taco" else "e1_sortwaste.yaml")).read_text())["names"]
    actual = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    if tuple(actual) != (RESNET_CLASSES if args.dataset == "taco" else CLASSES):
        raise ValueError(f"Unexpected detector class order: {actual}")
    settings.update({"mlflow": False, "wandb": False, "tensorboard": False})
    torch.manual_seed(42)
    torch.set_num_threads(4)
    np.random.seed(42)
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    config = {"dataset": args.dataset, "dataset_name": "TACO" if args.dataset == "taco" else "SortWaste",
        "split": "test", "official_split_preserved": True, "test_images": len(splits["test"]),
        "split_counts": {key: len(value) for key, value in splits.items()}, "data_root": str(root),
        "image_size": args.image_size, "confidence": args.confidence, "max_detections": args.max_detections,
        "matching_iou": args.matching_iou, "matching": "class-independent greedy descending IoU, one-to-one",
        "primary_population": ("all annotated six-class TACO test objects matched to a predicted box; no crop-size exclusions"
                               if args.dataset == "taco" else "matched objects present in original classifier test export (no cartons; min crop side 32 px)"),
        "secondary_population": "all matched shared-class objects, including small crops",
        "crop_context_padding": 0, "hybrid_policy": "replace every YOLO class with ResNet argmax; retain identical boxes",
        "carton_policy": "excluded from primary/shared-class scores; retained in explicit seven-class and carton diagnostics",
        "no_test_set_tuning": True, "seed": 42, "device": device,
        "yolo_checkpoint": args.yolo, "yolo_sha256": sha256(args.yolo),
        "detector_name": args.detector_name,
        "resnet_checkpoint": args.resnet, "resnet_sha256": sha256(args.resnet),
        "classifier_index_sha256": None if args.dataset == "taco" else sha256(args.classifier_index),
        "test_list_sha256": split_hash(root, splits["test"], args.dataset, "test")}
    if args.dataset == "taco":
        config.update(split_manifest_sha256=sha256(root / "split_manifest.json"),
                      label_mapping_sha256=sha256(root / "label_mapping.json"), carton_policy="not a class in TACO's six-class mapping")
    records, timings, eligible_seen = [], [], set()
    with tempfile.TemporaryDirectory(prefix="sortwaste_paired_eval_") as temporary:
        local = Path(temporary)
        shutil.copyfile(args.yolo, local / "yolo.pt")
        shutil.copyfile(args.resnet, local / "resnet.pt")
        yolo = YOLO(str(local / "yolo.pt"))
        if yolo.task != "detect":
            raise ValueError("Expected a YOLO object detector")
        config["detector_classes"] = list(detector_classes(yolo.names))
        resnet = ResNetClassifier(local / "resnet.pt", device)
        if args.dataset == "sortwaste" and resnet.config["index_sha256"] != config["classifier_index_sha256"]:
            raise ValueError("Classifier export changed since C4 training")
        config.update(resnet_epoch=resnet.epoch, resnet_experiment=resnet.config["experiment"],
                      resnet_preprocessing=resnet.config["preprocessing"])
        post_resnet = None
        if args.post_resnet:
            shutil.copyfile(args.post_resnet, local / "post_resnet.pt")
            post_resnet = ResNetClassifier(local / "post_resnet.pt", device)
            if (post_resnet.config.get("source_yolo_sha256") != config["yolo_sha256"] or
                    post_resnet.config.get("source_resnet_sha256") != config["resnet_sha256"]):
                raise ValueError("Post-training must use this exact detector and original C4 checkpoint")
            if args.dataset == "taco" and (post_resnet.config.get("training_datasets") != ["taco"] or
                    post_resnet.config.get("split_manifest_sha256") != config["split_manifest_sha256"]):
                raise ValueError("TACO comparison requires the new C4 checkpoint trained on this TACO training split")
            config.update(post_resnet_checkpoint=args.post_resnet, post_resnet_sha256=sha256(args.post_resnet),
                          post_resnet_epoch=post_resnet.epoch, post_resnet_experiment=post_resnet.config["experiment"])

        def stage(relative):
            target = local / relative.name
            shutil.copyfile(root / relative, target)
            label = root / "labels" / relative.relative_to("images").with_suffix(".txt")
            objects = read_labels(label)
            return relative, target, objects

        # Stage the complete test split once; subsequent inference reads local files.
        with ThreadPoolExecutor(max_workers=8) as pool:
            staged = list(pool.map(stage, splits["test"]))
        write_json(output / "status.json", {"state": "evaluating", "test_images": len(staged)})
        for number, (relative, image_path, targets) in enumerate(staged, 1):
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = perf_counter()
            result = yolo.predict(source=image, imgsz=args.image_size, conf=args.confidence,
                                  max_det=args.max_detections, device=device, verbose=False, save=False)[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            yolo_ms = (perf_counter() - start) * 1000
            boxes = result.boxes.xyxy.cpu().tolist()
            identifiers = result.boxes.cls.cpu().int().tolist()
            scores = result.boxes.conf.cpu().tolist()
            normalized = [[x1 / width, y1 / height, x2 / width, y2 / height] for x1, y1, x2, y2 in boxes]
            start = perf_counter()
            crops = [image.crop((max(0, math.floor(x1)), max(0, math.floor(y1)),
                                 min(width, max(math.floor(x1) + 1, math.ceil(x2))),
                                 min(height, max(math.floor(y1) + 1, math.ceil(y2))))) for x1, y1, x2, y2 in boxes]
            crop_ms = (perf_counter() - start) * 1000
            alternate = resnet.predict(crops)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            resnet_ms = (perf_counter() - start) * 1000
            timings.append({"image": relative.name, "yolo_ms": yolo_ms, "resnet_crop_and_inference_ms": resnet_ms,
                            "hybrid_ms": yolo_ms + resnet_ms, "detections": len(boxes)})
            post_predictions = None
            if post_resnet:
                start = perf_counter()
                post_predictions = post_resnet.predict(crops)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                post_ms = (perf_counter() - start) * 1000 + crop_ms
                timings[-1].update(post_resnet_crop_and_inference_ms=post_ms, post_hybrid_ms=yolo_ms + post_ms)
            matches = match_boxes(normalized, [item["box"] for item in targets], args.matching_iou)
            by_target = {target: (prediction, overlap) for prediction, target, overlap in matches}
            matched_predictions = {prediction for prediction, _, _ in matches}
            for target_index, target in enumerate(targets):
                key = (relative.stem, target_index)
                is_eligible = key in eligible
                if is_eligible:
                    if eligible[key] != CLASSES[target["class_id"]]:
                        raise ValueError(f"Classifier/detector ground-truth class mismatch for {key}")
                    eligible_seen.add(key)
                prediction, overlap = by_target.get(target_index, (None, None))
                records.append(dict(image=relative.name, target_index=target_index, prediction_index=prediction,
                    gt_class=CLASSES[target["class_id"]], classifier_eligible=is_eligible, iou=overlap,
                    yolo_class=CLASSES[identifiers[prediction]] if prediction is not None else None,
                    yolo_resnet_class=RESNET_CLASSES[alternate[prediction][0]] if prediction is not None else None,
                    yolo_confidence=scores[prediction] if prediction is not None else None,
                    resnet_confidence=alternate[prediction][1] if prediction is not None else None))
                if post_predictions is not None:
                    records[-1].update(yolo_post_resnet_class=RESNET_CLASSES[post_predictions[prediction][0]] if prediction is not None else None,
                                       post_resnet_confidence=post_predictions[prediction][1] if prediction is not None else None)
            for prediction in range(len(boxes)):
                if prediction not in matched_predictions:
                    records.append(dict(image=relative.name, target_index=None, prediction_index=prediction,
                        gt_class=None, classifier_eligible=False, iou=None, yolo_class=CLASSES[identifiers[prediction]],
                        yolo_resnet_class=RESNET_CLASSES[alternate[prediction][0]], yolo_confidence=scores[prediction],
                        resnet_confidence=alternate[prediction][1]))
                    if post_predictions is not None:
                        records[-1].update(yolo_post_resnet_class=RESNET_CLASSES[post_predictions[prediction][0]],
                                           post_resnet_confidence=post_predictions[prediction][1])
            if number % 25 == 0 or number == len(staged):
                write_json(output / "status.json", {"state": "evaluating", "images_complete": number, "test_images": len(staged)})
                print(f"Evaluated {number}/{len(staged)} images", flush=True)
        if eligible_seen != set(eligible):
            raise ValueError(f"Classifier test crops missing from detector annotations: {len(set(eligible) - eligible_seen)}")
    summary = summarize(records)
    if args.dataset == "taco":
        if summary["systems"]["yolo"]["primary_classifier_eligible"]["macro_f1_classes"] != list(RESNET_CLASSES):
            raise ValueError("The TACO test comparison must contain matched targets from all six classes")
    summary.update(config=config, test_images=len(splits["test"]),
                   paired_uncertainty=paired_bootstrap(records))
    if args.post_resnet:
        summary["post_training_paired_uncertainty"] = paired_bootstrap(records, system="yolo_post_resnet")
    if args.training_metrics:
        training_path = Path(args.training_metrics)
        summary["training"] = json.loads(training_path.read_text())
        with training_path.with_name("history.csv").open(newline="") as stream:
            summary["training_history"] = list(csv.DictReader(stream))
        if (summary["training"]["epochs_completed"] != 10 or
                summary["training"]["best_checkpoint_sha256"] != config.get("post_resnet_sha256")):
            raise ValueError("Training metadata does not identify the evaluated ten-epoch checkpoint")
    steady = timings[1:]  # First image includes model warm-up in this one-pass run.
    summary["timing"] = {"scope": "single-image GPU inference, includes crop preparation; excludes I/O; first image excluded",
        **{key: {"median_ms": float(np.median([row[key] for row in steady])),
                 "p95_ms": float(np.percentile([row[key] for row in steady], 95))}
           for key in ("yolo_ms", "resnet_crop_and_inference_ms", "hybrid_ms",
                       *(("post_resnet_crop_and_inference_ms", "post_hybrid_ms") if args.post_resnet else ()))}}
    save_outputs(output, records, summary)
    write_json(output / "status.json", {"state": "success", "test_images": len(splits["test"]),
                                       "metrics": str(output / "metrics.json")})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sortwaste", "taco"], default="sortwaste")
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--classifier-index", default=DEFAULT_INDEX)
    parser.add_argument("--yolo", default=DEFAULT_YOLO)
    parser.add_argument("--resnet", default=DEFAULT_RESNET)
    parser.add_argument("--post-resnet")
    parser.add_argument("--training-metrics")
    parser.add_argument("--detector-name", default="SortWaste YOLO")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--matching-iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1 or not 0 < args.matching_iou <= 1:
        parser.error("Invalid confidence or matching IoU")
    return run_evaluation(args)


if __name__ == "__main__":
    main()
