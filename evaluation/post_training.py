"""Adapt C4 ResNet on fixed YOLO-predicted crops with validation early stopping.

The official training split supplies optimization examples. Validation selects
the checkpoint. Test image and label files are never opened. Outputs are best.pt,
last.pt, history.csv, metrics.json, and one progress status.json in the run folder.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import shutil
import tempfile
from time import perf_counter

if __package__:
    from . import sortwaste as base
else:
    import sortwaste as base


RUN_NAME = "sortwaste_yolo_resnet_post_training"
DEFAULT_OUTPUT = "/Volumes/workspace/default/models/" + RUN_NAME


@dataclass
class EarlyStopping:
    patience: int = 0
    best: float = -float("inf")
    stale_epochs: int = 0

    def __post_init__(self):
        if self.patience < 0:
            raise ValueError("patience must be nonnegative (0 disables early stopping)")

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError("Validation macro-F1 must be finite")
        if score > self.best:
            self.best, self.stale_epochs = score, 0
        else:
            self.stale_epochs += 1
        return self.patience > 0 and self.stale_epochs >= self.patience


def verify_reference(root, args, splits):
    """Fail before training if TACO membership differs from the original run."""
    expected = getattr(args, "reference_manifest_sha256", None)
    if expected:
        if args.dataset != "taco" or base.sha256(root / "split_manifest.json") != expected:
            raise ValueError("TACO split manifest differs from the original standalone YOLO split")
        counts = {split: len(entries) for split, entries in splits.items()}
        if counts != {"train": 1126, "val": 197, "test": 177}:
            raise ValueError(f"Unexpected original TACO split sizes: {counts}")
    yolo_hash = getattr(args, "reference_yolo_sha256", None)
    if yolo_hash and base.sha256(args.yolo) != yolo_hash:
        raise ValueError("YOLO checkpoint differs from the original TACO standalone model")
    return {"reference_manifest_sha256": expected,
            "reference_yolo_sha256": yolo_hash,
            "original_taco_split_verified": bool(expected and yolo_hash)}


def crop_targets(predicted_boxes, targets, eligible, source, threshold=.5):
    """Labels come from ground truth, including when YOLO's own class is wrong."""
    matches = base.match_boxes(predicted_boxes, [target["box"] for target in targets], threshold)
    rows = []
    for prediction, target_index, _ in matches:
        key = (source, target_index)
        if key not in eligible:
            continue
        label = targets[target_index]["class_id"]
        if label >= len(base.RESNET_CLASSES) or eligible[key] != base.CLASSES[label]:
            raise ValueError(f"Classifier/detector label mismatch for {key}")
        rows.append((prediction, label))
    return rows


def trainable_parameters(model):
    """Adapt layer4 and fc; retain the initial low-level feature extractor."""
    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc.requires_grad_(True)
    return [{"params": list(model.fc.parameters()), "lr": 1e-4},
            {"params": list(model.layer4.parameters()), "lr": 1e-5}]


def training_mode(model):
    import torch

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


class CropDataset:
    """Two temporary memory-mapped caches avoid thousands of crop image files."""

    def __init__(self, cache, transform):
        import numpy as np

        self.labels = cache["labels"]
        self.pixels = np.memmap(cache["path"], dtype=np.uint8, mode="r",
                                shape=(len(self.labels), 224, 224, 3))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        image = Image.fromarray(np.array(self.pixels[index], copy=True))
        return self.transform(image), self.labels[index]


def build_cache(split, entries, eligible, root, local, yolo, device, output):
    import numpy as np
    from PIL import Image

    if split not in {"train", "val"}:
        raise ValueError("Only training and validation images may enter post-training")
    split_dir = local / split
    split_dir.mkdir()

    def stage(relative):
        if relative.parent not in (Path("images/sortwaste") / split, Path("images") / split):
            raise ValueError(f"Unexpected {split} source: {relative}")
        destination = split_dir / relative.name
        shutil.copyfile(root / relative, destination)
        objects = base.read_labels(root / "labels" / relative.relative_to("images").with_suffix(".txt"))
        return relative, destination, objects

    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(stage, entries))
    cache_path = local / f"{split}_crops.bin"
    labels, yolo_labels, eligible_seen = [], [], set()
    with cache_path.open("wb") as stream:
        for number, (relative, image_path, targets) in enumerate(images, 1):
            for index, target in enumerate(targets):
                key = (relative.stem, index)
                if key in eligible:
                    if eligible[key] != base.CLASSES[target["class_id"]]:
                        raise ValueError(f"Classifier/detector label mismatch: {key}")
                    eligible_seen.add(key)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            result = yolo.predict(source=image, imgsz=640, conf=.35, max_det=300,
                                  device=device, verbose=False, save=False)[0]
            boxes = result.boxes.xyxy.cpu().tolist()
            identifiers = result.boxes.cls.cpu().int().tolist()
            normalized = [[x1 / width, y1 / height, x2 / width, y2 / height] for x1, y1, x2, y2 in boxes]
            for prediction, label in crop_targets(normalized, targets, eligible, relative.stem):
                x1, y1, x2, y2 = boxes[prediction]
                bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                          min(width, max(math.floor(x1) + 1, math.ceil(x2))),
                          min(height, max(math.floor(y1) + 1, math.ceil(y2))))
                crop = image.crop(bounds).resize((224, 224), Image.Resampling.BILINEAR)
                stream.write(np.asarray(crop, dtype=np.uint8).tobytes())
                labels.append(label)
                yolo_labels.append(identifiers[prediction])
            if number % 100 == 0 or number == len(images):
                state = dict(state="preparing_crops", split=split, images_complete=number,
                             split_images=len(images), matched_crops=len(labels))
                base.write_json(output / "status.json", state)
                print(state, flush=True)
    if eligible_seen != set(eligible):
        raise ValueError(f"Missing {split} ground-truth crop IDs: {len(set(eligible) - eligible_seen)}")
    if not labels:
        raise ValueError(f"No matched {split} crops")
    return {"path": cache_path, "labels": labels, "yolo_labels": yolo_labels,
            "images": len(images), "eligible_objects": len(eligible),
            "matched_counts": dict(Counter(base.CLASSES[label] for label in labels))}


def validate(model, loader, device):
    import torch

    model.eval()
    truth, predictions, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            truth.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = base.classification_metrics(
        [(base.CLASSES[truth_id], base.CLASSES[prediction]) for truth_id, prediction in zip(truth, predictions)],
        base.RESNET_CLASSES)
    metrics["loss"] = loss_sum / len(truth)
    return metrics


def train(args):
    import numpy as np
    import torch
    from torchvision import transforms
    from ultralytics import YOLO, settings

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output / "status.json", {"state": "preparing"})
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    stopper = EarlyStopping(getattr(args, "patience", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("This full post-training run requires its A10 GPU")
    device = "cuda:0"
    torch.manual_seed(42)
    torch.set_num_threads(4)
    np.random.seed(42)
    random.seed(42)
    settings.update({"mlflow": False, "wandb": False, "tensorboard": False})
    root = Path(args.data_root)
    splits = base.read_splits(root, args.dataset)
    reference = verify_reference(root, args, splits)
    eligibility = {split: (base.taco_eligibility(root, splits[split]) if args.dataset == "taco"
                           else base.classifier_eligibility(args.classifier_index, splits, partition=split))
                   for split in ("train", "val")}
    source_yolo_hash = base.sha256(args.yolo)
    source_resnet_hash = base.sha256(args.resnet)
    history, best, best_metrics, best_epoch = [], (-float("inf"), -float("inf")), None, None
    with tempfile.TemporaryDirectory(prefix=RUN_NAME + "_") as temporary:
        local = Path(temporary)
        shutil.copyfile(args.yolo, local / "yolo.pt")
        shutil.copyfile(args.resnet, local / "resnet.pt")
        yolo = YOLO(str(local / "yolo.pt"))
        if yolo.task != "detect":
            raise ValueError("Expected a YOLO object detector")
        detector_names = base.detector_classes(yolo.names)
        yolo.model.requires_grad_(False).eval()
        classifier = base.ResNetClassifier(local / "resnet.pt", device)
        if classifier.config["experiment"] != "C4" or classifier.epoch != 14:
            raise ValueError("Expected the selected C4 ResNet checkpoint from epoch 14")
        index_hash = classifier.config["index_sha256"] if args.dataset == "taco" else base.sha256(args.classifier_index)
        if args.dataset == "sortwaste" and classifier.config["index_sha256"] != index_hash:
            raise ValueError("Classifier export changed since C4 training")
        caches = {split: build_cache(split, splits[split], eligibility[split], root, local,
                                     yolo, device, output) for split in ("train", "val")}
        if args.dataset == "taco" and any(set(cache["matched_counts"]) != set(base.RESNET_CLASSES) for cache in caches.values()):
            raise ValueError("TACO train and validation crops must cover all six classes")
        settings = classifier.config["preprocessing"]
        if settings["image_size"] != [224, 224] or settings["color_mode"] != "RGB":
            raise ValueError("Expected the C4 checkpoint's 224x224 RGB preprocessing")
        transform = transforms.Compose([transforms.RandomHorizontalFlip(.5),
            transforms.ColorJitter(brightness=.1, contrast=.1), transforms.ToTensor(),
            transforms.Normalize(settings["mean"], settings["std"])])
        datasets = {"train": CropDataset(caches["train"], transform),
                    "val": CropDataset(caches["val"], classifier.transform)}
        loaders = {split: torch.utils.data.DataLoader(dataset, batch_size=64,
                    shuffle=split == "train", num_workers=0, pin_memory=True,
                    generator=torch.Generator().manual_seed(42)) for split, dataset in datasets.items()}
        model = classifier.model
        optimizer = torch.optim.AdamW(trainable_parameters(model), weight_decay=1e-4)
        initial = validate(model, loaders["val"], device)
        yolo_validation = base.classification_metrics(
            [(base.CLASSES[truth], base.CLASSES[prediction]) for truth, prediction in
             zip(caches["val"]["labels"], caches["val"]["yolo_labels"])], base.RESNET_CLASSES)
        config = {**{key: classifier.config[key] for key in ("architecture", "classes", "preprocessing")},
            "experiment": args.run_name, "initial_experiment": "C4", "seed": 42,
            "detector_classes": list(detector_names),
            "data_root": str(root), "classifier_index": None if args.dataset == "taco" else args.classifier_index,
            "source_resnet_epoch": classifier.epoch, "source_resnet": args.resnet,
            "source_resnet_sha256": source_resnet_hash, "source_yolo": args.yolo,
            "source_yolo_sha256": source_yolo_hash, "index_sha256": index_hash,
            "training_datasets": [args.dataset], "epochs": args.epochs, "batch_size": 64,
            "early_stopping_patience": stopper.patience, "early_stopping_metric": "validation macro-F1",
            "early_stopping_min_delta": 0.0, "reference_verification": reference,
            "lr": 1e-4, "backbone_lr": 1e-5, "optimizer": "AdamW", "weight_decay": 1e-4, "warmup_epochs": 0,
            "trainable_modules": ["layer4", "fc"], "batchnorm_running_stats": "frozen",
            "sampling": "natural class frequencies; each matched training crop once per epoch",
            "training_augmentation": {"horizontal_flip_probability": .5, "brightness": .1, "contrast": .1},
            "training_crop_source": "fixed YOLO predicted boxes matched to ground truth; labels are ground truth",
            "matching_iou": .5, "yolo_confidence": .35, "yolo_image_size": 640, "max_detections": 300,
            "carton_policy": "not a class in six-class TACO" if args.dataset == "taco" else "excluded consistently with the original six-class classifier",
            "checkpoint_selection": f"{args.dataset} validation macro-F1, accuracy tiebreaker",
            "test_images_used": 0, "test_labels_used": False, "yolo_frozen": True,
            "split_list_sha256": {split: base.split_hash(root, splits[split], args.dataset, split) for split in ("train", "val")},
            "crop_counts": {split: {key: value for key, value in cache.items() if key in
                            {"images", "eligible_objects", "matched_counts"}} for split, cache in caches.items()}}
        if args.dataset == "taco":
            config.update(split_manifest_sha256=base.sha256(root / "split_manifest.json"),
                          label_mapping_sha256=base.sha256(root / "label_mapping.json"))
        summary = {"run_name": args.run_name, "config": config, "initial_validation": initial,
                   "yolo_validation": yolo_validation, "epochs_completed": 0}
        base.write_json(output / "metrics.json", summary)
        print({"initial_validation_accuracy": initial["accuracy"], "initial_validation_macro_f1": initial["macro_f1"]}, flush=True)
        for epoch in range(1, args.epochs + 1):
            started = perf_counter()
            training_mode(model)
            loss_sum, correct, seen = 0.0, 0, 0
            for inputs, targets in loaders["train"]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                count = targets.shape[0]
                loss_sum += loss.item() * count
                correct += (logits.argmax(dim=1) == targets).sum().item()
                seen += count
            metrics = validate(model, loaders["val"], device)
            if seen != len(datasets["train"]):
                raise RuntimeError("An epoch did not consume every training crop exactly once")
            history.append(dict(epoch=epoch, train_examples=seen, train_loss=loss_sum / seen, train_accuracy=correct / seen,
                val_loss=metrics["loss"], val_accuracy=metrics["accuracy"], val_macro_f1=metrics["macro_f1"],
                seconds=perf_counter() - started))
            rank = (metrics["macro_f1"], metrics["accuracy"])
            improved = rank > best
            stopped_early = stopper.update(metrics["macro_f1"])
            checkpoint = {"model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                          "config": config, "epoch": epoch, "val_metrics": metrics}
            torch.save(checkpoint, local / "checkpoint.pt")
            if improved:
                best, best_metrics, best_epoch = rank, metrics, epoch
                shutil.copyfile(local / "checkpoint.pt", output / "best.pt")
            shutil.copyfile(local / "checkpoint.pt", output / "last.pt")
            summary.update(epochs_completed=epoch, best_epoch=best_epoch, best_validation=best_metrics,
                           last_validation=metrics, best_checkpoint=str(output / "best.pt"),
                           epochs_requested=args.epochs, stopped_early=stopped_early,
                           stale_epochs=stopper.stale_epochs)
            base.write_json(output / "metrics.json", summary)
            with (output / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            base.write_json(output / "status.json", {"state": "training", "epochs_completed": epoch,
                            "epochs_requested": args.epochs, "best_epoch": best_epoch,
                            "patience": stopper.patience, "stale_epochs": stopper.stale_epochs,
                            "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"]})
            print(history[-1], flush=True)
            if stopped_early:
                print(f"Early stopping: no validation macro-F1 improvement for {stopper.patience} epochs", flush=True)
                break
        # Verify the selected checkpoint loads through the same inference adapter.
        loaded = base.ResNetClassifier(output / "best.pt", device)
        checked = validate(loaded.model, loaders["val"], device)
        if abs(checked["macro_f1"] - best_metrics["macro_f1"]) > 1e-10:
            raise RuntimeError("Saved checkpoint does not reproduce its selected validation score")
        if base.sha256(args.yolo) != source_yolo_hash or base.sha256(args.resnet) != source_resnet_hash:
            raise RuntimeError("A source checkpoint changed during the run")
        summary.update(saved_checkpoint_verified=True, source_checkpoints_preserved=True,
                       best_checkpoint_sha256=base.sha256(output / "best.pt"),
                       stop_reason="early_stopping" if stopped_early else "maximum_epochs")
        base.write_json(output / "metrics.json", summary)
        base.write_json(output / "status.json", {"state": "success", "epochs_completed": len(history),
                        "epochs_requested": args.epochs, "patience": stopper.patience,
                        "stop_reason": summary["stop_reason"],
                        "best_epoch": best_epoch, "best_checkpoint": str(output / "best.pt")})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sortwaste", "taco"], default="sortwaste")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", default=base.DEFAULT_DATA)
    parser.add_argument("--classifier-index", default=base.DEFAULT_INDEX)
    parser.add_argument("--yolo", default=base.DEFAULT_YOLO)
    parser.add_argument("--resnet", default=base.DEFAULT_RESNET)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--reference-manifest-sha256")
    parser.add_argument("--reference-yolo-sha256")
    parser.add_argument("--run-name", default=RUN_NAME)
    return train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
