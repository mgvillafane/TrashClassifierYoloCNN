"""Load the label config and map every source label onto the shared 6-class schema."""

from __future__ import annotations

import json
from pathlib import Path

DATASETS = ("realwaste", "trashnet", "sortwaste", "taco")


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    labels = config["common_labels"]
    if len(set(labels)) != len(labels):
        raise ValueError("common_labels contains duplicates")
    if not set(config["shared3_labels"]) <= set(labels):
        raise ValueError("shared3_labels must be a subset of common_labels")
    for dataset in ("realwaste", "trashnet", "sortwaste"):
        for source, rule in config[dataset].items():
            if rule["common"] not in labels:
                raise ValueError(f"{dataset}/{source}: unknown common label {rule['common']!r}")
            if rule.get("classifier", True) is False and not rule.get("reason"):
                raise ValueError(f"{dataset}/{source}: excluded labels need a reason")
    if sorted(config["sortwaste"], key=int) != [str(i) for i in range(len(config["sortwaste"]))]:
        raise ValueError("sortwaste keys must be the contiguous YOLO class ids 0..n-1")


def taco_rules(config: dict) -> dict[str, dict]:
    """TACO category -> common label, with classifier exclusions applied on top."""
    groups = config["taco"]["groups"]
    excluded = config["taco"]["classifier_exclude"]
    rules: dict[str, dict] = {}
    for common, names in groups.items():
        if common not in config["common_labels"]:
            raise ValueError(f"TACO group {common!r} is not a common label")
        for name in names:
            if name in rules:
                raise ValueError(f"TACO category mapped twice: {name}")
            rules[name] = {"common": common, "classifier": name not in excluded, "reason": excluded.get(name)}
    unknown = set(excluded) - set(rules)
    if unknown:
        raise ValueError(f"classifier_exclude names unknown TACO categories: {sorted(unknown)}")
    return rules


def build_rules(config: dict) -> dict[str, dict[str, dict]]:
    """{dataset: {original_label: {"common", "classifier", "reason"}}}"""
    return {
        "realwaste": config["realwaste"],
        "trashnet": config["trashnet"],
        "sortwaste": {rule["name"]: rule for rule in config["sortwaste"].values()},
        "taco": taco_rules(config),
    }


def apply_mapping(row: dict, rules: dict, config: dict) -> dict:
    """Add common label, detector/classifier usage flags and the exclusion reason to one manifest row."""
    if row["example_type"] == "no_object":
        return {**row, "use_detector_train": True, "use_detector_eval": True,
                "use_classifier_train": False, "use_classifier_eval": False,
                "exclude_reason": "no objects in image"}

    rule = rules[row["dataset"]].get(row["original_label"])
    if rule is None:
        # New or misspelled labels must fail loudly, never fall back to "trash".
        raise KeyError(f"unmapped label: {row['dataset']} / {row['original_label']}")

    has_box = row["example_type"] == "bounding_box"
    reason = row.get("exclude_reason") or (None if rule.get("classifier", True) else rule["reason"])
    min_side = config["classifier_min_crop_side_px"]
    if reason is None and has_box:
        short_side = min(row["bbox_w"] * row["img_w"], row["bbox_h"] * row["img_h"])
        if short_side < min_side:
            reason = f"crop short side < {min_side}px"
    usable = reason is None

    return {
        **row,
        "common_label": rule["common"],
        "common_class_id": config["common_labels"].index(rule["common"]),
        "use_detector_train": has_box,
        "use_detector_eval": has_box,
        "use_classifier_train": usable and row["dataset"] != "taco",  # TACO is the external test set
        "use_classifier_eval": usable,
        "exclude_reason": reason,
    }
