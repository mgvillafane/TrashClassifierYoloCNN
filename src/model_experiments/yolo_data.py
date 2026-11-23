"""Check a YOLO experiment dataset in processed/yolo_classes/ and stage it on local disk for training.

Ultralytics reads thousands of small files per epoch, which is slow on a network volume, so the
training notebook copies the listed images and their label files to the cluster's local disk first.
Torch-free so the checks can be tested without a GPU.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "carton"]
EXPECTED_IMAGES = {  # yaml -> images per role (see scripts/export_yolo_experiments.py)
    "yolo_sortwaste.yaml": {"train": 3705, "val": 780},
    "yolo_sortwaste_trashnet_realwaste.yaml": {"train": 3705 + 2018 + 2870, "val": 780 + 251 + 359},
}
TEST_LISTS = ["taco_all.txt", "sortwaste_test.txt", "trashnet_test.txt", "realwaste_test.txt"]


def read_spec(yaml_path: Path) -> dict:
    """train/val/test as lists of list-file names, names as a list in id order."""
    spec = yaml.safe_load(Path(yaml_path).read_text())
    as_list = lambda v: v if isinstance(v, list) else [v]  # noqa: E731
    names = spec["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    return {"train": as_list(spec["train"]), "val": as_list(spec["val"]), "test": as_list(spec.get("test", [])),
            "names": names}


def read_images(root: Path, list_files: list[str]) -> list[str]:
    """Image paths relative to root (the lists hold './images/...')."""
    paths = []
    for name in list_files:
        paths += [line.strip().removeprefix("./") for line in (Path(root) / name).read_text().splitlines()
                  if line.strip()]
    return paths


def label_for(image_rel: str) -> str:
    """Ultralytics finds a label by swapping the last 'images/' for 'labels/' and the suffix for .txt."""
    head, tail = image_rel.rsplit("images/", 1)
    return f"{head}labels/{Path(tail).with_suffix('.txt').as_posix()}"


def check_dataset(yaml_name: str, spec: dict, train: list[str], val: list[str], tests: list[str]) -> None:
    """Raise on anything that would make the comparison unfair or the run different from the plan."""
    problems = []
    if spec["names"] != CLASS_NAMES:
        problems.append(f"class names {spec['names']} != {CLASS_NAMES}")
    for role, paths in (("train", train), ("val", val)):
        if any(p.startswith("images/taco/") for p in paths):
            problems.append(f"TACO images in {role}: TACO is test data only")
        if len(paths) != len(set(paths)):
            problems.append(f"duplicate images in {role}")
        expected = EXPECTED_IMAGES.get(yaml_name, {}).get(role)
        if expected is not None and len(paths) != expected:
            problems.append(f"{role}: {len(paths)} images, expected {expected}")
    if set(train) & set(val):
        problems.append(f"{len(set(train) & set(val))} images in both train and val")
    if (set(train) | set(val)) & set(tests):
        problems.append(f"{len((set(train) | set(val)) & set(tests))} test images in train/val")
    names = lambda paths: {Path(p).name for p in paths if p.startswith("images/sortwaste/")}  # noqa: E731
    if names(train) & names(val) or (names(train) | names(val)) & names(tests):
        problems.append("a SortWaste photo appears in two splits")
    if problems:
        raise ValueError(f"{yaml_name}: " + "; ".join(problems))


def check_coco_weights(names: dict) -> None:
    """The starting weights must be the COCO-pretrained model (80 classes, 0 = person), not a waste model."""
    if len(names) != 80 or names.get(0) != "person":
        raise ValueError(f"starting weights are not COCO-pretrained (classes: {dict(list(names.items())[:8])}...); "
                         "every YOLO here must start from yolo26m.pt, never from a model trained on TACO")


def stage(root: Path, images: list[str], dest: Path, workers: int = 16) -> list[Path]:
    """Copy images and their labels to dest (same relative layout); return the local image paths."""
    root, dest = Path(root), Path(dest)

    def copy(rel):
        target = dest / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / rel).read_bytes())
        label = dest / label_for(rel)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_bytes((root / label_for(rel)).read_bytes())  # missing label -> error, by design
        return target

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(copy, images))


def write_dataset_yaml(dest: Path, train: list[Path], val: list[Path], names: list[str]) -> Path:
    dest = Path(dest)
    (dest / "train.txt").write_text("".join(f"{p}\n" for p in train))
    (dest / "val.txt").write_text("".join(f"{p}\n" for p in val))
    data = dest / "dataset.yaml"
    data.write_text(yaml.safe_dump({"path": str(dest), "train": "train.txt", "val": "val.txt",
                                    "names": dict(enumerate(names))}, sort_keys=False))
    return data


def fingerprint(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
