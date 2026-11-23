"""Write ready-to-train copies of the manifest that need neither the raw data nor this package.

classifier/  one JPEG per classifier-eligible example (whole image or ground-truth box crop),
             grouped as <dataset>/<split or foldN>/<label>/, plus index.csv with fixed label ids.
detector/    single-class (waste_object) YOLO data: resized upright JPEGs, label files, one image
             list per SortWaste split and per TACO fold, and dataset YAMLs for D1 and D2.
yolo_classes/ the same images with class-aware labels (6 common classes + carton) for the
             end-to-end YOLO baselines E1 (SortWaste) and E2 (SortWaste + TACO folds).

Box coordinates are normalized, so resizing keeps them valid. TACO images are EXIF-transposed
before saving (its boxes refer to the upright image) and saved without EXIF.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from .crop import pixel_box
from .paths import DataPaths

CLASSIFIER_MAX_SIDE = 512
DETECTOR_MAX_SIDE = 1280
JPEG_QUALITY = 95
DETECTOR_CLASS = "waste_object"
CARTON = "carton"
# Beverage cartons map to cardboard but are excluded from the classifier (configs/label_mapping.json).
# Class-aware YOLO gives them their own class, so they are neither noisy cardboard nor background.
CARTON_LABELS = {("sortwaste", "ecal"), ("taco", "Drink carton")}
INDEX_COLUMNS = ["path", "dataset", "partition", "label", "label_id", "original_label",
                 "use_classifier_train", "source_image_id", "taco_annotation_id"]


def partition(row) -> str:
    """Official/stratified split, or 'foldN' for TACO (which has no fixed split)."""
    return f"fold{row['taco_fold']}" if row["dataset"] == "taco" else row["split"]


def image_stem(row) -> str:
    """File stem unique within a dataset: TACO repeats stems across batches, so prefix the batch."""
    stem = Path(row["image_path"]).stem
    return f"batch{row['taco_batch']}_{stem}" if row["dataset"] == "taco" else stem


def classifier_index(df: pd.DataFrame) -> pd.DataFrame:
    """One row per exported classifier file; boxes are numbered within their source image."""
    box_number = df.groupby("image_path").cumcount()
    eligible = df[df["use_classifier_eval"]].copy()
    names = [image_stem(r) if r["example_type"] == "whole_image" else f"{image_stem(r)}_{n:03d}"
             for (_, r), n in zip(eligible.iterrows(), box_number[eligible.index])]
    eligible["partition"] = [partition(r) for _, r in eligible.iterrows()]
    eligible["path"] = (eligible["dataset"] + "/" + eligible["partition"] + "/"
                        + eligible["common_label"] + "/" + pd.Series(names, index=eligible.index) + ".jpg")
    if eligible["path"].duplicated().any():
        raise ValueError(f"duplicate classifier paths: {eligible.loc[eligible['path'].duplicated(), 'path'].tolist()[:5]}")
    index = eligible.rename(columns={"common_label": "label", "common_class_id": "label_id"})
    return index.reindex(columns=INDEX_COLUMNS + ["image_path", "example_type", "img_w", "img_h",
                                                  "bbox_cx", "bbox_cy", "bbox_w", "bbox_h"])


def yolo_class_names(common_labels: list[str]) -> list[str]:
    return [*common_labels, CARTON]


def yolo_class_id(box, carton_id: int) -> int:
    """Common class id, or the extra carton class for beverage cartons."""
    if (box.dataset, box.original_label) in CARTON_LABELS:
        return carton_id
    return int(box.common_class_id)


def detector_index(df: pd.DataFrame, class_id=None) -> pd.DataFrame:
    """One row per detector image with its exported path, partition list and YOLO label lines.

    class_id(box) gives each box's class; the default puts every box in the single class 0.
    """
    class_id = class_id or (lambda box: 0)
    rows = []
    for image_path, group in df[df["use_detector_train"]].groupby("image_path", sort=False):
        first = group.iloc[0]
        folder = first["split"] if first["dataset"] == "sortwaste" else f"batch{first['taco_batch']}"
        boxes = group[group["example_type"] == "bounding_box"]
        rows.append({
            "image_path": image_path,
            "path": f"images/{first['dataset']}/{folder}/{image_stem(first)}.jpg",
            "list": f"{first['dataset']}_{partition(first)}",
            "img_w": first["img_w"], "img_h": first["img_h"],
            "labels": "".join(f"{class_id(b)} {b.bbox_cx:.6f} {b.bbox_cy:.6f} {b.bbox_w:.6f} {b.bbox_h:.6f}\n"
                              for b in boxes.itertuples()),
        })
    index = pd.DataFrame(rows)
    if index["path"].duplicated().any():
        raise ValueError(f"duplicate detector paths: {index.loc[index['path'].duplicated(), 'path'].tolist()[:5]}")
    return index


def label_path(image_rel: str) -> str:
    """Ultralytics finds labels by swapping the last /images/ for /labels/ and the suffix for .txt."""
    head, tail = image_rel.rsplit("images/", 1)
    return f"{head}labels/{Path(tail).with_suffix('.txt').as_posix()}"


def _open_upright(path: Path) -> Image.Image:
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def _shrink(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _write_classifier_image(job) -> int:
    """Open one source image once and save every crop taken from it."""
    source, items, out_dir, max_side = job
    image = _open_upright(source)
    for item in items:
        if item["example_type"] == "whole_image":
            crop = image
        else:
            if image.size != (item["img_w"], item["img_h"]):
                raise ValueError(f"{source}: image is {image.size}, manifest says {(item['img_w'], item['img_h'])}")
            crop = image.crop(pixel_box(item["bbox_cx"], item["bbox_cy"], item["bbox_w"], item["bbox_h"],
                                        item["img_w"], item["img_h"]))
        target = out_dir / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _shrink(crop, max_side).save(target, quality=JPEG_QUALITY)
    return len(items)


def _write_detector_image(job) -> None:
    source, target, label_file, labels, size, max_side = job
    image = _open_upright(source)
    if image.size != size:
        raise ValueError(f"{source}: image is {image.size}, manifest says {size}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _shrink(image, max_side).save(target, quality=JPEG_QUALITY)
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text(labels)


def _run(fn, jobs, workers: int) -> None:
    if workers <= 1:
        for job in jobs:
            fn(job)
        return
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(fn, jobs, chunksize=16):
            pass


def export_classifier(df: pd.DataFrame, paths: DataPaths, out_dir: Path, workers: int = 1,
                      max_side: int = CLASSIFIER_MAX_SIDE) -> pd.DataFrame:
    out_dir = Path(out_dir)
    index = classifier_index(df)
    jobs = [(paths.resolve(image_path), group.to_dict("records"), out_dir, max_side)
            for image_path, group in index.groupby("image_path", sort=False)]
    _run(_write_classifier_image, jobs, workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    index[INDEX_COLUMNS].to_csv(out_dir / "index.csv", index=False)
    return index[INDEX_COLUMNS]


def dataset_yamls(n_folds: int, prefix: str = "d", names: list[str] | None = None) -> dict[str, str]:
    """<prefix>1 = SortWaste; <prefix>2 fold k = SortWaste + the other TACO folds, tested on fold k.

    No 'path' key: Ultralytics then resolves the lists relative to the YAML's own folder.
    """
    names = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names or [DETECTOR_CLASS]))
    yamls = {f"{prefix}1_sortwaste.yaml": "train: sortwaste_train.txt\nval: sortwaste_val.txt\n"
                                  f"test: sortwaste_test.txt\n{names}"}
    for k in range(n_folds):
        train = ", ".join(["sortwaste_train.txt"] + [f"taco_fold{j}.txt" for j in range(n_folds) if j != k])
        yamls[f"{prefix}2_taco_fold{k}.yaml"] = (f"train: [{train}]\nval: sortwaste_val.txt\n"
                                          f"test: taco_fold{k}.txt\n{names}")
    return yamls


# --- YOLO experiment data (end-to-end YOLO vs YOLO boxes + classifier, tested on TACO) ---------------
# TrashNet and RealWaste photos show one object and have no boxes; for YOLO training the whole photo
# is one box of the photo's class. Only classifier-eligible photos are used, so YOLO and the classifier
# see exactly the same TrashNet/RealWaste photos and labels.
WHOLE_IMAGE_DATASETS = ("trashnet", "realwaste")
WHOLE_IMAGE_BOX = "0.500000 0.500000 1.000000 1.000000"
# Test batches of the TACO-trained reference YOLO (taco_yolo.split_by_batch, seed 42: val = batches 3
# and 8, test = batches 2 and 15). Every model is compared on these 177 photos.
TACO_HELDOUT_BATCHES = (2, 15)
TACO_HELDOUT_LIST = "taco_heldout.txt"
EXPERIMENT_YAMLS = {  # yaml name -> datasets it trains and validates on (their own val splits)
    "yolo_sortwaste.yaml": ("sortwaste",),
    "yolo_sortwaste_trashnet_realwaste.yaml": ("sortwaste", "trashnet", "realwaste"),
}


def whole_image_index(df: pd.DataFrame) -> pd.DataFrame:
    """One row per classifier-eligible TrashNet/RealWaste photo, labelled as one box covering it."""
    rows = df[df["dataset"].isin(WHOLE_IMAGE_DATASETS) & (df["example_type"] == "whole_image")
              & df["use_classifier_eval"].astype(bool)]
    index = pd.DataFrame({
        "image_path": rows["image_path"],
        "path": [f"images/{r.dataset}/{r.split}/{r.common_label}/{Path(r.image_path).stem}.jpg"
                 for r in rows.itertuples()],
        "list": rows["dataset"] + "_" + rows["split"],
        "labels": [f"{int(c)} {WHOLE_IMAGE_BOX}\n" for c in rows["common_class_id"]],
    }).reset_index(drop=True)
    if index["path"].duplicated().any():
        raise ValueError(f"duplicate whole-image paths: {index.loc[index['path'].duplicated(), 'path'].tolist()[:5]}")
    return index


def _write_whole_image(job) -> None:
    source, target, label_file, labels, max_side = job
    target.parent.mkdir(parents=True, exist_ok=True)
    _shrink(_open_upright(source), max_side).save(target, quality=JPEG_QUALITY)
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text(labels)


def export_whole_images(df: pd.DataFrame, paths: DataPaths, out_dir: Path, workers: int = 1,
                        max_side: int = DETECTOR_MAX_SIDE) -> pd.DataFrame:
    """Add TrashNet/RealWaste photos (one box each) and their <dataset>_<split>.txt lists to out_dir."""
    out_dir = Path(out_dir)
    index = whole_image_index(df)
    jobs = [(paths.resolve(r.image_path), out_dir / r.path, out_dir / label_path(r.path), r.labels, max_side)
            for r in index.itertuples()]
    _run(_write_whole_image, jobs, workers)
    for name, group in index.groupby("list"):
        (out_dir / f"{name}.txt").write_text("".join(f"./{p}\n" for p in group["path"]))
    return index


def taco_heldout_paths(taco_list_text: str, batches=TACO_HELDOUT_BATCHES) -> list[str]:
    """Lines of a TACO image list (e.g. taco_all.txt) that come from the held-out batches."""
    folders = {f"images/taco/batch{b}/" for b in batches}
    return [line for line in taco_list_text.splitlines()
            if any(line.removeprefix("./").startswith(folder) for folder in folders)]


def experiment_yamls(names: list[str]) -> dict[str, str]:
    """Train and validate on each dataset's own train/val lists; test on the held-out TACO photos.

    No 'path' key: Ultralytics resolves the lists relative to the YAML's own folder.
    """
    def value(datasets, split):
        files = [f"{d}_{split}.txt" for d in datasets]
        return files[0] if len(files) == 1 else "[" + ", ".join(files) + "]"

    names_text = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    return {yaml: f"train: {value(ds, 'train')}\nval: {value(ds, 'val')}\ntest: {TACO_HELDOUT_LIST}\n{names_text}"
            for yaml, ds in EXPERIMENT_YAMLS.items()}


def export_detector(df: pd.DataFrame, paths: DataPaths, out_dir: Path, n_folds: int, workers: int = 1,
                    max_side: int = DETECTOR_MAX_SIDE, class_names: list[str] | None = None) -> pd.DataFrame:
    """Single-class data (D1/D2), or class-aware data (E1/E2) when class_names is given."""
    out_dir = Path(out_dir)
    carton_id = class_names.index(CARTON) if class_names else None
    index = detector_index(df, (lambda box: yolo_class_id(box, carton_id)) if class_names else None)
    jobs = [(paths.resolve(r.image_path), out_dir / r.path, out_dir / label_path(r.path), r.labels,
             (r.img_w, r.img_h), max_side) for r in index.itertuples()]
    _run(_write_detector_image, jobs, workers)
    # Ultralytics expands a leading './' in a list file to the list file's own folder.
    for name, group in index.groupby("list"):
        (out_dir / f"{name}.txt").write_text("".join(f"./{p}\n" for p in group["path"]))
    taco = index.loc[index["list"].str.startswith("taco_"), "path"]
    if len(taco):
        (out_dir / "taco_all.txt").write_text("".join(f"./{p}\n" for p in taco))
    prefix = "e" if class_names else "d"
    for name, text in dataset_yamls(n_folds, prefix, class_names).items():
        # skip YAMLs whose lists were not exported (e.g. only SortWaste was requested)
        if all((out_dir / f).exists() for f in re.findall(r"[\w]+\.txt", text)):
            (out_dir / name).write_text(text)
    return index
