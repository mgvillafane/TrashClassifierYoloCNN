"""Figures for the preprocessing summary (single-series bars, so one color and no legend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

from .crop import load_crop, pixel_box
from .paths import DataPaths

BAR = "#2a78d6"
GRID = "#e5e5e5"


def _style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_distribution(table: pd.DataFrame, title: str):
    """One panel per dataset with its own y-scale (SortWaste is ~20x larger than the rest)."""
    fig, axes = plt.subplots(1, len(table.index), figsize=(3.5 * len(table.index), 3.4))
    for ax, ds in zip(axes, table.index):
        ax.bar(table.columns, table.loc[ds], color=BAR, width=0.7)
        ax.set_title(f"{ds} (n={table.loc[ds].sum():,})", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        _style(ax)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_taco_trash(df: pd.DataFrame):
    trash = df[(df["dataset"] == "taco") & (df["common_label"] == "trash") & df["use_classifier_eval"]]
    counts = trash["original_label"].value_counts().sort_values()
    share = counts.get("Cigarette", 0) / counts.sum()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(counts.index, counts.values, color=BAR, height=0.6)
    ax.set_title(f"TACO 'trash' evaluation crops (n={counts.sum()}, {share:.0%} cigarettes)", fontsize=10)
    _style(ax, axis="x")
    fig.tight_layout()
    return fig


def plot_box_sizes(df: pd.DataFrame):
    """Crop short side in pixels for the two detection datasets (log scale)."""
    boxes = df[df["example_type"] == "bounding_box"]
    short = np.minimum(boxes["bbox_w"] * boxes["img_w"], boxes["bbox_h"] * boxes["img_h"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
    bins = [2 ** (i / 4) for i in range(8, 12 * 4 + 1)]  # 4 px .. 4096 px, quarter-octave steps
    for ax, ds in zip(axes, ["sortwaste", "taco"]):
        ax.hist(short[boxes["dataset"] == ds], bins=bins, color=BAR)
        ax.axvline(32, color="#555555", linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"{ds}: crop short side (px)", fontsize=10)
        _style(ax)
    axes[0].annotate("32 px min", xy=(32, 0.95), xycoords=("data", "axes fraction"),
                     xytext=(4, 0), textcoords="offset points", fontsize=8, color="#555555", va="top")
    fig.tight_layout()
    return fig


DATASETS = ["realwaste", "trashnet", "sortwaste", "taco"]
EXCLUDED = "#9a9a9a"


def plot_sample_grid(df: pd.DataFrame, paths: DataPaths, labels: list[str], seed: int = 0):
    """One classifier-eligible example per dataset x common class; empty cells show missing classes."""
    fig, axes = plt.subplots(len(DATASETS), len(labels), figsize=(2 * len(labels), 2.1 * len(DATASETS)))
    eligible = df[df["use_classifier_eval"]]
    for r, ds in enumerate(DATASETS):
        for c, label in enumerate(labels):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_color(GRID)
            pool = eligible[(eligible["dataset"] == ds) & (eligible["common_label"] == label)]
            if pool.empty:
                ax.text(0.5, 0.5, "no examples", ha="center", va="center", fontsize=9, color=EXCLUDED,
                        transform=ax.transAxes)
            else:
                row = pool.sample(1, random_state=seed).iloc[0]
                ax.imshow(load_crop(row, paths, padding=0.1))
            if r == 0:
                ax.set_title(label, fontsize=11)
            if c == 0:
                ax.set_ylabel(ds, fontsize=11)
    fig.suptitle("Sample classifier examples (whole images for RealWaste/TrashNet, box crops for SortWaste/TACO)",
                 fontsize=11)
    fig.tight_layout()
    return fig


FRAME_PICKS = {
    # chosen by eye: a typical belt frame with an ECAL carton, and an outdoor TACO scene
    "sortwaste": "data/raw/sortwaste/dataset/splited_all_dataset/test/images/0008_1872_5.png",
    "taco": "data/raw/taco/data/batch_7/000036.JPG",
}


def plot_annotated_frames(df: pd.DataFrame, paths: DataPaths, picks: dict | None = None):
    """One SortWaste frame and one TACO photo with every box; grey dashed = kept for the detector only."""
    picks = {**FRAME_PICKS, **(picks or {})}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [16 / 9, 4 / 3]})
    for ax, ds in zip(axes, ["sortwaste", "taco"]):
        boxes = df[(df["dataset"] == ds) & (df["example_type"] == "bounding_box")]
        image_id = picks[ds] or boxes["source_image_id"].iloc[0]
        image_rows = boxes[boxes["source_image_id"] == image_id]
        first = image_rows.iloc[0]
        with Image.open(paths.resolve(first["image_path"])) as raw:
            ax.imshow(ImageOps.exif_transpose(raw).convert("RGB"))
        for _, b in image_rows.iterrows():
            x0, y0_, x1, y1 = pixel_box(b["bbox_cx"], b["bbox_cy"], b["bbox_w"], b["bbox_h"], b["img_w"], b["img_h"])
            kept = bool(b["use_classifier_eval"])
            ax.add_patch(Rectangle((x0, y0_), x1 - x0, y1 - y0_, fill=False, linewidth=1.6 if kept else 1.2,
                                   edgecolor=BAR if kept else EXCLUDED, linestyle="-" if kept else "--"))
            ax.text(x0, y0_ - 6, b["common_label"] if kept else b["original_label"], fontsize=7,
                    color="white", bbox={"facecolor": BAR if kept else EXCLUDED, "pad": 1, "linewidth": 0})
        ax.set_title(f"{ds}: {len(image_rows)} boxes in one image", fontsize=10)
        ax.axis("off")
    axes[1].legend(handles=[Rectangle((0, 0), 1, 1, fill=False, edgecolor=BAR, label="classifier + detector"),
                            Rectangle((0, 0), 1, 1, fill=False, edgecolor=EXCLUDED, linestyle="--",
                                      label="detector only (excluded from classifier)")],
                   loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_figures(df: pd.DataFrame, class_table: pd.DataFrame, out_dir: Path, paths: DataPaths | None = None,
                 labels: list[str] | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "class_distribution.png": plot_class_distribution(
            class_table, "Classifier-eligible examples per common class (each panel has its own y-scale)"),
        "taco_trash_composition.png": plot_taco_trash(df),
        "box_short_side.png": plot_box_sizes(df),
    }
    if paths is not None:  # these two read the raw images
        figures["sample_crops.png"] = plot_sample_grid(df, paths, labels or list(class_table.columns))
        figures["annotated_frames.png"] = plot_annotated_frames(df, paths)
    written = []
    for name, fig in figures.items():
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        written.append(out_dir / name)
    return written
