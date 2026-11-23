"""Build one comparison results.pdf from saved metrics and object labels."""

import argparse
import csv
import json
from pathlib import Path


def plot_report(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    directory = Path(directory)
    summary = json.loads((directory / "metrics.json").read_text())
    systems = [key for key in ("yolo", "yolo_resnet", "yolo_post_resnet") if key in summary["systems"]]
    names = ["YOLO alone", "YOLO + C4", "YOLO + C4 post-trained"][:len(systems)]
    colors = ["#315f91", "#ce6a37", "#36835d"][:len(systems)]
    primary = [summary["systems"][key]["primary_classifier_eligible"] for key in systems]
    supported = primary[0]["macro_f1_classes"]
    support = {row["label"]: row["support"] for row in primary[0]["per_class"]}
    count = primary[0]["count"]
    with (directory / "objects.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["classifier_eligible"] == "True" and row["prediction_index"] != ""]
    if len(rows) != count or any(item["count"] != count for item in primary):
        raise ValueError("All systems must classify the same matched test objects")
    for item in primary:
        if item["macro_f1_classes"] != supported or {row["label"]: row["support"] for row in item["per_class"]} != support:
            raise ValueError("Test class supports must agree across systems")
    detector = summary["config"].get("detector_name", "SortWaste YOLO")
    dataset = summary["config"].get("dataset_name", "SortWaste")
    taco = summary["config"].get("dataset") == "taco"
    if taco and supported != ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        raise ValueError("The TACO report must evaluate all six classes")
    training = summary.get("training")
    history = summary.get("training_history")
    if training:
        if training["epochs_completed"] != 10 or not training["saved_checkpoint_verified"]:
            raise ValueError("Expected a verified ten-epoch training run")
        if [int(row["epoch"]) for row in history] != list(range(1, 11)):
            raise ValueError("Expected all ten training epochs")
        selected = max(history, key=lambda row: (float(row["val_macro_f1"]), float(row["val_accuracy"])))
        if int(selected["epoch"]) != training["best_epoch"]:
            raise ValueError("Checkpoint selection disagrees with validation history")

    target = directory / "results.pdf"
    with PdfPages(target) as pdf:
        pdf.infodict()["Title"] = f"{detector}: {dataset} three-way classification comparison"
        fig, axes = plt.subplots(1, 3, figsize=(11.7, 8.3), sharey=True)
        fig.subplots_adjust(top=.62, bottom=.30, left=.07, right=.98, wspace=.14)
        fig.suptitle(f"{dataset} TEST: {detector} + C4 ResNet", fontsize=17, y=.97)
        population = "All six classes" if taco else "True cartons excluded"
        fig.text(.5, .925, f"{summary['test_images']} held-out test images | {count:,} identical matched objects | {population}",
                 ha="center", fontsize=10, color="#555555")
        if training:
            fig.text(.5, .891, f"Post-training: 10 epochs on training data; epoch {training['best_epoch']} selected on validation before test evaluation",
                     ha="center", fontsize=10, color="#555555")
        table_axis = fig.add_axes([.06, .735, .88, .12])
        table_axis.axis("off")
        table_rows = []
        for system, name, item in zip(systems, names, primary):
            timing_key = {"yolo": "yolo_ms", "yolo_resnet": "hybrid_ms", "yolo_post_resnet": "post_hybrid_ms"}[system]
            time = summary["timing"][timing_key]["median_ms"]
            table_rows.append([name, f"{item['accuracy']:.2%}", f"{item['macro_f1']:.3f}", f"{item['weighted_f1']:.3f}", f"{time:.2f} ms"])
        table = table_axis.table(cellText=table_rows,
            colLabels=["System", "Test accuracy", "Macro-F1", "Weighted-F1", "Median A10 time"],
            colWidths=[.32, .17, .16, .17, .18], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        width = .25 if len(systems) == 3 else .36
        x = np.arange(len(supported))
        labels = [f"{label.capitalize()}\n(n={support[label]:,})" for label in supported]
        for axis, metric, title in zip(axes, ("precision", "recall", "f1"), ("Test precision", "Test recall", "Test F1 score")):
            previous = []
            for index, item in enumerate(primary):
                values = [next(row[metric] for row in item["per_class"] if row["label"] == label) for label in supported]
                bars = axis.bar(x + (index - (len(systems) - 1) / 2) * width, values, width=width, color=colors[index], label=names[index])
                annotations = axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3,
                                             fontsize=6 if taco else 7, rotation=90 if taco else 0)
                for position, (annotation, value) in enumerate(zip(annotations, values)):
                    close = sum(abs(value - earlier[position]) < .045 for earlier in previous)
                    if close and not taco:
                        annotation.set_position((0, 3 + 12 * close))
                previous.append(values)
            axis.set_xticks(x, labels, fontsize=7 if taco else 9, rotation=35 if taco else 0,
                            ha="right" if taco else "center")
            axis.set_ylim(0, 1.22)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.set_title(title, fontsize=12, pad=10)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Score")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .70),
                   ncol=len(systems), frameon=False, fontsize=9)
        eligible = summary["counts"]["classifier_eligible_ground_truth"]
        fig.text(.07, .18, f"Objects compared: {count:,} of {eligible:,} eligible annotated items were located ({count / eligible:.1%}).\n"
                 "Classification percentages apply to those matched objects. Missed objects cannot be recovered by changing the classifier.",
                 fontsize=10, linespacing=1.7)
        fig.text(.07, .09, f"All systems use the same fixed {detector} boxes: 640-pixel input, confidence 0.35, matching IoU 0.50.\n"
                 "C4 is the original resnet-layer4 classifier. Its post-trained version adapts layer4 and the head; YOLO stays frozen.",
                 fontsize=10, linespacing=1.7, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

        matrices = []
        predicted_labels = supported if taco else [*supported, "Other labels"]
        for system in systems:
            matrix = np.zeros((len(supported), len(predicted_labels)), dtype=int)
            for row in rows:
                prediction = row[system + "_class"]
                if taco and prediction not in supported:
                    raise ValueError(f"Unexpected TACO prediction: {prediction}")
                column = supported.index(prediction) if prediction in supported else len(supported)
                matrix[supported.index(row["gt_class"]), column] += 1
            matrices.append(matrix)
        maximum = max(int(matrix.max()) for matrix in matrices)
        fig, axes = plt.subplots(1, len(systems), figsize=(11.7, 8.3), squeeze=False)
        axes = axes[0]
        fig.subplots_adjust(top=.79, bottom=.42, left=.09, right=.92, wspace=.40)
        fig.suptitle(f"{detector}: TEST confusion matrices", fontsize=17, y=.97)
        fig.text(.5, .925, "All panels use the same object-count color scale. Parentheses show percentages within each true class.",
                 ha="center", fontsize=10, color="#555555")
        for axis, matrix, name in zip(axes, matrices, names):
            fractions = matrix / matrix.sum(axis=1, keepdims=True)
            image = axis.imshow(matrix, vmin=0, vmax=maximum, cmap="Blues", aspect="auto")
            for i in range(len(supported)):
                for j in range(len(predicted_labels)):
                    axis.text(j, i, f"{matrix[i, j]:,}\n({fractions[i, j]:.1%})", ha="center", va="center", fontsize=6.5 if taco else 8,
                              color="white" if matrix[i, j] > maximum * .5 else "black")
            axis.set_xticks(range(len(predicted_labels)), [label.capitalize() for label in predicted_labels], rotation=45 if taco else 35, ha="right", fontsize=7 if taco else 8)
            axis.set_yticks(range(len(supported)), [f"{label.capitalize()}\n(n={support[label]:,})" for label in supported], fontsize=7 if taco else 8)
            axis.set_xlabel("Predicted material", fontsize=9)
            axis.set_title(name, fontsize=11, pad=12)
        axes[0].set_ylabel("True material")
        fig.colorbar(image, ax=list(axes), fraction=.02, pad=.025, label="Objects")
        label_note = ("All six true and predicted classes are displayed; every matched-object classification error remains in the scores."
                      if taco else "Other labels includes predictions outside the supported true classes; those errors remain in the scores.")
        fig.text(.07, .265, "Rows have different numbers of examples: " + ", ".join(f"{support[label]:,} {label}" for label in supported) + ".\n"
                 "Read across a row for recall; false positives from other rows reduce a class's precision. F1 combines both.\n"
                 + label_note,
                 fontsize=10, linespacing=1.7, va="top")
        outcomes = [summary["paired_primary_outcomes"]]
        if len(systems) == 3:
            outcomes.append(summary["post_training_primary_outcomes"])
        text = []
        for name, outcome in zip(names[1:], outcomes):
            text.append(f"{name}: corrected {outcome.get('resnet_corrected_yolo', 0):,} YOLO errors; introduced {outcome.get('resnet_broke_correct_yolo', 0):,} new errors.")
        fig.text(.07, .115, "\n".join(text), fontsize=10, linespacing=1.7, color="#555555", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        if training:
            epochs = [int(row["epoch"]) for row in history]
            best_epoch = training["best_epoch"]
            fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
            fig.subplots_adjust(top=.80, bottom=.43, left=.08, right=.96, wspace=.23)
            fig.suptitle(f"{dataset} ten-epoch post-training: TRAIN and VALIDATION history", fontsize=16, y=.97)
            fig.text(.5, .925, "This page explains checkpoint selection. The final TEST comparison is on pages 1-2.", ha="center", fontsize=10, color="#555555")
            axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="Training loss", color=colors[0])
            axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="Validation loss", color=colors[1])
            axes[0].set_ylabel("Cross-entropy loss")
            axes[0].set_title("Loss during adaptation")
            axes[1].plot(epochs, [float(row["val_macro_f1"]) for row in history], 'o-', color=colors[2], label="Post-training validation F1")
            axes[1].axhline(training["initial_validation"]["macro_f1"], linestyle=':', color=colors[1], label="Original C4 validation F1")
            axes[1].axhline(training["yolo_validation"]["macro_f1"], linestyle='--', color=colors[0], label="YOLO validation F1")
            axes[1].set_title("Selection by validation macro-F1")
            axes[1].set_ylabel("Macro-F1")
            for axis in axes:
                axis.axvline(best_epoch, color="#777777", linestyle='--', alpha=.6)
                axis.set_xlabel("Epoch")
                axis.set_xticks(epochs)
                axis.spines[["top", "right"]].set_visible(False)
                axis.grid(alpha=.2)
                axis.legend(fontsize=8)
            config = training["config"]
            train_crops = sum(config["crop_counts"]["train"]["matched_counts"].values())
            fig.text(.07, .30, f"Completed all 10 epochs, using {train_crops:,} matched training crops per epoch. Best validation checkpoint: epoch {best_epoch}.\n"
                     f"Official split: {config['crop_counts']['train']['images']:,} training images; {config['crop_counts']['val']['images']:,} validation images.\n"
                     "The test set did not select the checkpoint, confidence threshold, or training hyperparameters.",
                     fontsize=10, linespacing=1.8, va="top")
            fig.text(.07, .17, "Initialization: original C4 ResNet-layer4. YOLO frozen; ResNet layer4 and classifier head trainable.\n"
                     "AdamW, batch size 64, head LR 0.0001, layer4 LR 0.00001, seed 42; natural training class frequencies.\n"
                     f"Post-training uses {dataset} training crops only; validation selects the checkpoint; no test labels enter training.",
                     fontsize=10, linespacing=1.8, va="top", color="#555555")
            fig.text(.07, .05, "Selected model: " + training["best_checkpoint"], fontsize=8, color="#555555")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Report: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "reports/sortwaste_yolo_resnet_test")
    plot_report(parser.parse_args().directory)
