# TACO object detection with YOLO11

The implementation is in `taco_yolo.py`. It uses Ultralytics' YOLO11 Nano, transfers
pretrained visual features, and learns a detection head for TACO's 60 classes.
The default `train` command uses every training image; it does not train from scratch.

This release prepares 1,126 training images (3,470 objects), 197 validation images
(709 objects), and 177 test images (605 objects). The classes represented in these
splits number 59, 52, and 38 respectively; a 60-class prediction head is still used.

## Run it in PowerShell

The environment is already installed for this project. Select
`.venv/Scripts/python.exe` as your IDE interpreter. For a fresh checkout, set it up
with these commands (reuse an existing `.venv` instead of recreating it):

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
uv pip install --python .venv\Scripts\python.exe -r requirements-yolo.txt
.\.venv\Scripts\python.exe -B taco_yolo.py prepare
```

The raw release stays in `data/raw/taco`. Preparation writes
`data/yolo/taco/dataset.yaml`, image copies, YOLO text labels, and a split manifest.
It applies EXIF rotation before resizing to at most 1280 pixels, preserving aspect
ratio. Normalized box coordinates stay valid after resizing. COCO boxes that extend
outside the image are clipped; every clipping/skipping decision is recorded in
`split_manifest.json`.

Splits are reproducible with seed 42. Entire source batches stay together, reducing
leakage from nearby or similar photos. This is a local split, not an official TACO
benchmark split. It cannot guarantee that every related scene across different
batches is separated. Some rare classes may be absent from training or evaluation.

Train a full experiment:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py train --epochs 60 --imgsz 640 --name taco
```

On a configured CUDA machine, add `--device 0`. CPU training works but is slower.
The installation above uses the official CPU wheels tested on this Windows PC;
for CUDA, install the corresponding PyTorch GPU build instead. The newest CPU
build was blocked by Windows Application Control, while 2.9.1 loaded successfully.
The first training run downloads `yolo11n.pt` into `data/yolo/weights`.
Each experiment gets its own folder; when a name already exists, Ultralytics adds
a numeric suffix. Use the actual printed experiment path in subsequent commands.

The example run for this task uses a smaller input and freezes the first ten
backbone layers to fit a CPU demonstration:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py train --epochs 12 --imgsz 320 --batch 8 --freeze 10 --name demo --threads 4 --device cpu
```

Its successful run is saved under `reports/taco_yolo/demo-2`. The earlier `demo`
folder records an interrupted setup attempt without trained weights. Full training
defaults above use 640-pixel inputs and leave the backbone trainable.

## Results from the saved CPU demonstration

The best checkpoint comes from epoch 11 of the 12-epoch run. A separate evaluation
of that saved checkpoint used all 197 validation images and their 709 objects:

| Measure | Result |
|---|---:|
| Mean precision | 48.88% |
| Mean recall | 10.65% |
| mAP50 | 9.58% |
| mAP50-95 | 7.31% |

These are weak baseline results. Scores average over the 52 classes present in
validation, not all 60 possible classes. High mean precision alongside low recall
does not establish a useful detector; read both alongside AP and the failure images.
The test set has not been used to choose these weights or settings.

Artifacts:

- Weights: `reports/taco_yolo/demo-2/weights/best.pt`.
- Learning curves: `reports/taco_yolo/demo-2/results.png`.
- Evaluation: `reports/taco_yolo/validation_demo/metrics.json` and `per_class.csv`.
- Six fixed validation examples: `reports/taco_yolo/predictions/image_000778.jpg`
  through `image_000783.jpg`, with matching JSON files.

In `image_000782.jpg`, the model correctly detects the clear plastic bottle at
confidence 0.495 and misses its small cap: one true positive, zero false positives,
and one false negative at confidence 0.25 and IoU 0.50. Other saved examples contain
misses and class confusions, so this example should not stand in for overall accuracy.

The class breakdown helps explain this difference:

| Class | Validation objects | AP50 |
|---|---:|---:|
| Clear plastic bottle | 46 | 35.13% |
| Plastic bottle cap | 23 | 6.74% |
| Aerosol | 1 | 99.50% |

The aerosol score relies on a single object and is not a reliable estimate of
generalization. Bottle caps are clearly a weakness in this run. Training at a higher
resolution is a reasonable next experiment for small objects; measure the change
on the same validation split rather than assuming it helps.

Reproduce the example with the saved checkpoint:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py predict --weights reports/taco_yolo/demo-2/weights/best.pt --imgsz 320 --split val --image-id 782 --conf 0.25
```

Re-evaluate it in a new output directory:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py evaluate --weights reports/taco_yolo/demo-2/weights/best.pt --imgsz 320 --split val --name validation_repeat
```

Validate and save predictions:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py evaluate --weights reports/taco_yolo/taco/weights/best.pt --split val --name validation
.\.venv\Scripts\python.exe -B taco_yolo.py predict --weights reports/taco_yolo/taco/weights/best.pt --split val --conf 0.25
```

Inspect a particular validation example with `--image-id 123`. The command rejects
IDs from other splits. To use your own photo:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py predict --weights reports/taco_yolo/taco/weights/best.pt --source path/to/photo.jpg
```

After choosing hyperparameters and the confidence threshold using validation,
evaluate the test set once:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py evaluate --weights reports/taco_yolo/taco/weights/best.pt --split test --name final_test
```

## What YOLO learns

A forward pass processes the whole image. The backbone extracts visual features,
the neck combines them at multiple scales, and the detection head predicts object
boxes and class scores. Training adjusts these predictions toward the labeled
boxes and classes. YOLO11 also removes overlapping duplicate predictions during
postprocessing. Transfer learning helps because the pretrained backbone already
recognizes useful shapes and textures, but the original COCO class names are not
TACO's class names. The model must learn the new TACO detection task.

Each generated label row is:

```text
class_id center_x center_y width height
```

Coordinates are fractions of image width/height, between 0 and 1. Category IDs are
mapped explicitly to contiguous YOLO IDs; `dataset.yaml` records the class names.
Only bounding boxes are used here. TACO segmentation polygons remain in the raw JSON.

## Read a result image

The left panel shows the dataset's ground-truth boxes in green. The right panel
shows the model's predicted boxes in orange, with class names and confidence scores.

For the displayed example, a true positive requires the right class and at least
0.50 intersection-over-union (IoU). Each ground-truth object can match only one
prediction. A wrong-class box counts as both a false positive for the prediction
and a missed ground-truth object. Duplicate predictions count as false positives.
The accompanying JSON saves boxes, scores, and example TP/FP/FN counts.

Confidence is the model's ranking score, not a guaranteed probability of being
correct. Raising `--conf` usually removes weak detections and misses more objects;
lowering it usually finds more objects and introduces more false alarms. A pretty
example is useful for debugging, but does not measure overall model performance.

## Analyze the model

Start with `metrics.json` and `per_class.csv` in the evaluation directory:

| Metric | Meaning | What a low value suggests |
|---|---|---|
| Precision | TP / (TP + FP): how many reported objects are correct | Too many false alarms or class confusions |
| Recall | TP / (TP + FN): how many labeled objects are found | Missed litter, especially small or obscured objects |
| mAP50 | Mean area under per-class precision-recall curves at IoU 0.50 | Weak classification and/or localization |
| mAP50-95 | AP averaged over IoU 0.50, 0.55, ..., 0.95 | Boxes are inaccurate even when the broad object is found |

Scores are stored on a 0-1 scale; 0.30 means 30%. There is no single universal
"good" score for TACO. Compare models on the same split, resolution, and metric.
The exported precision and recall are class averages, not pooled counts across
all objects. AP is also averaged only over classes with ground-truth support in
the selected split; `evaluated_classes` records how many were included.
Ultralytics reports aggregate precision/recall near the confidence operating point
that maximizes mean F1; they are not necessarily measured at the `predict` command's
0.25 threshold. AP uses detections over a sweep of confidence thresholds. Evaluation
starts at confidence 0.001 so weak predictions remain available for that sweep.

Use these saved plots together:

1. **`results.png` / `results.csv` in the training folder.** Watch validation mAP
   and box/classification/distribution-focal losses across epochs. Training loss
   falling while validation performance deteriorates suggests overfitting. Both
   staying poor can mean too little training, weak labels, or a difficult setup.
   Training augmentation makes training and validation loss imperfectly comparable.
2. **`BoxPR_curve.png` and `BoxF1_curve.png` in the evaluation folder.** PR shows
   the precision/recall tradeoff; F1 helps choose a confidence threshold. Plot
   filenames can vary with the installed Ultralytics version.
3. **`confusion_matrix_normalized.png`.** Look for classes confused with each other.
   Read the axis labels: a background row/column represents missed objects or
   unmatched predictions, rather than an extra TACO class.
4. **`per_class.csv`.** Inspect object support as well as AP. A high score from two
   objects is much less reliable than a similar score from hundreds. Missing classes
   have blank metrics, not invented zero-performance estimates.
5. **Prediction comparisons.** Inspect easy, crowded, small-object, and failure
   cases. The default examples use fixed image IDs, not selection based on success.

TACO is small and highly imbalanced. Distinguishing all 60 labels is difficult:
cigarettes and plastic film have far more examples than batteries or rare packages.
Batch-based splits can also expose strong domain shifts. An aggregate mAP can hide
failure on an important class, so always inspect class support and failure images.

Improve one thing at a time: train longer if validation still improves, increase
resolution for small objects, try `--weights yolo11s.pt`, or collect more examples
of weak classes. Use the validation split for these decisions. Keep test examples
out of tuning and model selection. A brief CPU demonstration is a working baseline,
not evidence of a deployment-ready detector.

## Check the implementation

```powershell
python -B -m unittest discover -s tests -v
```

The tests cover coordinate conversion, invalid/clipped boxes, batch split isolation,
EXIF rotation, class ID mapping, source preservation, and one-to-one detection
matching. The full training/evaluation run verifies the package integration.

Official references:

- [YOLO11 architecture and pretrained models](https://docs.ultralytics.com/models/yolo11/)
- [Training arguments](https://docs.ultralytics.com/modes/train/)
- [YOLO labels and dataset YAML](https://docs.ultralytics.com/datasets/detect/)
- [Evaluation metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
