# Classifier experiments C1–C4

Run from the project directory on Databricks GPU compute with the packages in
`requirements-fabien.txt`. The trainer reads the prepared `classifier/index.csv`
and its images under `/Volumes/workspace/default/waste_data/processed/classifier`.
Use `--data-root` for another copy of the export and `--weights` for another copy
of the cached ImageNet ResNet-18 weights.

```sh
python train_classifier.py C1 --dry-run
python train_classifier.py C1

# Example if resnet-layer4 ranks first in C1's comparison.csv
python train_classifier.py C2 --model resnet-layer4

python train_classifier.py C3

# Add SortWaste with class-and-dataset-balanced sampling
python train_classifier.py C4 --model resnet-layer4 --dry-run
python train_classifier.py C4 --model resnet-layer4
```

By default C1, C3, and C4 run `cnn`, `resnet-head`, and `resnet-layer4` in sequence.
Use `--model` to run any one separately. C2 takes the architecture ranked first
by C1's TrashNet validation macro-F1, with accuracy as the tiebreaker. The CNN
defaults to 55 epochs and both ResNet variants default to 20. `--epochs`
overrides the budget for every selected model. All models default to batch
size 32, seed 42, and AdamW. In the layer4 variant the
classifier trains alone for the first two epochs, then layer4 trains at a lower
learning rate. Both ResNet variants keep pretrained BatchNorm statistics fixed.
The custom CNN uses `--cnn-lr 0.0003`, each ResNet head uses
`--head-lr 0.001`, and the fine-tuned layer 4 uses
`--backbone-lr 0.0001` by default.

The training loader samples a random resized crop targeting 80–100% of the
input image area, applies horizontal and vertical flips independently at 50%
each, samples a rotation from −25° to +25°, and jitters brightness and contrast
by up to 25% each. The run configuration records these settings. Validation and
inference use a deterministic resize and the same normalization as training.

| Experiment | Training data | Checkpoint selection |
| --- | --- | --- |
| C1 | TrashNet train | TrashNet val |
| C2 | RealWaste train | RealWaste val |
| C3 | TrashNet + RealWaste train | Pooled TrashNet + RealWaste val |
| C4 | TrashNet + RealWaste + SortWaste train | Mean of the three dataset macro-F1 scores |

## C4 balanced sampling

SortWaste has 48,193 eligible training crops, including 45,724 plastic crops,
and only contains cardboard, metal, and plastic labels. Appending every row to
C3 with ordinary shuffling would make 88.3% of the combined training set
plastic. C4 therefore uses weighted sampling with replacement while C1–C3 keep
their existing shuffled loaders.

For an example with class `c` from dataset `d`, C4 assigns the sampling weight

```text
1 / (number of datasets containing c * number of examples of c in d)
```

The total sampling probability is equal for all six classes. Within each class,
it is divided equally between the datasets that contain that class. With the
default 6,000 draws per epoch, the expected composition is approximately 1,000
examples per class, 2,500 examples from TrashNet, 2,500 from RealWaste, and
1,000 from SortWaste. Use `--c4-samples-per-epoch` to change the epoch size.

Sampling uses replacement, so scarce examples can appear more than once in an
epoch. The training augmentation produces a new random crop, flips, rotation,
and color jitter each time an example is loaded. The run's `config.json` records
the weight formula, raw dataset/class counts, replacement setting, and number
of draws. The weights choose which examples enter each batch; the loss remains
ordinary unweighted cross-entropy.

Each selected checkpoint is then evaluated on TrashNet, RealWaste, and SortWaste
validation separately. C3 also reports pooled metrics on TrashNet and RealWaste,
while C4 calculates macro-F1 and accuracy separately for all three datasets.
C4 selects the checkpoint with the highest mean of the three macro-F1 scores,
using mean dataset accuracy as the tiebreaker. Its `history.csv` contains both
averages plus each dataset's metrics for every epoch. The selected epoch and its
averaged and per-dataset scores are also written to
`validation/selection_metrics.json`.

For C4 checkpoint selection, SortWaste uses a fixed class-balanced subset of
250 cardboard, 250 metal, and 250 plastic images. Change this with
`--c4-sortwaste-val-per-class`. After loading the selected checkpoint, C4
generates final reports with all 10,513 SortWaste validation images. C1–C3 keep
the proportional subset controlled by `--sortwaste-val-limit`, which defaults
to 500 images.

Each final `<dataset>_per_class.csv` records accuracy, precision, recall, F1,
and support for all six labels. Per-class accuracy is the fraction of actual
examples from that class predicted correctly, so it is numerically equal to
recall. SortWaste's macro-F1 averages its three supported classes, while its
confusion matrix retains all six prediction columns. TACO and every test
partition are reserved for final testing.

## Outputs

Each invocation writes to `classifier_experiments/<experiment>/<run-id>/`
under `/Volumes/workspace/default/models`. Use `--output-root` to override that
location. The invocation's `comparison.csv` ranks the completed architectures
by the selection score and includes the other validation scores and checkpoint
paths. For C4, `dataset_average_*` contains the balanced selection score used
for ranking and `final_dataset_average_*` contains the average from the full
final reports. Each architecture's subdirectory contains `best.pt`, `config.json`,
`history.csv`, `learning_curves.png`, and `validation/` with metrics JSON,
per-class accuracy and F1 CSVs, and confusion matrices in CSV and PNG form.

## Predict labels for YOLO crops

```python
from classifier import CropClassifier

classifier = CropClassifier("/path/to/C3/run-id/resnet-layer4/best.pt")
predictions = classifier.predict(rgb_crops)
# Each result contains label_id, label, and confidence, in crop order.
```

Pass a batch of PIL images or RGB `uint8` NumPy arrays in height × width × 3
format. Convert OpenCV crops from BGR to RGB before passing them as arrays.
Inference uses the checkpoint's label order and preprocessing. Crops are resized
to 224 × 224 with bilinear interpolation, scaled to [0, 1], and normalized with
ImageNet mean and standard deviation. Confidence is the maximum softmax score.
