# Classifier experiments C1–C3

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
```

By default C1 and C3 run `cnn`, `resnet-head`, and `resnet-layer4` in sequence.
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

Each selected checkpoint is then evaluated on TrashNet, RealWaste, and SortWaste
validation separately. C3 also reports pooled metrics on TrashNet and RealWaste.
SortWaste uses a fixed, proportional 500-image subset of its validation crops to
keep this secondary transfer check quick. Set `--sortwaste-val-limit 0` to use
all SortWaste validation crops or pass another limit. Its macro-F1 averages the
classes with ground-truth support (cardboard, metal, and plastic), while its
confusion matrix includes all six prediction columns. TACO and every test
partition are reserved for final testing.

## Outputs

Each invocation writes to `classifier_experiments/<experiment>/<run-id>/`
under `/Volumes/workspace/default/models`. Use `--output-root` to override that
location. The invocation's `comparison.csv` ranks the completed architectures
by the selection split and includes the other validation scores and checkpoint
paths. Each architecture's subdirectory contains `best.pt`, `config.json`,
`history.csv`, `learning_curves.png`, and `validation/` with metrics JSON,
per-class CSVs, and confusion matrices in CSV and PNG form.

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
