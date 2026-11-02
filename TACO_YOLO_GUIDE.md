# TACO object detection with YOLO11

This project trains a YOLO11 Nano detector on TACO's 60 classes. It supports
local training and the uploaded dataset in Databricks. Raw images and source
annotations remain unchanged.

## Project files

- `taco_yolo.py`: prepare, train, validate, predict, and visualize results.
- `taco_databricks.py`: the Databricks GPU experiment and TensorBoard viewer.
- `databricks.yml`: the `taco_small_gpu` job on one serverless A10 GPU.
- `requirements-yolo.txt`: dependencies for local training and visualization.
- `tests/test_taco_yolo.py`: dataset, experiment, and TensorBoard checks.

The dataset stays in `data/`; saved local experiments stay in `reports/`.
Databricks connection settings and bundle state stay in `.databricks/`.

## Run on Databricks

The uploaded volume is `/Volumes/workspace/default/tacotrashdata`:

```text
taco/annotations.json         Raw COCO annotations
taco/batch_*/                Raw images
yolo/taco/images/            Prepared images
yolo/taco/labels/            YOLO labels
yolo/taco/dataset.yaml       Dataset configuration
yolo/weights/yolo11n.pt      Pretrained weights
reports/taco_yolo/           Saved experiments
```

In VS Code, press **Ctrl+Shift+P**, choose **Databricks: Deploy the bundle
and run a resource**, and select **taco_small_gpu**. You can also select the job
in the Databricks **Bundle Resource Explorer** and click its play button.
The editor's **Run File as Workflow** action uses selected compute and does
not apply the bundle job's A10 GPU settings.

With the Databricks CLI on PATH:

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run taco_small_gpu -t dev
```

Edit `CONFIG` in `taco_databricks.py` to change the experiment. Its small default
uses 2 epochs, 320-pixel images, batch 16, 10% of the training split, and two
prediction examples. `device="0"` requires a GPU. Increase `fraction` to 1.0
and increase `epochs` for a full experiment.

The notebook uses the headless Ultralytics package for compatibility with the
server environment. It stages outputs in a writable temporary directory because
the trainer appends to CSV files. It then copies the completed experiment into
one folder in the volume and updates the artifact paths.

## TensorBoard

New training runs enable Ultralytics' TensorBoard callback automatically.
Event files are kept inside the same experiment folder as the checkpoints and
CSV. The callback logs losses, learning rates, validation precision/recall, mAP,
and the model graph. A summary event file adds training-batch images, class
distribution plots, and saved prediction comparisons.

### View in Databricks

Open `taco_databricks.py` as a notebook and run its final **TensorBoard viewer**
cell in an active notebook session. Edit `RUN_TO_VIEW` to select a saved run.
The viewer can use CPU compute; it does not train or require CUDA.
Install the notebook's dependencies first if the session is fresh.

The training job exits before the viewer cell. This preserves its event files
without starting a short-lived dashboard in a completed job. An interactive
TensorBoard UI needs an active notebook session.

- **Scalars / Time Series**: inspect losses, learning rates, precision, recall,
  and mAP by epoch.
- **Images**: inspect training batches, class distributions, and prediction
  comparisons.
- **Graphs**: inspect the model graph recorded by new training runs.

Older runs with only `results.csv` are converted when the viewer is opened.
The conversion does not retrain, and repeated views reuse the same summary
event file. Historical CSV runs have no recorded model graph.

For multiple runs in one notebook, prepare any older run directories first,
then start TensorBoard on the parent directory:

```python
from tensorboard import notebook
notebook.start("--logdir /Volumes/workspace/default/tacotrashdata/reports/taco_yolo --load_fast=false")
```

### View locally

Install dependencies in the project's existing virtual environment:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-yolo.txt
```

New local experiments generate their event files automatically. To convert an
older saved run, for example:

```powershell
.\.venv\Scripts\python.exe -B -c "from taco_yolo import prepare_tensorboard_logs; prepare_tensorboard_logs('reports/taco_yolo/demo-2')"
.\.venv\Scripts\python.exe -B -m tensorboard.main --logdir reports/taco_yolo --port 6006 --load_fast=false
```

Open `http://localhost:6006`. Databricks volume paths are remote; download
their event files into a local report directory to visualize them locally.

## Data preparation

`prepare()` reads COCO annotations, keeps each source batch in a single split,
applies EXIF rotation to image copies, resizes them, and creates normalized YOLO
labels. The split contains 1,126 training, 197 validation, and 177 test images.

Existing data is reused when its annotation hash and preparation settings match
and every expected image and label exists. Moving the same dataset from Windows
to Databricks is allowed. The code refreshes `dataset.yaml` for the current
location and reports genuine settings differences by name. The raw
`annotations.json` must still be present for that check.

## Local experiments and saved outputs

Edit `DEFAULT_EXPERIMENT` in `taco_yolo.py`, then run:

```powershell
.\.venv\Scripts\python.exe -B taco_yolo.py
```

The default local experiment uses 60 epochs and automatic CUDA/CPU selection.
The CLI also provides `prepare`, `train`, `evaluate`, and `predict`; use
`python taco_yolo.py <command> --help` for their options.

Each experiment keeps:

- `weights/best.pt` and `weights/last.pt`.
- `results.csv`, `results.png`, and `events.out.tfevents.*`.
- `validation/metrics.json`, per-class metrics, and evaluation plots.
- `predictions/` with ground-truth/prediction comparisons.
- `run_config.json` and `artifacts.json`.

The completed Databricks smoke run is in
`/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_small_gpu`.
It ran 2 epochs on NVIDIA A10G, using 113 training images and all 197 validation
images. Validation mAP50 was 0.64% and mAP50-95 was 0.43%; this is a short
pipeline demonstration. The test split is reserved for final scoring.
Additional completed runs receive numbered experiment directories.

## Verify the code

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

References:

- [TensorBoard in Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/tensorboard)
- [Ultralytics TensorBoard integration](https://docs.ultralytics.com/integrations/tensorboard/)
- [YOLO training arguments](https://docs.ultralytics.com/modes/train/)
- [TensorBoard documentation](https://www.tensorflow.org/tensorboard)
