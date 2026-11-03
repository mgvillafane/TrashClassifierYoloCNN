# TACO object detection with YOLO26

This project fine-tunes a pretrained YOLO26 Medium detector on TACO's 60 classes. It supports
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
yolo/weights/yolo26m.pt      Pretrained YOLO26 Medium weights
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

Edit `CONFIG` in `taco_databricks.py` to change the experiment. Its default
uses YOLO26 Medium for 2 epochs, 640-pixel images, batch 8, the full 1,126-image
training split (`fraction=1.0`), and two prediction examples. Early stopping is
disabled (`patience=0`) so both epochs run. `device="0"` requires a GPU.
Increase `epochs` for a longer experiment; validation and test images stay held out.

The notebook uses the headless Ultralytics package for compatibility with the
server environment. It stages outputs in a writable temporary directory because
the trainer appends to CSV files. Closed TensorBoard event files are published
to the volume after every epoch. The remaining completed outputs are copied
into that same experiment folder after training and evaluation finish.

## TensorBoard

The Databricks experiment publishes losses, learning rates, validation
precision/recall, and mAP after each completed epoch. Initial training-batch
images and class distributions appear with epoch one; prediction comparisons
appear after training. Event files stay in the same experiment folder as the
checkpoints and CSV. Each epoch is a small closed event file, which avoids
append/seek limitations on Unity Catalog volumes and duplicate scalar logs.

Local runs without `tensorboard_dir` use Ultralytics' native TensorBoard callback,
including its model graph. The live Databricks logger records scalars and images.

### View in Databricks

For an **interactive GPU notebook**, run the setup/configuration cells, then the
TensorBoard cell immediately before training. Leave its dashboard visible while
running the training cell. It begins empty and updates as epochs finish.

For a **bundle GPU job**, open the deployed `taco_databricks.py` in a separate
active notebook session with CPU compute. Install the notebook dependencies if
the session is fresh (`%pip install tensorboard==2.21.0 Pillow`, followed by
`dbutils.library.restartPython()` in a separate cell), then run only its final
**TensorBoard viewer** cell. No PyTorch installation is needed just to view live logs. Its
default log directory lists all runs; select the current experiment in the run
selector. Alternatively, set `RUN_TO_VIEW` to the **Live TensorBoard and saved
run:** path printed by the GPU job before training starts. The viewer does not
train or require CUDA. Both sessions read the same volume.

TensorBoard checks for new events every five seconds. Training and validation
metrics arrive at epoch boundaries; nothing appears until epoch one finishes.
Enable automatic reload in the TensorBoard UI, or use its refresh button. Keep
the viewing notebook session active; the GPU job can end independently.

- **Scalars / Time Series**: inspect losses, learning rates, precision, recall,
  and mAP by epoch.
- **Images**: inspect training batches, class distributions, and prediction
  comparisons.
- **Graphs**: available for local runs using the native logger.

For older runs with only `results.csv`, call `prepare_tensorboard_logs(run_dir)`
once or use `display_tensorboard(run_dir)` without `live=True`. Conversion does
not retrain; repeated views reuse the same summary event file. Historical CSV
runs have no recorded model graph.

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

For the completed YOLO26 Medium run, its event files are also downloaded to
`reports/taco_yolo/databricks_yolo26m_2epochs`. To reopen that dashboard in a
normal browser tab, start this command in a terminal and leave it running:

```powershell
.\.venv\Scripts\python.exe -B -m tensorboard.main --logdir reports/taco_yolo/databricks_yolo26m_2epochs --host 127.0.0.1 --port 6006 --load_fast=false --reload_interval 5
```

Open `http://localhost:6006` in a new browser tab. The local viewer reads the
downloaded files; viewing new cloud epochs locally requires copying their new
event files while training runs. The active Databricks viewer reads the volume
directly for live updates.

## Validation and final testing

Training updates weights on 1,126 images. Validation uses 197 separate images
to select checkpoints and compare settings. Final testing evaluates the chosen
checkpoint on the remaining 177 held-out images without changing its weights.

- `mAP50` averages detection precision-recall scores across represented classes
  with the correct class and at least 50% intersection-over-union box overlap.
- `mAP50-95` averages the same measure at overlap thresholds from 50% to 95% in
  5-point steps. It requires more precise localization and is usually lower.

These are detection-quality scores, not the percentage of images classified
correctly. The two-epoch model's validation mAP50 of 2.75% and mAP50-95 of 1.64%
are low scores from a short training run.

To test a saved experiment in Databricks, run only the **Held-out TACO test**
cell in `taco_databricks.py`, selecting `RUN_TO_TEST`. It calls
`test_saved_run()` using the saved TACO-trained `best.pt`, stages evaluation
locally, and saves reports and plots under that experiment's `test/` directory.
It also logs `test/*` metrics to TensorBoard. Final test data stays separate
from validation used for tuning; test scores should not select training settings.

The saved two-epoch YOLO26 Medium checkpoint was tested on all 177 held-out
images using NVIDIA A10G, without retraining: test mAP50 **2.37%**, test
mAP50-95 **1.56%**. The test split contains 38 represented classes out of the
model's 60; absent classes are unscored in the report. Test metrics, per-class
scores, and a confusion-matrix plot are also downloaded into
`reports/taco_yolo/databricks_yolo26m_2epochs/test`.

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

The default local experiment uses pretrained `yolo26m.pt` for 2 epochs at 640
pixels, batch 8, the full training split, no early stopping, and automatic
CUDA/CPU selection.
The CLI also provides `prepare`, `train`, `evaluate`, and `predict`; use
`python taco_yolo.py <command> --help` for their options.

Each experiment keeps:

- `weights/best.pt` and `weights/last.pt`.
- `results.csv`, `results.png`, and `events.out.tfevents.*`.
- `validation/metrics.json`, per-class metrics, and evaluation plots.
- `predictions/` with ground-truth/prediction comparisons.
- `run_config.json` and `artifacts.json`.

The YOLO26 Medium Databricks experiment saves to
`/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_yolo26m_2epochs`.
Additional runs receive numbered experiment directories.

The verified YOLO26 Medium run completed 2 epochs on NVIDIA A10G with all 1,126
training images, 640-pixel inputs, and batch 8. Validation on 197 images gave
mAP50 2.75% and mAP50-95 1.64%; these are results from only two training epochs.
Both epoch logs, checkpoints, validation reports, and prediction examples were
saved. [Open the successful run](https://dbc-fd89e1b5-e4fc.cloud.databricks.com/jobs/701121890710253/runs/213741636457951?o=7474659242886734).

The earlier YOLO11 Nano smoke run remains in
`/Volumes/workspace/default/tacotrashdata/reports/taco_yolo/databricks_small_gpu`.
It ran 2 epochs on NVIDIA A10G, using 113 training images and all 197 validation
images. Validation mAP50 was 0.64% and mAP50-95 was 0.43%; this is a short
pipeline demonstration. The test split is reserved for final scoring.

## Verify the code

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

References:

- [TensorBoard in Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/tensorboard)
- [Ultralytics TensorBoard integration](https://docs.ultralytics.com/integrations/tensorboard/)
- [YOLO training arguments](https://docs.ultralytics.com/modes/train/)
- [YOLO26 models](https://docs.ultralytics.com/models/yolo26/)
- [TensorBoard documentation](https://www.tensorflow.org/tensorboard)
