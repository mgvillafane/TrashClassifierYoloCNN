# Training TrashCNN from scratch

Use the same GPU environment and dependencies as the ResNet experiments. Run
commands from the project directory. No pretrained weights are loaded.

```sh
# Single run
python train_cnn.py --epochs 40 --lr 0.001 --weight-decay 0.0001

# Preview all six grid commands without PyTorch, a GPU, or output files
python grid_cnn.py --dry-run

# Run the default grid on GPU compute
python grid_cnn.py

# Custom grid: 3 learning rates x 2 weight decays x 2 batch sizes = 12 runs
python grid_cnn.py --learning-rates 0.0003 0.001 0.003 --weight-decays 0 0.0001 --batch-sizes 16 32 --epochs 40

# Repeat a promising configuration across three seeds
python grid_cnn.py --learning-rates 0.001 --weight-decays 0.0001 --seeds 42 43 44
```

## Training behavior

`train_cnn.py` follows the structure of `train_resnet.py`, but constructs your
`TrashCNN`, trains every parameter with AdamW, and uses normal training-mode
BatchNorm rather than freezing its statistics. Cross-entropy consumes raw logits.
The default learning rate is 0.001 and weight decay is 0.0001. Weight decay is
regularization that discourages large weights; zero disables it. It applies to
all parameters, including biases and BatchNorm affine parameters.

Each run uses the shared split, 224-pixel images, horizontal-flip augmentation,
and ImageNet normalization from the existing loader. This normalization is just
input preprocessing; it does not make the model pretrained. Seed settings and
output conventions match the other training scripts.

Every run saves config.json, split.csv, history.csv, learning_curves.png, best.pt,
and summary.json. Checkpoints maximize validation macro F1; accuracy in the
summary is measured at that selected epoch. Equal F1 retains the earlier epoch.
The test loader is never iterated. These are inference checkpoints, not complete
training-resume checkpoints: optimizer/RNG state is not saved.

## Grid behavior and interpreting results

The default grid tests learning rates [0.0003, 0.001, 0.003] and weight decays
[0, 0.0001], keeping batch size 32 and seed 42 fixed: six sequential runs of
40 epochs (240 total epochs). These are initial search values, not proven best
settings. There is no scheduler or early stopping, so each successful trial gets
the same epoch budget. Look at the curves and extend promising runs if they are
still improving; every invocation starts fresh rather than resuming.

A fresh subprocess trains each trial with the current Python interpreter.
The runner saves grid_config.json, updates results.csv after each finished trial
(sorted by best validation macro F1), and writes best_trial.json when complete.
Each trial directory contains its individual training run. Failed runs are marked
and the grid continues; partial failures produce a nonzero exit status at the
end. Interrupted grids retain completed rows but do not automatically resume.

Default outputs are /Volumes/workspace/default/models/cnn_experiments for single
runs and /Volumes/workspace/default/models/cnn_grids for grids. Both scripts
accept --output-root, --data-root, --manifest-path, and --num-workers.

With multiple seeds, results.csv ranks individual runs, not means across seeds.
Compare mean and spread across seeds before choosing among close configurations;
best_trial.json alone does not establish the best average configuration. Use
validation results for tuning and reserve the test set for final evaluation.

## Local validation

Syntax, CLI help, a dry-run grid, and mocked grid ranking/failure handling were
checked without launching GPU training. Actual model training remains to be
validated in the GPU environment.
