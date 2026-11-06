# ResNet-18 fine-tuning experiments

Run from the project directory on GPU compute with the same dependencies, cached
ImageNet weights, dataset, and split CSV as `train_resnet.py`.

```sh
# Recommended first experiment: last residual block and classifier
python train_resnet_finetune.py --epochs 20

# Adapt the last two residual blocks and classifier
python train_resnet_finetune.py --trainable-blocks layer3 layer4 --epochs 20

# Head-only comparison using this script and the same checkpoint criterion
python train_resnet_finetune.py --trainable-blocks fc --warmup-epochs 0 --epochs 20

# Short smoke run: skip the default two warm-up epochs
python train_resnet_finetune.py --epochs 1 --warmup-epochs 0
```

## Changes from train_resnet.py

1. **Selectable layers.** `--trainable-blocks` defaults to `layer4`. You can select
   `layer3`, `layer4`, or both. `fc` is always trained; selecting `fc` alone keeps
   the backbone frozen. These names refer to residual stages containing multiple
   layers. Earlier stages remain frozen.
2. **Warm-up.** The first `--warmup-epochs` epochs (default 2) train only the new
   classifier. Selected backbone parameters become trainable next. Warm-up is
   included in `--epochs`, whose default is now 20: two warm-up and eighteen
   fine-tuning epochs. Set warm-up to 0 to train selected blocks immediately.
   It must be smaller than the total epoch count. Every invocation starts from
   cached ImageNet weights and a new classifier; it does not resume a checkpoint.
3. **Separate learning rates.** Adam uses `--lr` (default 0.001) for the classifier
   and `--backbone-lr` (default 0.0001) for selected backbone blocks. The smaller
   backbone rate limits changes to pretrained features. Both groups are registered
   once, preserving classifier optimizer state across warm-up; frozen parameters
   receive no gradients and are not updated.
4. **Training mode and BatchNorm.** Training calls `model.train()`, then sets all
   BatchNorm modules to evaluation mode. Their pretrained running means and
   variances stay fixed, including in selected blocks. Their affine weights and
   biases can learn when their block is unfrozen. Evaluation mode does not disable
   autograd. Validation calls `model.eval()` and disables gradients.
5. **Checkpoint selection.** `best.pt` now maximizes validation macro F1, giving
   each class equal weight. Accuracy and loss are still logged. Ties keep the
   earlier epoch. `summary.json` records `best_val_macro_f1` and
   `val_accuracy_at_best_epoch`; the latter is accuracy at the selected F1 epoch,
   not the maximum accuracy over all epochs. Warm-up epochs can also win.
6. **Experiment records.** `config.json` records the requested blocks, learning
   rates, warm-up duration, effective trainable modules after warm-up, BatchNorm
   policy, and checkpoint criterion. `history.csv` adds a `phase` column. A head-only
   run labels epochs after warm-up as `finetune`, but still trains only the head.

The original script is unchanged. The new script retains its data augmentation,
ImageNet normalization, shared split, seed settings, GPU requirement, output
location, unique run directories, plots, and checkpoint format. It does not
iterate over the test loader or download weights.

Compare head-only, layer4, and layer3+layer4 runs with the same split, seeds, and
training budget. Check learning curves for overfitting and use multiple seeds
when feasible. Twenty epochs and the learning rates are starting points, not
validated optimal settings. Choose settings using validation results, then use
the held-out test set for final evaluation.
