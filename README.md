Fabien :

In this branch I train basic CNN and ResNet on TrashNet. 
I also download both models. 

## Classifier experiment results

Fine-tuning ResNet-18's last residual block (`resnet-layer4`) gave the highest selection macro-F1 in C1, C2, C3, and the completed C4 runs. It scored **0.896** on TrashNet in C1, **0.889** on RealWaste in C2, **0.901** on pooled TrashNet and RealWaste in C3, and **0.847** across C4's three balanced per-dataset validation scores.

| Run | Model | Learning rate | Best epoch | TrashNet F1 | RealWaste F1 | SortWaste F1 | Pooled F1 | Dataset-average F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | `resnet-layer4` | `0.001` head, `0.0001` layer4 | 16 | **0.896** | 0.352 | 0.281 | — | — |
| C1 | `resnet-head` | `0.001` head | 20 | 0.814 | 0.396 | 0.185 | — | — |
| C1 | `cnn` | `0.001` | 55 | 0.769 | 0.199 | 0.197 | — | — |
| C1 CNN rerun | `cnn` | `0.0003` | 39 | 0.780 | 0.196 | 0.132 | — | — |
| C2 | `resnet-layer4` | `0.001` head, `0.0001` layer4 | 10 | 0.329 | **0.889** | 0.270 | — | — |
| C2 | `cnn` | `0.0003` | 39 | 0.165 | 0.719 | 0.140 | — | — |
| C3 | `resnet-layer4` | `0.001` head, `0.0001` layer4 | 20 | 0.932 | 0.886 | 0.272 | **0.901** | — |
| C3 | `resnet-head` | `0.001` head | 20 | 0.752 | 0.760 | 0.259 | 0.764 | — |
| C3 | `cnn` | `0.0003` | 48 | 0.719 | 0.706 | 0.302 | 0.713 | — |
| C4 | `resnet-layer4` | `0.001` head, `0.0001` layer4 | 14 | **0.884** | **0.880** | **0.530** | — | **0.765** |
| C4 | `cnn` | `0.0003` | 50 | 0.750 | 0.658 | 0.337 | — | 0.582 |

C1 trained on 2,018 TrashNet images and ranked models on TrashNet validation. The CNN rerun lowered the learning rate from `0.001` to `0.0003` with the same 55-epoch budget, seed, batch size, and augmentation settings. TrashNet macro-F1 rose from 0.769 to 0.780, while accuracy fell from 78.5% to 77.7%. Two earlier incomplete C1 CNN attempts both used `0.001`.

C1 trains only on TrashNet, C2 trains only on RealWaste, and C3 trains on both. C1 and C2 select checkpoints on their respective validation datasets, while C3 uses the pooled TrashNet and RealWaste validation set. C2 performs strongly on RealWaste but transfers poorly to TrashNet; C3 maintains strong performance on both datasets.

C3 added 2,870 RealWaste training images to the 2,018 TrashNet images. Its winner reached 90.2% pooled validation accuracy. In every `resnet-layer4` run, the head trained alone for two epochs before layer4 was unfrozen.

C4 drew 6,000 balanced samples with replacement per epoch from TrashNet, RealWaste, and SortWaste. ResNet-layer4 was selected at epoch 14 with 0.847 mean macro-F1 and 84.7% mean accuracy across TrashNet, RealWaste, and a balanced 750-image SortWaste selection subset. The CNN was selected at epoch 50 with 0.698 mean macro-F1 and 70.6% mean accuracy. On final validation using all 10,513 SortWaste images, their three-dataset average macro-F1 scores were 0.765 and 0.582, respectively.

The original C1 comparison and C4 evaluated all 10,513 SortWaste validation crops. The C1 CNN rerun, C2, and C3 each used the same fixed 500-image subset, so their SortWaste scores use a different sample. All scores above come from validation splits; C4's Dataset-average F1 column is the mean of its three displayed final per-dataset scores.

The run directories below are under `/Volumes/workspace/default/models/classifier_experiments/`. Each contains a `comparison.csv` and a `best.pt` checkpoint in each completed model's subdirectory.

- C1 comparison · `C1/20261119T042847Z_50587bf0/`
- C1 CNN rerun · `C1/20261119T054151Z_b672c674/`
- C2 ResNet-layer4 · `C2/20261119T182605Z_7c529570/`
- C2 CNN · `C2/20261119T185218Z_38ad8701/`
- C3 comparison · `C3/20261119T061308Z_4adf9ebf/`
- C4 ResNet-layer4 · `C4/20261120T034908Z_9bfaa9fb/`
- C4 CNN · `C4/20261120T044952Z_e9b5ffc8/`

See [Classifier experiments C1–C4](CLASSIFIER_EXPERIMENTS.md) for training commands, C4's balanced-sampling design, and the output layout.
