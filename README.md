# TrashClassifierYoloCNN

Waste sorting with a detector + material classifier, compared against end-to-end YOLO, across four
datasets. YOLO-on-TACO baseline notes: [TACO_YOLO_GUIDE.md](TACO_YOLO_GUIDE.md).

## Data

All four datasets are unified into one manifest (99,592 rows: one per whole image or bounding box)
under six classes: **cardboard, glass, metal, paper, plastic, trash** (`label_id` 0–5 in that order).

| Dataset | Type | Images | Boxes | Classes | Classifier examples | Split |
|---|---|---:|---:|---|---:|---|
| RealWaste | whole images | 4,752 | – | 6 of 6 | 3,587 | stratified 80/10/10 (seed 42) |
| TrashNet | whole images | 2,527 | – | 6 of 6 | 2,521 | stratified 80/10/10 (seed 42) |
| SortWaste | conveyor belt, boxes | 5,261 | 87,525 | 3 of 6 | 68,298 | official 3,705 / 780 / 776 images |
| TACO | outdoor litter, boxes | 1,500 | 4,784 | 6 of 6 | 3,627 | 3 folds of 5 batches, **test only** |

Classifier examples per class:

| | cardboard | glass | metal | paper | plastic | trash |
|---|---:|---:|---:|---:|---:|---:|
| RealWaste | 461 | 420 | 790 | 500 | 921 | 495 |
| TrashNet | 403 | 498 | 409 | 594 | 480 | 137 |
| SortWaste | 2,156 | 0 | 1,437 | 0 | 64,705 | 0 |
| TACO | 205 | 204 | 497 | 202 | 2,095 | 424 |

### Rules

- **Detector:** every box is a target, as one class `waste_object`.
- **Classifier:** only examples with an unambiguous material and a crop short side of at least
  32 px. Excluded (they stay in the manifest with `exclude_reason`):
  - beverage cartons: SortWaste `ecal` (19,227 boxes), TACO `Drink carton` (45)
  - TACO `Unlabeled litter` (517), food waste, shoes, rope
  - RealWaste `Vegetation`, `Food Organics`, `Textile Trash` (1,165 images)
  - 3 TrashNet images stored byte-identically under two classes (6 files)
  - crops under 32 px: 551 TACO boxes (340 of them cigarettes), none in SortWaste
- **TACO is never used to train the classifier.** It is the external test set; look at TACO results
  once, after every setting is frozen. Model selection uses `val` splits only.
- **Trash** on TACO is reported with and without cigarettes (77% of TACO trash crops).
- **End-to-end YOLO** keeps beverage cartons as an extra `carton` class (id 6) and is compared with
  the pipelines on cardboard, metal and plastic only.

### Things to know

- SortWaste has only cardboard, metal and plastic, and 95% of its classifier crops are plastic.
  Adding it to classifier training (C4) needs class-balanced sampling or capping plastic.
- 90% of SortWaste's "cardboard" boxes are beverage cartons, hence their exclusion/own class.
- SortWaste's YOLO copy has labels for every split but images only for test; images come from its
  COCO copy (same file stems and splits). The dataset docs say 87,252 boxes; both copies contain 87,525.

### Experiments and their data

| Experiment | Train | Select on | Test | Data |
|---|---|---|---|---|
| D1 detector | SortWaste train | SortWaste val | SortWaste test, TACO fold k | `detector/d1_sortwaste.yaml` |
| D2 detector (k = 0, 1, 2) | SortWaste train + other 2 TACO folds | SortWaste val | TACO fold k | `detector/d2_taco_fold{k}.yaml` |
| E1 end-to-end YOLO | SortWaste train | SortWaste val | SortWaste test, TACO fold k | `yolo_classes/e1_sortwaste.yaml` |
| E2 end-to-end YOLO | SortWaste train + other 2 TACO folds | SortWaste val | TACO fold k | `yolo_classes/e2_taco_fold{k}.yaml` |
| C1 classifier | TrashNet `train` | `val` | `test` + all TACO | `classifier/index.csv` |
| C2 classifier | RealWaste `train` | `val` | `test` + all TACO | 〃 |
| C3 classifier | TrashNet + RealWaste `train` | `val` | `test` + all TACO | 〃 |
| C4 classifier | C3 + SortWaste `train` | `val` | `test` + all TACO | 〃 |
| P1 / P2 pipeline | D1 / D2 + best classifier | – | TACO fold k | – |

Sizes: C1 2,018 / 251 / 252, C2 2,870 / 359 / 358, C3 4,888 / 610 / 610, C4 53,081 / 11,123 / 10,202
crops (train / val / test), TACO test 3,627 crops. D1 3,705 / 780 / 776 images; D2 trains on
4,680 / 4,716 / 4,719 images for folds 0 / 1 / 2. D1 and E1 on TACO fold k:
`model.val(data="d2_taco_fold{k}.yaml" or "e2_taco_fold{k}.yaml", split="test")`, the same images
as D2 and E2.

### Building the data

**On Databricks:** open `notebooks/databricks_prepare_data.py` (Git folder on this branch) and run
all cells on CPU compute. It reads the raw data in the volumes (read-only), checks every file count,
builds the manifest and writes everything to the shared folder
`/Volumes/workspace/default/waste_data/processed`. Finished parts carry a `_SUCCESS` marker and are
skipped on later runs, so it only needs to be built once. Use `rebuild=yes` only when the data or
the mapping changes.

**Locally** (raw data in `data/raw/`):

```bash
python scripts/build_manifest.py            # data/manifests/: manifest, summaries, figures
python scripts/export_datasets.py           # data/processed/: classifier, detector, yolo_classes
python -m unittest tests.test_data_preprocessing
```

Output layout (about 6 GB):

```
processed/
├── README.md            per-part counts, rules, a PyTorch Dataset over index.csv
├── manifests/           unified_manifest.csv, summaries
├── classifier/          index.csv + <dataset>/<train|val|test|foldN>/<label>/*.jpg (longest side ≤ 512 px)
├── detector/            YOLO, one class: images/, labels/, image lists, d1/d2 YAMLs (longest side ≤ 1280 px)
└── yolo_classes/        same images, 7 classes (6 + carton): e1/e2 YAMLs
```

Use `index.csv` and its `label_id` for the classifier, not torchvision `ImageFolder`: SortWaste has
no glass/paper/trash folders, so `ImageFolder` would number its classes differently.

### Where things live

- `configs/label_mapping.json`: every original label → common class, with exclusion reasons
- `configs/taco_folds.csv`: fixed TACO batch → fold assignment (do not regenerate once results exist)
- `src/data_preprocessing/`: readers, label mapping, splits, manifest checks, export
- `notebooks/label_preprocessing.ipynb`: dataset analysis behind these decisions
- `data/manifests/`: committed summaries (`summary.csv`, `class_table.csv`, `exclusions.csv`) and figures
