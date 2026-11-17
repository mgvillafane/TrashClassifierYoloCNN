import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing import (DataPaths, build_manifest, check_raw_files, load_config,  # noqa: E402
                                validate_manifest)
from data_preprocessing.crop import pixel_box  # noqa: E402
from data_preprocessing.datasets import CONFLICT_REASON, list_images, whole_image_rows  # noqa: E402
from data_preprocessing.datasets import sortwaste  # noqa: E402
from data_preprocessing.datasets.sortwaste import parse_yolo_line  # noqa: E402
from data_preprocessing.datasets.taco import coco_to_yolo_box  # noqa: E402
from data_preprocessing.export import (classifier_index, dataset_yamls, export_classifier,  # noqa: E402
                                       export_detector, label_path, yolo_class_id, yolo_class_names)
from data_preprocessing.labels import apply_mapping, build_rules  # noqa: E402
from data_preprocessing.splits import assign_whole_image_splits, make_taco_folds  # noqa: E402

CONFIG = load_config(ROOT / "configs/label_mapping.json")
RULES = build_rules(CONFIG)
HAS_RAW_DATA = (ROOT / "data/raw/taco/data/annotations.json").exists()


def box_row(dataset, label, w=0.2, h=0.2, img=(1000, 1000)):
    return {"dataset": dataset, "example_type": "bounding_box", "original_label": label,
            "bbox_cx": 0.5, "bbox_cy": 0.5, "bbox_w": w, "bbox_h": h, "img_w": img[0], "img_h": img[1]}


def image(path, color="white", size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class LabelMappingTests(unittest.TestCase):
    def test_sortwaste_ids_are_list_positions_not_name_prefixes(self):
        names = {int(k): v["name"] for k, v in CONFIG["sortwaste"].items()}
        self.assertEqual(names[3], "ecal")
        self.assertEqual(names[4], "metal")
        self.assertEqual(names[7], "pet_oleo")
        commons = {int(k): v["common"] for k, v in CONFIG["sortwaste"].items()}
        self.assertEqual(commons, {0: "plastic", 1: "plastic", 2: "plastic", 3: "cardboard",
                                   4: "metal", 5: "cardboard", 6: "plastic", 7: "plastic"})

    def test_realwaste_and_trashnet_map_into_common_schema(self):
        self.assertEqual(set(CONFIG["realwaste"]), {"Cardboard", "Food Organics", "Glass", "Metal",
                                                    "Miscellaneous Trash", "Paper", "Plastic",
                                                    "Textile Trash", "Vegetation"})
        self.assertTrue(all(name == rule["common"] for name, rule in CONFIG["trashnet"].items()))
        for dataset in ("realwaste", "trashnet", "sortwaste", "taco"):
            for rule in RULES[dataset].values():
                self.assertIn(rule["common"], CONFIG["common_labels"])

    def test_ambiguous_labels_stay_for_detector_but_leave_classifier(self):
        cases = [("sortwaste", "ecal"), ("taco", "Drink carton"), ("taco", "Unlabeled litter"),
                 ("taco", "Food waste"), ("taco", "Shoe"), ("taco", "Rope & strings")]
        for dataset, label in cases:
            with self.subTest(label=label):
                row = apply_mapping(box_row(dataset, label), RULES, CONFIG)
                self.assertTrue(row["use_detector_train"] and row["use_detector_eval"])
                self.assertFalse(row["use_classifier_train"] or row["use_classifier_eval"])
                self.assertTrue(row["exclude_reason"])

    def test_cartons_share_one_common_label_across_datasets(self):
        self.assertEqual(RULES["sortwaste"]["ecal"]["common"], "cardboard")
        self.assertEqual(RULES["taco"]["Drink carton"]["common"], "cardboard")

    def test_realwaste_organics_and_textile_are_excluded(self):
        for label in ("Food Organics", "Vegetation", "Textile Trash"):
            row = apply_mapping({"dataset": "realwaste", "example_type": "whole_image",
                                 "original_label": label}, RULES, CONFIG)
            self.assertFalse(row["use_classifier_eval"])
            self.assertFalse(row["use_detector_train"])

    def test_small_crops_are_excluded_from_classifier_only(self):
        small = apply_mapping(box_row("taco", "Cigarette", w=0.03, h=0.2), RULES, CONFIG)  # 30 px wide
        self.assertIn("32px", small["exclude_reason"])
        self.assertTrue(small["use_detector_train"])
        large = apply_mapping(box_row("taco", "Cigarette", w=0.04, h=0.2), RULES, CONFIG)  # 40 px wide
        self.assertTrue(large["use_classifier_eval"])

    def test_taco_is_never_classifier_training_data(self):
        row = apply_mapping(box_row("taco", "Glass bottle"), RULES, CONFIG)
        self.assertEqual(row["common_label"], "glass")
        self.assertTrue(row["use_classifier_eval"])
        self.assertFalse(row["use_classifier_train"])

    def test_unknown_labels_fail_instead_of_becoming_trash(self):
        with self.assertRaises(KeyError):
            apply_mapping(box_row("taco", "Mystery object"), RULES, CONFIG)

    @unittest.skipUnless(HAS_RAW_DATA, "raw TACO annotations not present")
    def test_every_taco_category_has_a_rule(self):
        coco = json.loads((ROOT / "data/raw/taco/data/annotations.json").read_text(encoding="utf-8"))
        self.assertEqual({c["name"] for c in coco["categories"]} - set(RULES["taco"]), set())


class ParsingTests(unittest.TestCase):
    def test_yolo_line(self):
        self.assertEqual(parse_yolo_line("3 0.5 0.25 0.1 0.2", 8), (3, 0.5, 0.25, 0.1, 0.2))
        for bad in ["3 0.5 0.25 0.1", "3 0.5 0.25 0.1 0.2 0.9", "8 0.5 0.5 0.1 0.1",
                    "1 1.2 0.5 0.1 0.1", "1 0.5 0.5 0 0.1", "1 -0.1 0.5 0.1 0.1"]:
            with self.subTest(line=bad), self.assertRaises(ValueError):
                parse_yolo_line(bad, 8)

    def test_coco_box_normalization_and_clipping(self):
        self.assertEqual(coco_to_yolo_box([10, 20, 30, 40], 100, 200), (0.25, 0.2, 0.3, 0.2, False))
        cx, cy, w, h, clipped = coco_to_yolo_box([-10, 0, 30, 50], 100, 100)
        self.assertTrue(clipped)
        self.assertAlmostEqual(w, 0.2)
        self.assertAlmostEqual(cx, 0.1)
        for bad in ([0, 0, 0, 10], [200, 0, 10, 10], [0, 0, float("nan"), 10]):
            with self.subTest(box=bad), self.assertRaises(ValueError):
                coco_to_yolo_box(bad, 100, 100)

    def test_pixel_box_pads_and_clamps(self):
        self.assertEqual(pixel_box(0.5, 0.5, 0.2, 0.4, 100, 50), (40, 15, 60, 35))
        self.assertEqual(pixel_box(0.5, 0.5, 0.2, 0.4, 100, 50, padding=0.5), (30, 5, 70, 45))
        self.assertEqual(pixel_box(0.05, 0.5, 0.2, 0.2, 100, 100), (0, 40, 15, 60))
        with self.assertRaises(ValueError):
            pixel_box(0.5, 0.5, 0.001, 0.2, 100, 100)


class FileDiscoveryTests(unittest.TestCase):
    def test_list_images_skips_macos_metadata_and_non_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "glass/a.jpg")
            image(root / "glass/b.PNG")
            image(root / "glass/._a.jpg")
            image(root / "__MACOSX/glass/c.jpg")
            (root / "glass/notes.txt").write_text("x")
            (root / "glass/.DS_Store").write_text("x")
            self.assertEqual([p.name for p in list_images(root)], ["a.jpg", "b.PNG"])

    def test_identical_images_in_two_classes_are_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "glass/g1.png", "red")
            image(root / "metal/m1.png", "red")      # same bytes, different class
            image(root / "metal/m2.png", "blue")
            rows = whole_image_rows("trashnet", root, lambda p: p.relative_to(root).as_posix())
            flagged = {r["image_path"] for r in rows if r.get("exclude_reason") == CONFLICT_REASON}
            self.assertEqual(flagged, {"glass/g1.png", "metal/m1.png"})

    def test_sortwaste_pairs_yolo_labels_with_coco_copy_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = DataPaths(root)
            for split in sortwaste.SPLITS:
                image(paths.sortwaste_images / split / "images" / f"{split}_1.png")
                annotations = paths.sortwaste_images / split / "annotations/coco.json"
                annotations.parent.mkdir(parents=True)
                annotations.write_text(json.dumps({"images": [
                    {"file_name": f"{split}_1.png", "width": 16, "height": 12}]}))
                label = paths.sortwaste_labels / split / "labels" / f"{split}_1.txt"
                label.parent.mkdir(parents=True)
                label.write_text("3 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.1 0.1\n" if split == "train" else "")
            names = {int(k): v["name"] for k, v in CONFIG["sortwaste"].items()}
            rows = sortwaste.rows(paths, names)
            self.assertEqual([r["original_label"] for r in rows if r["split"] == "train"], ["ecal", "metal"])
            self.assertEqual([r["example_type"] for r in rows if r["split"] != "train"], ["no_object"] * 2)
            self.assertTrue(rows[0]["image_path"].startswith("data/raw/sortwaste/dataset/splited_all_dataset/"))

            (paths.sortwaste_labels / "val/labels/orphan.txt").write_text("")
            with self.assertRaisesRegex(ValueError, "labels without image"):
                sortwaste.rows(paths, names)


class SplitTests(unittest.TestCase):
    def frame(self):
        rows = [{"dataset": "trashnet", "original_label": label, "image_path": f"{label}/{i}.jpg",
                 "content_md5": f"{label}-{i}"} for label, n in [("glass", 100), ("trash", 37)] for i in range(n)]
        rows.append({"dataset": "trashnet", "original_label": "glass", "image_path": "glass/dup.jpg",
                     "content_md5": "glass-0"})  # byte-identical to glass/0.jpg
        return pd.DataFrame(rows)

    def test_split_is_deterministic_stratified_and_order_independent(self):
        df = self.frame()
        ratios = CONFIG["splits"]["whole_image_ratios"]
        first = assign_whole_image_splits(df, 42, ratios)
        shuffled = df.sample(frac=1, random_state=7)
        self.assertTrue(first.equals(assign_whole_image_splits(shuffled, 42, ratios).sort_index()))
        self.assertFalse(first.equals(assign_whole_image_splits(df, 43, ratios)))
        counts = pd.crosstab(df["original_label"], first)
        self.assertEqual(counts.loc["trash"].to_dict(), {"test": 4, "train": 30, "val": 3})
        self.assertEqual(first[df.index[df["image_path"] == "glass/dup.jpg"][0]], first[0])

    def test_ratios_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            assign_whole_image_splits(self.frame(), 42, {"train": 0.8, "val": 0.1, "test": 0.2})

    def test_taco_folds_are_equal_sized_deterministic_and_balanced(self):
        counts = pd.DataFrame({"images": [100] * 6, "glass": [9, 1, 1, 9, 5, 5]}, index=range(1, 7))
        folds, score = make_taco_folds(counts, n_folds=3)
        self.assertEqual(folds, make_taco_folds(counts, n_folds=3)[0])
        self.assertEqual(sorted(pd.Series(folds).value_counts()), [2, 2, 2])
        self.assertEqual(score, 0.0)  # {9,1}, {1,9}, {5,5}
        with self.assertRaises(ValueError):
            make_taco_folds(counts.iloc[:5], n_folds=3)


def export_frame():
    """Two whole images and one rotated TACO photo with two boxes (one detector-only)."""
    base = {"split": None, "taco_batch": pd.NA, "taco_fold": pd.NA, "taco_annotation_id": pd.NA,
            "source_image_id": None, "original_label": None, "bbox_cx": None, "bbox_cy": None,
            "bbox_w": None, "bbox_h": None, "use_detector_train": False, "use_classifier_train": True}
    rows = [
        {**base, "dataset": "trashnet", "split": "train", "example_type": "whole_image",
         "image_path": "raw/trashnet/glass/glass1.jpg", "img_w": 16, "img_h": 12,
         "common_label": "glass", "common_class_id": 1, "use_classifier_eval": True},
        {**base, "dataset": "trashnet", "split": "val", "example_type": "whole_image",
         "image_path": "raw/trashnet/glass/glass2.jpg", "img_w": 16, "img_h": 12,
         "common_label": "glass", "common_class_id": 1, "use_classifier_eval": False},
    ]
    taco = {**base, "dataset": "taco", "taco_batch": 7, "taco_fold": 0, "example_type": "bounding_box",
            "image_path": "raw/taco/batch_7/000001.jpg", "img_w": 40, "img_h": 80,
            "use_detector_train": True, "use_classifier_train": False}
    rows += [
        {**taco, "bbox_cx": 0.25, "bbox_cy": 0.25, "bbox_w": 0.5, "bbox_h": 0.5, "taco_annotation_id": 1,
         "common_label": "metal", "common_class_id": 2, "use_classifier_eval": True},
        {**taco, "bbox_cx": 0.75, "bbox_cy": 0.75, "bbox_w": 0.1, "bbox_h": 0.1, "taco_annotation_id": 2,
         "common_label": "trash", "common_class_id": 5, "use_classifier_eval": False},
    ]
    return pd.DataFrame(rows)


class ExportTests(unittest.TestCase):
    def test_label_ids_follow_common_label_order(self):
        for i, label in enumerate(CONFIG["common_labels"]):
            row = apply_mapping({"dataset": "realwaste", "example_type": "whole_image",
                                 "original_label": label.capitalize() if label != "trash" else "Miscellaneous Trash"},
                                RULES, CONFIG)
            if row["use_classifier_eval"]:
                self.assertEqual(row["common_class_id"], i)

    def test_classifier_index_paths_and_partitions(self):
        index = classifier_index(export_frame())
        self.assertEqual(index["path"].tolist(), ["trashnet/train/glass/glass1.jpg",
                                                  "taco/fold0/metal/batch7_000001_000.jpg"])
        self.assertEqual(index["label_id"].tolist(), [1, 2])

    def test_data_can_live_outside_the_repo_with_the_same_manifest_paths(self):
        paths = DataPaths(Path("/repo"), raw_dir=Path("/Volumes/w/d/waste_data"),
                          taco_dir=Path("/Volumes/w/d/tacotrashdata/taco"))
        for real, logical in [
            ("/Volumes/w/d/waste_data/trashnet/dataset-resized/glass/glass1.jpg",
             "data/raw/trashnet/dataset-resized/glass/glass1.jpg"),
            ("/Volumes/w/d/tacotrashdata/taco/batch_7/000036.JPG", "data/raw/taco/data/batch_7/000036.JPG"),
        ]:
            self.assertEqual(paths.rel(Path(real)), logical)
            self.assertEqual(paths.resolve(logical), Path(real))
        local = DataPaths(Path("/repo"))
        self.assertEqual(local.rel(local.taco / "batch_1/x.jpg"), "data/raw/taco/data/batch_1/x.jpg")
        self.assertEqual(local.resolve("data/raw/taco/data/batch_1/x.jpg"), Path("/repo/data/raw/taco/data/batch_1/x.jpg"))

    def test_raw_file_check_reports_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            check = check_raw_files(DataPaths(Path(tmp)))
            self.assertFalse(check["ok"].any())
            self.assertEqual(check["found"].sum(), 0)

    def test_label_path_mirrors_images_folder(self):
        self.assertEqual(label_path("images/taco/batch7/batch7_000001.jpg"), "labels/taco/batch7/batch7_000001.txt")

    def test_export_writes_upright_crops_and_single_class_yolo_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "repo", Path(tmp) / "out"
            image(root / "raw/trashnet/glass/glass1.jpg")
            image(root / "raw/trashnet/glass/glass2.jpg")
            # stored landscape 80x40 with EXIF "rotate 90" -> upright 40x80, as the manifest says
            photo = Image.new("RGB", (80, 40), "red")
            exif = photo.getexif()
            exif[0x0112] = 6
            (root / "raw/taco/batch_7").mkdir(parents=True)
            photo.save(root / "raw/taco/batch_7/000001.jpg", exif=exif)

            df = export_frame()
            export_classifier(df, DataPaths(root), out / "classifier")
            with Image.open(out / "classifier/taco/fold0/metal/batch7_000001_000.jpg") as crop:
                self.assertEqual(crop.size, (20, 40))
            self.assertFalse((out / "classifier/trashnet/val").exists())
            self.assertEqual(len(pd.read_csv(out / "classifier/index.csv")), 2)

            export_detector(df, DataPaths(root), out / "detector", n_folds=3)
            det = out / "detector"
            self.assertEqual((det / "taco_fold0.txt").read_text(), "./images/taco/batch7/batch7_000001.jpg\n")
            lines = (det / "labels/taco/batch7/batch7_000001.txt").read_text().splitlines()
            self.assertEqual([line.split()[0] for line in lines], ["0", "0"])  # detector-only box kept
            with Image.open(det / "images/taco/batch7/batch7_000001.jpg") as upright:
                self.assertEqual(upright.size, (40, 80))
            self.assertFalse((det / "d2_taco_fold0.yaml").exists())  # no SortWaste lists in this frame
            self.assertIn("train: [sortwaste_train.txt, taco_fold1.txt, taco_fold2.txt]",
                          dataset_yamls(3)["d2_taco_fold0.yaml"])
            self.assertFalse(any(det.rglob("*glass*")))

            export_detector(df, DataPaths(root), out / "yolo_classes", n_folds=3,
                            class_names=yolo_class_names(CONFIG["common_labels"]))
            lines = (out / "yolo_classes/labels/taco/batch7/batch7_000001.txt").read_text().splitlines()
            self.assertEqual([line.split()[0] for line in lines], ["2", "5"])  # metal, trash

    def test_beverage_cartons_get_their_own_yolo_class(self):
        names = yolo_class_names(CONFIG["common_labels"])
        self.assertEqual(names[-1], "carton")
        carton, cardboard = len(names) - 1, CONFIG["common_labels"].index("cardboard")
        for dataset, label in [("sortwaste", "ecal"), ("taco", "Drink carton")]:
            box = pd.Series({"dataset": dataset, "original_label": label, "common_class_id": cardboard})
            self.assertEqual(yolo_class_id(box, carton), carton)
        box = pd.Series({"dataset": "taco", "original_label": "Pizza box", "common_class_id": cardboard})
        self.assertEqual(yolo_class_id(box, carton), cardboard)
        yaml = dataset_yamls(3, "e", names)["e1_sortwaste.yaml"]
        self.assertIn("  6: carton", yaml)


@unittest.skipUnless(HAS_RAW_DATA, "raw datasets not present")
class RealDataTests(unittest.TestCase):
    def test_manifest_builds_and_validates(self):
        paths = DataPaths(ROOT)
        manifest = build_manifest(paths, CONFIG)
        validate_manifest(manifest, CONFIG)
        self.assertEqual(manifest.loc[manifest["dataset"] == "taco", "original_label"].eq("Cigarette").sum(), 667)


if __name__ == "__main__":
    unittest.main()
