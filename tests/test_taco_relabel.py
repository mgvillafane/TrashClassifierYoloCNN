from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

import taco_yolo as detector
import sync_taco_databricks as cloud_sync
from taco_label_preview import preview, verify_relabel


class TacoRelabelTests(unittest.TestCase):
    def make_dataset(self, root):
        source = root / "raw"
        categories = [{"id": 71, "name": "Drink can"}, {"id": 95, "name": "Pop tab"},
                      {"id": 120, "name": "Foam cup"}, {"id": 300, "name": "Unlabeled litter"}]
        images, annotations = [], []
        for index in range(3):
            image = source / f"batch_{index}/sample.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60), "white").save(image)
            images.append({"id": index, "file_name": f"batch_{index}/sample.jpg", "width": 80, "height": 60})
            for category in categories:
                annotations.append({"id": len(annotations), "image_id": index,
                                    "category_id": category["id"], "bbox": [5, 10, 20, 30]})
        detector.write_json(source / "annotations.json", {
            "images": images, "annotations": annotations, "categories": categories})
        return source

    def test_sparse_taco_categories_merge_without_changing_coordinates_or_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_dataset(root)
            original = (source / "annotations.json").read_bytes()
            detector.prepare(source, root / "old")
            detector.prepare(source, root / "new", label_scheme="trashnet")
            summary = verify_relabel(root / "old", root / "new")
            self.assertEqual(summary["images"], 3)
            self.assertEqual(summary["objects"], 12)
            self.assertEqual(summary["class_counts"], {"metal": 6, "plastic": 3, "trash": 3})
            manifest = json.loads((root / "new/split_manifest.json").read_text())
            self.assertEqual(manifest["class_names"], {
                "0": "cardboard", "1": "glass", "2": "metal", "3": "paper", "4": "plastic", "5": "trash"})
            self.assertEqual((source / "annotations.json").read_bytes(), original)
            for row in manifest["images"]:
                self.assertEqual(row["class_ids"], [2, 2, 4, 5])

    def test_unknown_category_fails_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_dataset(root)
            annotation = json.loads((source / "annotations.json").read_text())
            annotation["categories"][0]["name"] = "Unknown new material"
            detector.write_json(source / "annotations.json", annotation)
            with self.assertRaisesRegex(ValueError, "explicit TrashNet mappings"):
                detector.prepare(source, root / "new", label_scheme="trashnet")
            self.assertFalse((root / "new").exists())

    def test_image_cache_preserves_image_bytes_and_rejects_different_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_dataset(root)
            detector.prepare(source, root / "old")
            with patch.object(detector.Image, "open", side_effect=AssertionError("Images should be copied")):
                detector.prepare(source, root / "new", label_scheme="trashnet", image_cache=root / "old")
            self.assertEqual(verify_relabel(root / "old", root / "new")["objects"], 12)
            manifest = json.loads((root / "new/split_manifest.json").read_text())
            for record in manifest["images"]:
                self.assertEqual((root / "old" / record["image"]).read_bytes(),
                                 (root / "new" / record["image"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "matching annotations"):
                detector.prepare(source, root / "different", label_scheme="trashnet",
                                 max_side=640, image_cache=root / "old")

    def test_policy_changes_invalidate_cache_and_reuse_preserves_mapping_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_dataset(root)
            detector.prepare(source, root / "new", label_scheme="trashnet")
            report = (root / "new/label_mapping.json").read_bytes()
            with patch.object(detector.Image, "open", side_effect=AssertionError("Images should be reused")):
                detector.prepare(source, root / "new", label_scheme="trashnet")
            self.assertEqual((root / "new/label_mapping.json").read_bytes(), report)
            with self.assertRaisesRegex(ValueError, "different settings"):
                detector.prepare(source, root / "new", label_scheme="taco")
            policy = json.loads(detector.TRASHNET_MAPPING.read_text())
            policy["groups"]["trash"].remove("Unlabeled litter")
            policy["groups"]["plastic"].append("Unlabeled litter")
            detector.write_json(root / "changed_mapping.json", policy)
            with patch.object(detector, "TRASHNET_MAPPING", root / "changed_mapping.json"):
                with self.assertRaisesRegex(ValueError, "mapping_sha256: saved="):
                    detector.prepare(source, root / "new", label_scheme="trashnet")

    def test_verifier_detects_box_corruption_and_previews_show_real_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_dataset(root)
            detector.prepare(source, root / "old")
            detector.prepare(source, root / "new", label_scheme="trashnet")
            paths = preview(root / "old", root / "new", root / "preview", image_id=0)
            for path in paths.values():
                with Image.open(path) as image:
                    image.verify()
            transitions = json.loads((root / "preview/image_000000_transitions.json").read_text())
            self.assertEqual(transitions[0], {"object": 1, "before": "Drink can", "after": "metal"})
            manifest = json.loads((root / "new/split_manifest.json").read_text())
            label = root / "new" / manifest["images"][0]["label"]
            label.write_text(label.read_text().replace("0.18750000", "0.20000000", 1))
            with self.assertRaisesRegex(ValueError, "coordinates changed"):
                verify_relabel(root / "old", root / "new")

    def test_sixty_categories_are_explicit_and_old_checkpoints_are_rejected(self):
        policy = json.loads(detector.TRASHNET_MAPPING.read_text())
        categories = [name for group in policy["groups"].values() for name in group]
        self.assertEqual(len(categories), 60)
        self.assertEqual(len(set(categories)), 60)
        names, _, _ = detector.dataset_classes(
            [{"id": index, "name": name} for index, name in enumerate(categories)], "trashnet")
        with self.assertRaisesRegex(ValueError, "same label scheme"):
            detector.check_class_names({0: "Aluminium foil"}, {"class_names": names})

    def make_sync_dataset(self, root):
        source = self.make_dataset(root)
        target = root / "data/raw/taco"
        target.parent.mkdir(parents=True)
        source.rename(target)
        detector.prepare(target, root / "data/yolo/taco")
        detector.prepare(target, root / "data/yolo/taco_trashnet", label_scheme="trashnet")

    def test_cloud_sync_uses_volume_yaml_verifies_metadata_and_deploys_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sync_dataset(root)

            def command_result(command, **kwargs):
                if command[1:3] == ["fs", "cat"]:
                    path = (root / "data/yolo/taco_trashnet/split_manifest.json"
                            if command[3].endswith("split_manifest.json")
                            else root / "reports/taco_relabel/databricks_dataset.yaml")
                    return SimpleNamespace(stdout=path.read_text())
                return SimpleNamespace(stdout="")

            with patch.object(cloud_sync, "ROOT", root), patch.object(cloud_sync, "find_cli", return_value="databricks"), \
                    patch.object(cloud_sync.subprocess, "run", side_effect=command_result) as execute, \
                    redirect_stdout(io.StringIO()):
                cloud_sync.main(["--volume", "/Volumes/workspace/default/test_volume"])
            commands = [call.args[0] for call in execute.call_args_list]
            self.assertEqual(commands[-1][1:3], ["bundle", "deploy"])
            self.assertFalse(any(command[1:3] == ["bundle", "run"] for command in commands))
            staged = (root / "reports/taco_relabel/databricks_dataset.yaml").read_text()
            self.assertIn('path: "/Volumes/workspace/default/test_volume/yolo/taco_trashnet"', staged)
            self.assertNotIn(str(root), staged)
            result = json.loads((root / "reports/taco_relabel/cloud_sync.json").read_text())
            self.assertTrue(result["metadata_verified"])
            self.assertFalse(result["training_started"])

    def test_cloud_sync_dry_run_never_calls_databricks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sync_dataset(root)
            with patch.object(cloud_sync, "ROOT", root), patch.object(cloud_sync, "find_cli", return_value="databricks"), \
                    patch.object(cloud_sync.subprocess, "run") as execute, redirect_stdout(io.StringIO()):
                cloud_sync.main(["--dry-run"])
            execute.assert_not_called()
            self.assertFalse((root / "reports/taco_relabel/cloud_sync.json").exists())


if __name__ == "__main__":
    unittest.main()
