import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import dataset_tools as tools


class DatasetTests(unittest.TestCase):
    def test_sampling_covers_classes_and_is_repeatable(self):
        items = [f'{label}/{i}.jpg' for label in ('a', 'b', 'c') for i in range(10)]
        key = lambda item: item.split('/')[0]
        first = tools.balanced_sample(items, key, 2, 42)
        self.assertEqual(first, tools.balanced_sample(list(reversed(items)), key, 2, 42))
        self.assertEqual({key(item) for item in first}, {'a', 'b', 'c'})
        self.assertEqual(len(first), 6)

    def test_scene_selection_covers_rare_labels(self):
        coco = {
            'images': [{'id': i} for i in range(4)],
            'annotations': [
                {'image_id': 0, 'category_id': 7},
                {'image_id': 1, 'category_id': 7},
                {'image_id': 2, 'category_id': 7},
                {'image_id': 3, 'category_id': 91},
            ],
        }
        selected = tools.diverse_coco_sample(coco, 2, 42)
        self.assertIn(3, [image['id'] for image in selected])
        self.assertEqual(selected, tools.diverse_coco_sample(coco, 2, 42))

    def test_coco_preserves_ids_canvas_and_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (100, 80)).save(root / 'actual.jpg')
            coco = {
                'categories': [{'id': 37, 'name': 'PET', 'supercategory': 'Plastic'}],
                'images': [
                    {'id': 91, 'file_name': 'actual.jpg', 'width': 200, 'height': 160},
                    {'id': 92, 'file_name': 'missing.jpg', 'width': 200, 'height': 160},
                ],
                'annotations': [{'id': 5, 'image_id': 91, 'category_id': 37, 'bbox': [20, 30, 40, 50]}],
            }
            rows, missing = tools.coco_rows(coco, coco['images'], root, 'sortwaste', 'train')
            self.assertEqual(missing, ['missing.jpg'])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['width'], 100)
            self.assertEqual(rows[0]['annotation_width'], 200)
            self.assertEqual(rows[0]['annotations'][0]['bbox_xywh'], [20, 30, 40, 50])
            self.assertEqual(rows[0]['annotations'][0]['label'], 'PET')
            self.assertEqual(rows[0]['annotations'][0]['category_id'], 37)

    def test_image_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                tools.safe_child(directory, '../outside.jpg')

    def test_import_preserves_other_splits_and_source_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (100, 80)).save(root / 'actual.jpg')
            coco = {'categories': [], 'annotations': [], 'images': [
                {'id': 1, 'file_name': 'actual.jpg', 'width': 100, 'height': 80}
            ]}
            annotation_path = root / 'original.json'
            original = json.dumps(coco)
            annotation_path.write_text(original, encoding='utf-8')
            for split in ('train', 'val', 'train'):
                tools.import_coco(annotation_path, root, 'sortwaste', split, root / 'data')
            rows = tools.read_rows(root / 'data/processed/manifest.jsonl')
            self.assertEqual(len(rows), 2)
            self.assertEqual({row['split'] for row in rows}, {'train', 'val'})
            self.assertEqual(annotation_path.read_text(encoding='utf-8'), original)

    def test_import_limit_selects_first_ten_source_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The eleventh file deliberately does not exist: it must not be checked.
            ids = [21, 3, 40, 8, 19, 2, 14, 6, 30, 1, 99]
            for image_id in ids[:10]:
                Image.new('RGB', (20, 10)).save(root / f'{image_id}.jpg')
            coco = {'categories': [], 'annotations': [], 'images': [
                {'id': image_id, 'file_name': f'{image_id}.jpg', 'width': 20, 'height': 10}
                for image_id in ids
            ]}
            annotation_path = root / 'original.json'
            annotation_path.write_text(json.dumps(coco), encoding='utf-8')
            rows = tools.import_coco(annotation_path, root, 'sortwaste', 'train', root / 'data', limit=10)
            self.assertEqual([row['image_id'] for row in rows], ids[:10])
            self.assertEqual(len(tools.read_rows(root / 'data/processed/manifest.jsonl')), 10)
            metadata = json.loads((root / 'data/raw/sortwaste/import_train.json').read_text())
            self.assertEqual(metadata['requested_images'], 10)
            self.assertEqual(metadata['total_source_images'], 11)
            self.assertEqual(metadata['missing_images'], [])

    def test_import_rejects_nonpositive_limit(self):
        with self.assertRaisesRegex(ValueError, 'limit must be positive'):
            tools.import_coco('unused.json', '.', 'sortwaste', 'train', Path('.'), limit=0)


if __name__ == '__main__':
    unittest.main()
