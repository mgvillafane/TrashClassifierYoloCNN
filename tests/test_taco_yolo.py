import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import taco_yolo as detector


class TacoYoloTests(unittest.TestCase):
    def test_box_conversion_and_boundary_clipping(self):
        self.assertEqual(detector.normalized_box([20, 10, 40, 20], 100, 50), (0.4, 0.4, 0.4, 0.4))
        self.assertEqual(detector.normalized_box([-10, 0, 30, 50], 100, 50), (0.1, 0.5, 0.2, 1.0))
        for box in ([0, 0, -1, 5], [200, 0, 10, 10], [0, 0, float('nan'), 10]):
            with self.assertRaises(ValueError):
                detector.normalized_box(box, 100, 50)

    def test_source_batches_never_cross_splits(self):
        images = [{'id': i * 2 + j, 'file_name': f'batch_{i}/{j}.jpg'} for i in range(15) for j in range(2)]
        first = detector.split_by_batch(images)
        self.assertEqual(first, detector.split_by_batch(list(reversed(images))))
        self.assertEqual(set(first.values()), {'train', 'val', 'test'})
        for i in range(15):
            self.assertEqual(first[i * 2], first[i * 2 + 1])

    def test_preparation_orients_copies_and_remaps_sparse_category_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'raw', Path(directory) / 'prepared'
            images, annotations = [], []
            for i in range(3):
                path = source / f'batch_{i}/sample.jpg'
                path.parent.mkdir(parents=True)
                exif = Image.Exif()
                exif[274] = 6
                Image.new('RGB', (80, 40), 'red').save(path, exif=exif)
                images.append({'id': i, 'file_name': f'batch_{i}/sample.jpg', 'width': 40, 'height': 80})
                annotations.append({'id': i, 'image_id': i, 'category_id': 37, 'bbox': [10, 20, 20, 40]})
            original = json.dumps({'images': images, 'annotations': annotations, 'categories': [{'id': 37, 'name': 'Bottle'}]})
            (source / 'annotations.json').write_text(original)
            digest = hashlib.sha256((source / images[0]['file_name']).read_bytes()).hexdigest()
            config = detector.prepare(source, output, max_side=40)
            self.assertTrue(config.is_file())
            manifest = json.loads((output / 'split_manifest.json').read_text())
            self.assertEqual(manifest['image_counts'], {'test': 1, 'val': 1, 'train': 1})
            for row in manifest['images']:
                with Image.open(output / row['image']) as prepared:
                    self.assertEqual(prepared.size, (20, 40))
                    self.assertNotIn(274, prepared.getexif())
                label = (output / row['label']).read_text().split()
                self.assertEqual([float(x) for x in label], [0, .5, .5, .5, .5])
            self.assertEqual((source / 'annotations.json').read_text(), original)
            self.assertEqual(hashlib.sha256((source / images[0]['file_name']).read_bytes()).hexdigest(), digest)

    def test_detection_matching_requires_class_and_unique_target(self):
        targets = [{'class_id': 1, 'xyxy': [0, 0, 10, 10]}]
        predictions = [
            {'class_id': 1, 'xyxy': [0, 0, 10, 10], 'confidence': .8},
            {'class_id': 1, 'xyxy': [0, 0, 10, 10], 'confidence': .7},
            {'class_id': 2, 'xyxy': [0, 0, 10, 10], 'confidence': .9},
        ]
        _, counts = detector.match_detections(predictions, targets)
        self.assertEqual(counts['true_positives'], 1)
        self.assertEqual(counts['false_positives'], 2)
        self.assertEqual(counts['false_negatives'], 0)
        self.assertEqual(detector.intersection_over_union([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_pretrained_coco_classes_cannot_be_scored_as_taco_classes(self):
        manifest = {'class_names': {'0': 'Drink can'}}
        detector.check_class_names({0: 'Drink can'}, manifest)
        with self.assertRaisesRegex(ValueError, 'TACO-trained'):
            detector.check_class_names({0: 'person'}, manifest)


if __name__ == '__main__':
    unittest.main()
