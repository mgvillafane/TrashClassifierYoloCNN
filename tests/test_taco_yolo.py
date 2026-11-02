import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

import taco_yolo as detector


class TacoYoloTests(unittest.TestCase):
    def make_raw_dataset(self, root):
        source = root / 'raw'
        images, annotations = [], []
        for i in range(3):
            path = source / f'batch_{i}/sample.jpg'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (80, 40), 'red').save(path)
            images.append({'id': i, 'file_name': f'batch_{i}/sample.jpg', 'width': 80, 'height': 40})
            annotations.append({'id': i, 'image_id': i, 'category_id': 37, 'bbox': [10, 10, 20, 20]})
        (source / 'annotations.json').write_text(json.dumps({
            'images': images, 'annotations': annotations,
            'categories': [{'id': 37, 'name': 'Bottle'}]}))
        return source

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

    def test_moved_prepared_data_is_reused_and_yaml_points_at_new_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = self.make_raw_dataset(root), root / 'prepared'
            detector.prepare(source, output)
            manifest = json.loads((output / 'split_manifest.json').read_text())
            # A legacy Windows manifest must also be reusable on Linux.
            manifest['configuration']['source'] = r'C:\old-computer\data\raw\taco'
            detector.write_json(output / 'split_manifest.json', manifest)
            originals = {row[key]: (output / row[key]).read_bytes()
                         for row in manifest['images'] for key in ('image', 'label')}
            moved_source, moved_output = root / 'uploaded_raw', root / 'uploaded_prepared'
            source.rename(moved_source)
            output.rename(moved_output)
            with patch.object(detector.Image, 'open', side_effect=AssertionError('Unexpected image regeneration')):
                data = detector.prepare(moved_source, moved_output)
            self.assertEqual(data, moved_output / 'dataset.yaml')
            yaml = data.read_text()
            self.assertEqual(yaml.splitlines()[0], f'path: {json.dumps(moved_output.resolve().as_posix())}')
            self.assertIn('  0: "Bottle"', yaml)
            for relative, contents in originals.items():
                self.assertEqual((moved_output / relative).read_bytes(), contents)
            self.assertEqual(json.loads((moved_output / 'split_manifest.json').read_text()), manifest)

    def test_reuse_restores_missing_dataset_yaml_without_reprocessing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = self.make_raw_dataset(root), root / 'prepared'
            data = detector.prepare(source, output)
            data.unlink()
            with patch.object(detector.Image, 'open', side_effect=AssertionError('Unexpected image regeneration')):
                self.assertEqual(detector.prepare(source, output), data)
            self.assertTrue(data.is_file())

    def test_reuse_still_rejects_changed_settings_and_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = self.make_raw_dataset(root), root / 'prepared'
            detector.prepare(source, output)
            for key, value in (('seed', 99), ('max_side', 640), ('val_ratio', .2), ('test_ratio', .2)):
                with self.subTest(setting=key), self.assertRaisesRegex(ValueError, f'{key}: saved='):
                    detector.prepare(source, output, **{key: value})
            coco = json.loads((source / 'annotations.json').read_text())
            coco['categories'][0]['name'] = 'Changed class'
            (source / 'annotations.json').write_text(json.dumps(coco))
            with self.assertRaisesRegex(ValueError, 'annotation_sha256: saved='):
                detector.prepare(source, output)

    def test_missing_prepared_label_is_detected_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = self.make_raw_dataset(root), root / 'prepared'
            detector.prepare(source, output)
            manifest = json.loads((output / 'split_manifest.json').read_text())
            label = output / manifest['images'][0]['label']
            original = label.read_bytes()
            label.unlink()
            self.assertFalse(detector.prepared_files_exist(output, manifest['images']))
            detector.prepare(source, output)
            self.assertEqual(label.read_bytes(), original)
            self.assertTrue(detector.prepared_files_exist(output, manifest['images']))

    def test_pretrained_coco_classes_cannot_be_scored_as_taco_classes(self):
        manifest = {'class_names': {'0': 'Drink can'}}
        detector.check_class_names({0: 'Drink can'}, manifest)
        with self.assertRaisesRegex(ValueError, 'TACO-trained'):
            detector.check_class_names({0: 'person'}, manifest)

    def test_running_without_arguments_uses_editable_experiment(self):
        with patch.object(detector, 'run_experiment', return_value={'run_dir': 'saved'}) as run:
            self.assertEqual(detector.main([]), {'run_dir': 'saved'})
        run.assert_called_once_with(detector.DEFAULT_EXPERIMENT)

    def test_training_cli_still_accepts_explicit_parameters(self):
        with patch.object(detector, 'train') as train:
            detector.main(['train', '--epochs', '7', '--project', 'custom_runs', '--name', 'custom'])
        args = train.call_args.args[0]
        self.assertEqual((args.epochs, args.project, args.name), (7, Path('custom_runs'), 'custom'))

    def test_experiment_uses_actual_run_directory_and_best_weights_in_every_stage(self):
        for cuda_available, expected_device in ((False, 'cpu'), (True, '0')):
            with self.subTest(cuda=cuda_available), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'raw'
                source.mkdir()
                (source / 'annotations.json').write_text('{}')
                config = detector.ExperimentConfig(
                    source=source, prepared=root / 'prepared', project=root / 'runs',
                    name='experiment', epochs=3, imgsz=320, batch=4, lr=.002,
                    prediction_count=2, conf=.4)
                # The trainer has added a suffix that differs from config.name.
                run_dir = root / 'runs/experiment-2'
                best = run_dir / 'weights/best.pt'
                data = root / 'prepared/dataset.yaml'
                evaluation = run_dir / 'validation-2'
                torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))
                with patch.dict('sys.modules', {'torch': torch_stub}), \
                        patch.object(detector, 'prepare', return_value=data), \
                        patch.object(detector, 'train', return_value=best) as train, \
                        patch.object(detector, 'evaluate', return_value=evaluation) as evaluate, \
                        patch.object(detector, 'predict') as predict, \
                        patch.object(detector, 'prepare_tensorboard_logs', return_value=run_dir) as tensorboard:
                    artifacts = detector.run_experiment(config)
                tensorboard.assert_called_once_with(run_dir)
                training_args = train.call_args.args[0]
                evaluation_args = evaluate.call_args.args[0]
                prediction_args = predict.call_args.args[0]
                self.assertEqual((training_args.epochs, training_args.imgsz, training_args.lr), (3, 320, .002))
                self.assertEqual(evaluation_args.project, run_dir)
                self.assertEqual(prediction_args.output, run_dir / 'predictions')
                self.assertEqual((prediction_args.count, prediction_args.conf), (2, .4))
                for args in (training_args, evaluation_args, prediction_args):
                    self.assertEqual(args.device, expected_device)
                    self.assertEqual(args.data, data)
                for args in (evaluation_args, prediction_args):
                    self.assertEqual(args.weights, best)
                    self.assertEqual(args.split, 'val')
                self.assertEqual(artifacts['evaluation_dir'], str(evaluation))
                self.assertEqual(json.loads((run_dir / 'artifacts.json').read_text()), artifacts)
                saved = json.loads((run_dir / 'run_config.json').read_text())
                self.assertEqual((saved['device'], saved['source']), (expected_device, str(source)))

    def test_invalid_experiment_parameters_fail_before_preparing_or_training(self):
        with patch.object(detector, 'prepare') as prepare, patch.object(detector, 'train') as train:
            for config in (detector.ExperimentConfig(epochs=0), detector.ExperimentConfig(conf=1.1),
                           detector.ExperimentConfig(fraction=0), detector.ExperimentConfig(patience=-1)):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    detector.run_experiment(config)
            prepare.assert_not_called()
            train.assert_not_called()

    def test_display_results_reads_saved_artifacts_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, predictions = root / 'validation', root / 'predictions'
            detector.write_json(root / 'artifacts.json', {
                'evaluation_dir': str(evaluation), 'predictions_dir': str(predictions)})
            detector.write_json(evaluation / 'metrics.json', {
                'precision': .5, 'recall': .25, 'mAP50': .3, 'mAP50_95': .2})
            (evaluation / 'per_class.csv').write_text('class,objects,AP50\nBottle,2,0.3\n')
            predictions.mkdir()
            for path in (root / 'results.png', evaluation / 'confusion_matrix_normalized.png',
                         evaluation / 'BoxPR_curve.png', predictions / 'image_000001.jpg'):
                Image.new('RGB', (10, 10)).save(path)
            # Mock the notebook renderer while parsing the real saved metric tables.
            displayed, tables = [], []
            with patch.dict('sys.modules', {'IPython.display': SimpleNamespace(
                    Image=lambda filename: filename, display=lambda value: displayed.append(value))}), \
                    patch.object(detector, 'train') as train:
                detector.display_results(root, table_display=tables.append)
            train.assert_not_called()
            self.assertIsInstance(tables[0], pd.DataFrame)
            self.assertEqual(tables[0].iloc[0]['recall'], .25)
            self.assertEqual(tables[1].iloc[0]['class'], 'Bottle')
            self.assertEqual(len(displayed), 4)
            self.assertIn(str(predictions / 'image_000001.jpg'), displayed)

    def test_tensorboard_backfills_csv_and_images_without_retraining_or_duplicate_files(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results.csv').write_text(
                ' epoch ,train/box_loss,metrics/mAP50(B),note,invalid\n'
                '1,1.5,0.1,first,nan\n2,0.9,0.2,second,inf\n')
            Image.new('RGB', (20, 10), 'red').save(root / 'train_batch0.jpg')
            (root / 'predictions').mkdir()
            Image.new('RGB', (20, 10), 'blue').save(root / 'predictions/image_000001.jpg')
            with patch.object(detector, 'train') as train:
                self.assertEqual(detector.prepare_tensorboard_logs(root), root)
                event_path = root / 'events.out.tfevents.taco'
                original = event_path.read_bytes()
                detector.prepare_tensorboard_logs(root)
                self.assertEqual(event_path.read_bytes(), original)
            train.assert_not_called()
            self.assertEqual(len(list(root.glob('events.out.tfevents.*'))), 1)
            events = EventAccumulator(str(root)).Reload()
            losses = events.Scalars('train/box_loss')
            self.assertEqual([event.step for event in losses], [1, 2])
            self.assertAlmostEqual(losses[-1].value, .9)
            self.assertNotIn('invalid', events.Tags()['scalars'])
            self.assertEqual(set(events.Tags()['images']), {'data/train_batch0', 'data/image_000001'})

    def test_tensorboard_preserves_native_metrics_without_backfilling_duplicates(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SummaryWriter(str(root)) as writer:
                writer.add_scalar('train/box_loss', 1.5, 1)
            (root / 'results.csv').write_text('epoch,train/box_loss\n1,999\n')
            detector.prepare_tensorboard_logs(root)
            events = EventAccumulator(str(root)).Reload()
            self.assertEqual([event.value for event in events.Scalars('train/box_loss')], [1.5])

    def test_tensorboard_requires_training_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'No TensorBoard events or results.csv'):
                detector.prepare_tensorboard_logs(directory)


if __name__ == '__main__':
    unittest.main()
