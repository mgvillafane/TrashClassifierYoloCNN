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

    def test_official_model_names_use_weights_storage_and_custom_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(detector, 'STORAGE', root):
                for name in ('yolo26m.pt', 'yolo11n.pt'):
                    self.assertEqual(detector.resolve_weights(name), root / 'data/yolo/weights' / name)
                custom = root / 'custom/checkpoint.pt'
                self.assertEqual(detector.resolve_weights(custom), custom.resolve())
            self.assertTrue((root / 'data/yolo/weights').is_dir())

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

    def test_saved_run_scores_test_checkpoint_and_preserves_reports_after_staging_cleanup(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import torch
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'saved_run'
            artifacts = {
                'run_dir': str(run), 'best_weights': str(run / 'weights/best.pt'),
                'data': str(root / 'prepared/dataset.yaml'),
                'evaluation_dir': str(run / 'validation'),
                'predictions_dir': str(run / 'predictions'),
                'tensorboard_dir': str(run)}
            detector.write_json(run / 'artifacts.json', artifacts)
            with SummaryWriter(str(run)) as writer:
                writer.add_scalar('train/box_loss', 1.5, 1)
            next(run.glob('events.out.tfevents.*')).rename(run / 'events.out.tfevents.taco')
            events = EventAccumulator(str(run)).Reload()

            def evaluate_test(args):
                self.assertEqual(args.split, 'test')
                self.assertEqual(args.weights, Path(artifacts['best_weights']))
                self.assertEqual(args.data, Path(artifacts['data']))
                self.assertEqual((args.imgsz, args.batch, args.device, args.threads), (640, 8, '0', 4))
                output = args.project / 'test'
                detector.write_json(output / 'metrics.json', {
                    'split': 'test', 'precision': .75, 'recall': .5, 'mAP50': .2, 'mAP50_95': .1})
                (output / 'per_class.csv').write_text('class,objects,AP50\nBottle,2,0.2\n')
                (output / 'plots').mkdir()
                (output / 'plots/confusion.png').write_bytes(b'saved plot')
                return output

            with patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(detector, 'evaluate', side_effect=evaluate_test) as evaluate, \
                    patch.object(detector, 'train') as train, \
                    patch.object(detector, 'prepare') as prepare:
                target = detector.test_saved_run(run)
                event_path = run / 'events.out.tfevents.test'
                original = event_path.read_bytes()
                detector.test_saved_run(run)
                self.assertEqual(event_path.read_bytes(), original)
            train.assert_not_called()
            prepare.assert_not_called()
            self.assertEqual(evaluate.call_count, 2)
            self.assertTrue(all(not call.args[0].project.exists() for call in evaluate.call_args_list))
            self.assertEqual(target, run / 'test')
            self.assertEqual(json.loads((target / 'metrics.json').read_text())['split'], 'test')
            self.assertIn('Bottle,2,0.2', (target / 'per_class.csv').read_text())
            self.assertEqual((target / 'plots/confusion.png').read_bytes(), b'saved plot')
            self.assertEqual(json.loads((run / 'artifacts.json').read_text()), {
                **artifacts, 'test_dir': str(target)})
            events.Reload()
            expected = {'test/precision': .75, 'test/recall': .5, 'test/mAP50': .2, 'test/mAP50_95': .1}
            self.assertEqual(set(events.Tags()['scalars']), {'train/box_loss', *expected})
            self.assertEqual([item.value for item in events.Scalars('train/box_loss')], [1.5])
            for tag, value in expected.items():
                scalars = events.Scalars(tag)
                self.assertEqual([item.step for item in scalars], [0])
                self.assertAlmostEqual(scalars[0].value, value)

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

    def test_live_tensorboard_publishes_epochs_and_final_images_to_an_open_reader(self):
        from shutil import copyfile
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as directory:
            local, output = Path(directory) / 'local', Path(directory) / 'published'
            local.mkdir()
            Image.new('RGB', (20, 10), 'red').save(local / 'train_batch0.jpg')
            trainer = SimpleNamespace(
                save_dir=local, epoch=0, tloss=1.5,
                label_loss_items=lambda losses, prefix: {f'{prefix}/box_loss': losses},
                metrics={'metrics/mAP50(B)': .1}, lr={'lr/pg0': .001})

            first = detector.publish_tensorboard_epoch(trainer, output)
            self.assertTrue(first.is_file())
            self.assertEqual(first.name, 'events.out.tfevents.epoch000001')
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual([event.step for event in events.Scalars('train/box_loss')], [1])
            self.assertIn('data/train_batch0', events.Tags()['images'])

            trainer.epoch, trainer.tloss = 1, .9
            trainer.metrics = {'metrics/mAP50(B)': .2}
            detector.publish_tensorboard_epoch(trainer, output)
            events.Reload()
            for tag in ('train/box_loss', 'metrics/mAP50(B)', 'lr/pg0'):
                self.assertEqual([event.step for event in events.Scalars(tag)], [1, 2])
            self.assertAlmostEqual(events.Scalars('train/box_loss')[-1].value, .9)
            self.assertAlmostEqual(events.Scalars('metrics/mAP50(B)')[-1].value, .2)
            published = {path.name: path.read_bytes() for path in output.glob('events.out.tfevents.*')}

            # Ignore repeated callbacks for an already published training epoch.
            trainer.tloss = 999
            trainer.metrics = {'metrics/mAP50(B)': 999}
            detector.publish_tensorboard_epoch(trainer, output)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.glob('events.out.tfevents.*')}, published)
            events.Reload()
            self.assertEqual([event.step for event in events.Scalars('train/box_loss')], [1, 2])
            self.assertAlmostEqual(events.Scalars('train/box_loss')[-1].value, .9)

            # Real final_eval increments epoch while validating the best checkpoint.
            # It must not create a phantom third training epoch (also after early stop).
            trainer.epoch = 2
            trainer.validator = SimpleNamespace(training=False)
            self.assertIsNone(detector.publish_tensorboard_epoch(trainer, output))
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.glob('events.out.tfevents.*')}, published)
            self.assertFalse((local / 'events.out.tfevents.epoch000003').exists())
            events.Reload()
            self.assertEqual([event.step for event in events.Scalars('train/box_loss')], [1, 2])

            # The same reader must discover the final summary after the live epoch files.
            (local / 'predictions').mkdir()
            Image.new('RGB', (20, 10), 'blue').save(local / 'predictions/image_000001.jpg')
            (local / 'results.csv').write_text('epoch,train/box_loss\n1,999\n2,999\n')
            detector.prepare_tensorboard_logs(local)
            copyfile(local / 'events.out.tfevents.taco', output / 'events.out.tfevents.taco')
            events.Reload()
            self.assertIn('data/image_000001', events.Tags()['images'])
            self.assertEqual([event.step for event in events.Scalars('train/box_loss')], [1, 2])
            self.assertAlmostEqual(events.Scalars('train/box_loss')[-1].value, .9)

    def test_live_tensorboard_viewer_starts_on_an_empty_directory(self):
        from importlib import import_module

        # Force the real module: patching TensorBoard's initial lazy proxy misses later imports.
        notebook = import_module('tensorboard.notebook')

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'pending_run'
            with patch.object(notebook, 'start', return_value='viewer') as start, \
                    patch.object(detector, 'prepare_tensorboard_logs') as prepare:
                self.assertEqual(detector.display_tensorboard(output, live=True), 'viewer')
            prepare.assert_not_called()
            self.assertTrue(output.is_dir())
            command = start.call_args.args[0]
            self.assertIn(str(output.resolve()), command)
            self.assertIn('--reload_interval 5', command)

    def test_live_training_publishes_from_the_registered_epoch_callback(self):
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'dataset.yaml'
            data.write_text('names: [Bottle]\n')
            arguments = asdict(detector.ExperimentConfig(
                project=root / 'local', weights=root / 'initial.pt',
                tensorboard_dir=root / 'published', epochs=2, device='cpu'))
            arguments['data'] = data
            args = SimpleNamespace(**arguments)
            trainer = SimpleNamespace(save_dir=root / 'local/taco')
            callbacks, settings = {}, {'tensorboard': True}

            def train_model(**kwargs):
                self.assertFalse(settings['tensorboard'])
                callbacks['on_fit_epoch_end'](trainer)
                publish.assert_called_once_with(trainer, args.tensorboard_dir)

            model = SimpleNamespace(
                trainer=trainer, train=train_model,
                add_callback=lambda event, callback: callbacks.update({event: callback}))
            with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(settings=settings)}), \
                    patch.object(detector, 'load_yolo', return_value=model), \
                    patch.object(detector, 'publish_tensorboard_epoch') as publish:
                best = detector.train(args)
            self.assertEqual(best, trainer.save_dir / 'weights/best.pt')


if __name__ == '__main__':
    unittest.main()
