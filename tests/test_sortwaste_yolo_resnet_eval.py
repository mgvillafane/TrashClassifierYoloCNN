from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from evaluation.sortwaste import CLASSES, RESNET_CLASSES, classification_metrics, detector_classes, match_boxes, paired_bootstrap, read_splits, summarize, taco_eligibility
from evaluation.post_training import EarlyStopping, crop_targets, trainable_parameters, training_mode, verify_reference


class PairedEvaluationTests(unittest.TestCase):
    def test_taco_six_class_and_sortwaste_seven_class_detectors_keep_the_same_material_ids(self):
        self.assertEqual(detector_classes(dict(enumerate(RESNET_CLASSES))), RESNET_CLASSES)
        self.assertEqual(detector_classes(dict(enumerate(CLASSES))), CLASSES)
        with self.assertRaises(ValueError):
            detector_classes(['glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash'])

    def test_three_system_report_reuses_the_identical_matched_targets(self):
        records = [dict(image=name, gt_class='plastic', classifier_eligible=True, prediction_index=0,
                        yolo_class='metal', yolo_resnet_class='cardboard', yolo_post_resnet_class='plastic')
                   for name in ('a', 'b')]
        result = summarize(records)
        self.assertEqual(len(result['systems']), 3)
        self.assertEqual([item['primary_classifier_eligible']['count'] for item in result['systems'].values()], [2, 2, 2])
        self.assertEqual(result['systems']['yolo_post_resnet']['primary_classifier_eligible']['accuracy'], 1)
        self.assertEqual(result['post_training_primary_outcomes']['resnet_corrected_yolo'], 2)
        interval = paired_bootstrap(records, repetitions=100, system='yolo_post_resnet')
        self.assertEqual(interval['accuracy_delta_95ci'], [1, 1])

    def test_matching_uses_geometry_and_never_matches_one_object_twice(self):
        matches = match_boxes([[0, 0, 1, 1], [0, 0, .9, .9], [2, 2, 3, 3]], [[0, 0, 1, 1]])
        self.assertEqual(matches, [(0, 0, 1.0)])
        self.assertEqual(match_boxes([[0, 0, .1, .1]], [[0, 0, 1, 1]]), [])

    def test_predictions_outside_shared_classes_still_count_as_classification_errors(self):
        result = classification_metrics([('plastic', 'plastic'), ('cardboard', 'carton'), ('carton', 'cardboard')], RESNET_CLASSES)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['macro_f1_classes'], ['cardboard', 'plastic'])

    def test_primary_is_paired_and_separate_from_misses_false_positives_and_cartons(self):
        def row(truth, before, after, eligible=True, prediction=0):
            return dict(gt_class=truth, yolo_class=before, yolo_resnet_class=after,
                        classifier_eligible=eligible, prediction_index=prediction)
        records = [row('cardboard', 'plastic', 'cardboard'), row('metal', 'metal', 'plastic'),
                   row('plastic', 'carton', 'plastic'), row('carton', 'carton', 'paper', False),
                   row('plastic', None, None, prediction=None), row(None, 'glass', 'metal', False)]
        report = summarize(records)
        self.assertEqual(report['counts']['matched_objects'], 4)
        self.assertEqual(report['counts']['missed_objects'], 1)
        self.assertEqual(report['counts']['unmatched_predictions'], 1)
        self.assertEqual(report['localization_recall'], .8)
        self.assertEqual(report['systems']['yolo']['primary_classifier_eligible']['accuracy'], 1 / 3)
        self.assertEqual(report['systems']['yolo_resnet']['primary_classifier_eligible']['accuracy'], 2 / 3)
        self.assertEqual(report['paired_primary_outcomes']['resnet_corrected_yolo'], 2)
        self.assertEqual(report['paired_primary_outcomes']['resnet_broke_correct_yolo'], 1)
        self.assertEqual(report['systems']['yolo']['end_to_end_all_seven']['precision'], .4)
        self.assertEqual(report['systems']['yolo']['end_to_end_all_seven']['recall'], .4)
        self.assertEqual(report['systems']['yolo']['carton_matched']['accuracy'], 1)
        self.assertEqual(report['systems']['yolo_resnet']['carton_matched']['accuracy'], 0)

    def test_no_detections_is_a_valid_report_with_misses(self):
        report = summarize([dict(gt_class='plastic', yolo_class=None, yolo_resnet_class=None,
                                 classifier_eligible=True, prediction_index=None)])
        self.assertIsNone(report['systems']['yolo']['primary_classifier_eligible']['accuracy'])
        self.assertEqual(report['systems']['yolo']['end_to_end_all_seven']['recall'], 0)

    def test_source_image_overlap_between_test_and_training_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split, name in [('train', 'one'), ('val', 'two'), ('test', 'one')]:
                (root / f'sortwaste_{split}.txt').write_text(f'images/sortwaste/{split}/{name}.jpg\n')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                read_splits(root)

    def test_paired_image_bootstrap_preserves_a_known_improvement(self):
        records = [dict(image=name, gt_class='plastic', yolo_class='metal', yolo_resnet_class='plastic',
                        classifier_eligible=True, prediction_index=0) for name in ('a', 'b')]
        intervals = paired_bootstrap(records, repetitions=100)
        self.assertEqual(intervals['accuracy_delta_95ci'], [1, 1])
        self.assertEqual(intervals['macro_f1_delta_95ci'], [1, 1])


class PostTrainingTests(unittest.TestCase):
    def test_early_stopping_counts_consecutive_non_improving_epochs_and_resets(self):
        stopper = EarlyStopping(patience=2)
        self.assertFalse(stopper.update(.4))
        self.assertFalse(stopper.update(.4))
        self.assertFalse(stopper.update(.5))
        self.assertFalse(stopper.update(.49))
        self.assertTrue(stopper.update(.48))
        self.assertEqual(stopper.best, .5)
        disabled = EarlyStopping(0)
        for _ in range(50):
            self.assertFalse(disabled.update(.1))
        with self.assertRaises(ValueError):
            EarlyStopping(-1)

    def test_changed_original_manifest_is_rejected_before_any_images_are_opened(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'split_manifest.json').write_text('{}')
            args = SimpleNamespace(dataset='taco', reference_manifest_sha256='incorrect')
            with self.assertRaisesRegex(ValueError, 'original standalone YOLO split'):
                verify_reference(root, args, {})

    def test_training_targets_are_ground_truth_and_exclude_unmatched_boxes_and_cartons(self):
        targets = [dict(class_id=4, box=[0, 0, .4, .4]), dict(class_id=6, box=[.5, .5, 1, 1])]
        predictions = [[0, 0, .4, .4], [.5, .5, 1, 1], [.45, 0, .6, .2]]
        eligible = {('source', 0): 'plastic'}
        self.assertEqual(crop_targets(predictions, targets, eligible, 'source'), [(0, 4)])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            crop_targets(predictions, targets, {('source', 0): 'metal'}, 'source')

    def test_optimizer_changes_only_the_last_block_and_head_and_preserves_bn_statistics(self):
        import torch
        from torch import nn

        torch.manual_seed(42)
        model = nn.Module()
        model.layer1 = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        model.layer4 = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        model.fc = nn.Linear(4, 6)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(trainable_parameters(model))
        training_mode(model)
        logits = model.fc(model.layer4(model.layer1(torch.randn(8, 4))))
        nn.functional.cross_entropy(logits, torch.tensor([0, 2, 4, 0, 2, 4, 0, 4])).backward()
        optimizer.step()
        after = model.state_dict()
        for name in before:
            if name.startswith('layer1.') or 'running_' in name or 'num_batches_tracked' in name:
                self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertFalse(torch.equal(before['layer4.0.weight'], after['layer4.0.weight']))
        self.assertFalse(torch.equal(before['fc.weight'], after['fc.weight']))


class TacoDatasetTests(unittest.TestCase):
    def manifest(self):
        return dict(class_names=dict(enumerate(RESNET_CLASSES)),
                    image_counts=dict(train=1, val=1, test=1),
                    images=[dict(image=f'images/{split}/{number}.jpg', split=split,
                                 source_file=f'batch_{number}/source.jpg')
                            for number, split in enumerate(('train', 'val', 'test'))])

    def test_taco_splits_and_training_targets_do_not_open_test_images_or_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'split_manifest.json').write_text(json.dumps(self.manifest()))
            split = read_splits(root, 'taco')
            labels = root / 'labels/train'
            labels.mkdir(parents=True)
            (labels / '0.txt').write_text('\n'.join(f'{i} 0.5 0.5 0.5 0.5' for i in range(6)))
            targets = taco_eligibility(root, split['train'])
            self.assertEqual(set(targets.values()), set(RESNET_CLASSES))
            self.assertEqual(len(targets), 6)
            self.assertFalse((root / 'labels/test').exists())

    def test_duplicate_taco_source_in_different_split_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest()
            manifest['images'][2]['source_file'] = manifest['images'][0]['source_file']
            (root / 'split_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                read_splits(root, 'taco')

    def test_seventh_taco_class_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'labels/train').mkdir(parents=True)
            (root / 'labels/train/0.txt').write_text('6 0.5 0.5 0.5 0.5\n')
            with self.assertRaisesRegex(ValueError, 'seventh class'):
                taco_eligibility(root, [Path('images/train/0.jpg')])


class DetectionMetricTests(unittest.TestCase):
    def test_score_ranked_class_matching_counts_wrong_labels_and_duplicates(self):
        from evaluation.taco_detection import class_matches
        box = [0, 0, 1, 1]
        targets = [dict(class_id=4, box=box)]
        predictions = [dict(class_id=4, box=box, confidence=.9),
                       dict(class_id=4, box=box, confidence=.8),
                       dict(class_id=2, box=box, confidence=.95)]
        support, matches = class_matches(predictions, targets, 4)
        self.assertEqual(support, 1)
        self.assertEqual(matches, [(.9, True), (.8, False)])
        self.assertEqual(class_matches(predictions, targets, 2), (0, [(.95, False)]))

    def test_detection_macro_f1_includes_misses_and_bootstrap_stays_paired(self):
        from evaluation.taco_detection import summarize, SYSTEMS
        box = [0, 0, 1, 1]
        targets = [dict(class_id=i, box=box) for i in range(6)]
        perfect = [dict(class_id=i, box=box, confidence=.9) for i in range(6)]
        image = dict(image='one', targets=targets, yolo=[], yolo_resnet=perfect, yolo_post_resnet=perfect)
        report = summarize([image], repetitions=5)
        self.assertEqual(report['systems']['yolo']['macro_f1'], 0)
        self.assertEqual(report['systems']['yolo_post_resnet']['macro_f1'], 1)
        self.assertEqual(report['paired_differences']['yolo_post_resnet minus yolo']['macro_f1']['interval_95'], [1, 1])
        self.assertGreater(report['systems']['yolo_post_resnet']['map50'], .99)


if __name__ == '__main__':
    unittest.main()
