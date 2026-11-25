"""Classify the archived SortWaste test detections with both app C4 checkpoints."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile

import sortwaste as base

YOLO_SHA = 'ba6e96fc8bc0b6debc07110475a3a6e6ec79edf409e372791e48deca11973c20'
C4_SHA = '628c374dc53f5028ba21ed14387c9c54a89f271df4379c7c285c535a647a27f9'
POST_SHA = 'bc81ad64aa202dfda0269733b7770e7d5b4bce3605d4432aac1e4d98ccd296f4'
TEST_SHA = 'db5f78052ab584af207be689e6cdf8059ba17bf3c7742da5a9398529d8179efc'
SHARED = ('cardboard', 'metal', 'plastic')
NAMES = {'yolo': 'SortWaste-YOLO', 'yolo_resnet': 'SortWaste-YOLO-ResNet (no post-training)',
         'yolo_post_resnet': 'SortWaste-YOLO-ResNet post-trained'}


def evaluate(archive, data, classifier, adapted, output, input_hashes):
    import torch
    from PIL import Image

    archive, data, output = Path(archive), Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output / 'status.json', {'state': 'verifying_inputs'})
    for name, expected in input_hashes.items():
        assert base.sha256(archive / name) == expected, name
    original = json.loads((archive / 'metrics.json').read_text())
    assert original['config']['yolo_sha256'] == YOLO_SHA
    assert original['config']['resnet_sha256'] == C4_SHA
    assert original['config']['test_list_sha256'] == TEST_SHA
    assert base.sha256(data / 'sortwaste_test.txt') == TEST_SHA
    assert base.sha256(classifier) == C4_SHA and base.sha256(adapted) == POST_SHA
    assert torch.cuda.is_available(), 'The comparison uses the original GPU inference environment'
    device = 'cuda:0'
    test = [Path(line.strip()) for line in (data / 'sortwaste_test.txt').read_text().splitlines() if line.strip()]
    assert len(test) == 776 and len({p.name for p in test}) == 776
    assert all(not p.is_absolute() and p.is_relative_to('images/sortwaste/test') and '..' not in p.parts for p in test)
    with (archive / 'objects.csv').open(newline='') as stream:
        objects = list(csv.DictReader(stream))
    primary = [r for r in objects if r['classifier_eligible'] == 'True' and r['prediction_index'] != '']
    assert len(primary) == 8009
    assert Counter(r['gt_class'] for r in primary) == {'cardboard': 146, 'metal': 155, 'plastic': 7708}
    by_photo, primary_by_photo = defaultdict(list), defaultdict(list)
    for line in (archive / 'predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert float(row['yolo_confidence']) >= .35
        by_photo[row['image']].append(row)
    assert sum(map(len, by_photo.values())) == 13250
    for row in primary:
        primary_by_photo[row['image']].append(row)
    assert set(by_photo) == {p.name for p in test}
    original_c4 = base.ResNetClassifier(classifier, device)
    post_c4 = base.ResNetClassifier(adapted, device)
    assert original_c4.epoch == 14 and post_c4.epoch == 3
    assert post_c4.config['source_resnet_sha256'] == C4_SHA
    assert post_c4.config['source_yolo_sha256'] == YOLO_SHA
    assert post_c4.config['epochs'] == 10
    predictions, numerical_changes = [], 0
    with tempfile.TemporaryDirectory(prefix='sortwaste_matched_test_') as temporary:
        staged = Path(temporary)

        def stage(relative):
            target = staged / relative.name
            shutil.copyfile(data / relative, target)
            return relative.name, target

        with ThreadPoolExecutor(max_workers=8) as pool:
            image_paths = dict(pool.map(stage, test))
        for number, relative in enumerate(test, 1):
            name = relative.name
            cached = sorted(by_photo[name], key=lambda r: r['prediction_index'])
            assert [r['prediction_index'] for r in cached] == list(range(len(cached)))
            with Image.open(image_paths[name]) as opened:
                image = opened.convert('RGB')
            width, height = image.size
            crops = []
            for detection in cached:
                x1, y1, x2, y2 = detection['box_xyxy']
                crops.append(image.crop((max(0, math.floor(x1)), max(0, math.floor(y1)),
                    min(width, max(math.floor(x1) + 1, math.ceil(x2))),
                    min(height, max(math.floor(y1) + 1, math.ceil(y2))))))
            before, after = original_c4.predict(crops), post_c4.predict(crops)
            for target in primary_by_photo[name]:
                index = int(target['prediction_index'])
                detector = cached[index]
                assert detector['yolo_class'] == target['yolo_class']
                assert abs(detector['yolo_confidence'] - float(target['yolo_confidence'])) < 1e-8
                c4_label, post_label = base.RESNET_CLASSES[before[index][0]], base.RESNET_CLASSES[after[index][0]]
                numerical_changes += c4_label != target['yolo_resnet_class']
                predictions.append({'image': name, 'target_index': int(target['target_index']),
                    'prediction_index': index, 'truth': target['gt_class'], 'yolo': detector['yolo_class'],
                    'yolo_resnet': c4_label, 'yolo_post_resnet': post_label,
                    'yolo_confidence': detector['yolo_confidence'], 'resnet_probability': before[index][1],
                    'post_resnet_probability': after[index][1]})
            if number % 25 == 0 or number == len(test):
                progress = {'state': 'evaluating', 'images_completed': number, 'test_images': len(test),
                            'classified_objects': len(predictions)}
                base.write_json(output / 'status.json', progress)
                print(json.dumps(progress), flush=True)
    assert len(predictions) == 8009
    systems = {system: base.classification_metrics([(r['truth'], r[system]) for r in predictions], base.RESNET_CLASSES)
               for system in NAMES}
    expected = original['systems']['yolo']['primary_classifier_eligible']
    for key in ('count', 'correct', 'accuracy', 'macro_f1', 'weighted_f1'):
        assert abs(systems['yolo'][key] - expected[key]) < 1e-12, key
    for metrics in systems.values():
        supported = [r for r in metrics['per_class'] if r['support']]
        metrics['macro_precision'] = sum(r['precision'] for r in supported) / len(supported)
        metrics['macro_recall'] = sum(r['recall'] for r in supported) / len(supported)
    result = {'test_images': 776, 'matched_objects': 8009, 'systems': systems,
        'protocol': {'scope': 'Classification on identical archived, matched test objects',
            'detector_confidence': .35, 'matching_iou': .5, 'image_size': 640, 'max_detections': 300,
            'classes_scored': list(SHARED), 'missed_objects_and_unmatched_predictions': 'excluded',
            'carton': 'excluded; the six-class ResNet has no carton output',
            'classifier_confidence_filter': 'none; argmax for each of the same matched objects',
            'macro_average': 'unweighted mean over the three ground-truth-supported materials'},
        'checkpoint_sha256': {'yolo': YOLO_SHA, 'original_c4': C4_SHA, 'post_trained_c4': POST_SHA},
        'post_training': {'epochs_completed': 10, 'selected_epoch': 3, 'yolo_frozen': True},
        'test_list_sha256': TEST_SHA, 'input_sha256': input_hashes,
        'original_c4_predictions_changed_vs_archived_gpu': numerical_changes,
        'original_c4_archive_metrics': {k:v for k,v in original['systems']['yolo_resnet']['primary_classifier_eligible'].items()
                                       if k != 'per_class'},
        'device': torch.cuda.get_device_name(0), 'torch_version': str(torch.__version__)}
    base.write_json(output / 'metrics.json', result)
    with (output / 'objects.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'objects', 'correct', 'accuracy', 'macro_precision',
                                                  'macro_recall', 'macro_f1', 'weighted_f1'])
        writer.writeheader()
        for system, metrics in systems.items():
            writer.writerow({'model': NAMES[system], 'objects': metrics['count'], 'correct': metrics['correct'],
                **{k: metrics[k] for k in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1')}})
    with (output / 'per_class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'material', 'objects', 'precision', 'recall', 'f1'])
        writer.writeheader()
        for system, metrics in systems.items():
            for row in metrics['per_class']:
                if row['support']:
                    writer.writerow({'model': NAMES[system], 'material': row['label'], 'objects': row['support'],
                                     **{k: row[k] for k in ('precision', 'recall', 'f1')}})
    base.write_json(output / 'status.json', {'state': 'success', 'test_images': 776, 'matched_objects': 8009})
    return result
