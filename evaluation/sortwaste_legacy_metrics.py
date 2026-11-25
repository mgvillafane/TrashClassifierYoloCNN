"""Reproduce the archived five-system SortWaste test protocol exactly.

AP ranks by the final material confidence (detector x classifier for hybrids).
F1 filters by YOLO detector confidence, so all heads share the same proposals.
A false positive overlapping a carton at IoU >= 0.50 is ignored. A correct
shared-material match is retained even if its box also overlaps a carton.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .sortwaste_joint_metrics import LABELS, SHARED, overlaps


def prepare(images, system):
    prepared = []
    for label in SHARED:
        support, values = [], []
        for photo_index,image in enumerate(images):
            truth = [t['box'] for t in image['targets'] if t['class_id']==label]
            cartons = [t['box'] for t in image['targets'] if t['class_id']==6]
            support.append(len(truth))
            used = set()
            predictions = sorted((p for p in image[system] if p['class_id']==label),
                                 key=lambda p:p['confidence'],reverse=True)
            for prediction in predictions:
                ious = overlaps(prediction['box'],truth)
                if used:
                    ious[list(used)] = -1
                target = int(ious.argmax()) if len(ious) else -1
                correct = target>=0 and ious[target]>=.5
                if correct:
                    used.add(target)
                elif overlaps(prediction['box'],cartons).max(initial=0)>=.5:
                    continue
                values.append((prediction['confidence'],correct,photo_index,prediction['detector_confidence']))
        values.sort(key=lambda row:row[0],reverse=True)
        prepared.append((np.asarray(support),np.asarray([r[0] for r in values]),
                         np.asarray([r[1] for r in values],dtype=float),
                         np.asarray([r[2] for r in values],dtype=int),
                         np.asarray([r[3] for r in values])))
    return prepared


def score(prepared, weights=None):
    if weights is None:
        weights = np.ones(len(prepared[0][0]),dtype=int)
    rows = []
    for label,(support,confidence,correct,photo_index,detector_confidence) in zip(SHARED,prepared):
        count = int(weights @ support)
        pred_weights = weights[photo_index]
        keep = pred_weights>0
        true_positives = np.cumsum((correct*pred_weights)[keep])
        false_positives = np.cumsum(((1-correct)*pred_weights)[keep])
        precision = true_positives / np.maximum(true_positives+false_positives,1e-12)
        recall = true_positives / max(count,1)
        if count and len(precision):
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            indices = np.searchsorted(recall,np.linspace(0,1,101),side='left')
            ap50 = float(np.r_[envelope,0.][indices].mean())
        else:
            ap50 = 0.
        chosen = detector_confidence >= .25
        tp = int((correct*pred_weights)[chosen].sum())
        predicted = int(pred_weights[chosen].sum())
        rows.append({'class':LABELS[label],'n_gt':count,'ap50':ap50,
                     'precision':tp/predicted if predicted else 0.,
                     'recall':tp/count if count else 0.,
                     'f1':2*tp/(predicted+count) if predicted+count else 0.,
                     'true_positives':tp,'predicted_boxes':predicted})
    return {'map50':float(np.mean([r['ap50'] for r in rows])),
            'macro_f1':float(np.mean([r['f1'] for r in rows])),
            'macro_precision':float(np.mean([r['precision'] for r in rows])),
            'macro_recall':float(np.mean([r['recall'] for r in rows])), 'per_class':rows}


def summarize(images, systems=('yolo','pipeline'), repetitions=1000):
    prepared = {system:prepare(images,system) for system in systems}
    results = {system:score(value) for system,value in prepared.items()}
    samples = {s:{m:[] for m in ['map50','macro_f1','macro_precision','macro_recall']} for s in systems}
    rng = np.random.default_rng(42)
    for _ in range(repetitions):
        weights = rng.multinomial(len(images),np.full(len(images),1/len(images)))
        for system in systems:
            values = score(prepared[system],weights)
            for metric in samples[system]:
                samples[system][metric].append(values[metric])
    if repetitions:
        for system in systems:
            results[system]['intervals_95'] = {metric:np.quantile(value,[.025,.975]).tolist()
                                               for metric,value in samples[system].items()}
    paired = {}
    if repetitions and 'baseline_pipeline' in samples:
        for new,old,name in [('pipeline','baseline_pipeline','pipeline minus baseline C4'),
                             ('yolo','baseline_yolo','YOLO minus baseline YOLO')]:
            paired[name] = {metric:{'difference':results[new][metric]-results[old][metric],
                          'interval_95':np.quantile(np.asarray(samples[new][metric])-samples[old][metric],[.025,.975]).tolist()}
                            for metric in samples[new]}
    return {'systems':results,'test_images':len(images),'paired_differences':paired,
            'protocol':{'classes':[LABELS[i] for i in SHARED],'iou_match':.5,'image_size':640,
              'f1_confidence':.25,'f1_confidence_source':'YOLO detector confidence for every head',
              'predict_confidence':.01,'max_boxes_per_photo':100,'agnostic_nms_iou':.7,
              'carton':'Ignore unmatched predictions overlapping carton >= 0.50; preserve correct shared-class matches',
              'ap_implementation':'COCO 101-point interpolated precision',
              'hybrid_ap_score':'YOLO confidence multiplied by ResNet max softmax',
              'bootstrap':repetitions,'seed':42}}


def archived_images(baseline, labels, test_list):
    from .sortwaste import read_labels
    by_photo = {}
    for entry in Path(test_list).read_text().splitlines():
        path = Path(entry)
        by_photo[path.as_posix()] = {'image':path.as_posix(),
            'targets':read_labels(Path(labels)/path.with_suffix('.txt').name)}
    sources = [
        ('predictions_yolo_sortwaste.csv.gz',{'yolo':'e2e_sortwaste','classifier_sortwaste':'pipeline_sortwaste'}),
        ('predictions_yolo_all.csv.gz',{'yolo':'e2e_all','classifier_all':'pipeline_all','classifier_no_sortwaste':'pipeline_no_sortwaste'})]
    for filename,heads in sources:
        for image in by_photo.values():
            for system in heads.values():
                image[system] = []
        with gzip.open(Path(baseline)/filename,'rt',newline='') as stream:
            for row in csv.DictReader(stream):
                if row['photo'] not in by_photo:
                    continue
                box = [float(row[k]) for k in ['x1','y1','x2','y2']]
                for head,system in heads.items():
                    by_photo[row['photo']][system].append({'box':box,'class_id':int(row[head+'_label']),
                        'confidence':float(row[head+'_score']),'detector_confidence':float(row['det_conf'])})
    return list(by_photo.values())


def verify_archive(baseline, labels, test_list):
    baseline = Path(baseline)
    images = archived_images(baseline,labels,test_list)
    systems = ('e2e_sortwaste','pipeline_sortwaste','e2e_all','pipeline_all','pipeline_no_sortwaste')
    measured = summarize(images,systems,repetitions=0)
    with (baseline/'summary.csv').open(newline='',encoding='utf-8-sig') as stream:
        expected = {r['system']:r for r in csv.DictReader(stream)
                    if r['test_set']=='sortwaste_test' and r['classes_scored']=='shared3'}
    with (baseline/'per_class.csv').open(newline='',encoding='utf-8-sig') as stream:
        classes = {(r['system'],r['class']):r for r in csv.DictReader(stream)
                   if r['test_set']=='sortwaste_test' and r['class'] in [LABELS[i] for i in SHARED]}
    deviations = {}
    for system,value in measured['systems'].items():
        for metric in ['map50','macro_f1','macro_precision','macro_recall']:
            deviations[system+'/'+metric] = value[metric]-float(expected[system][metric])
        for row in value['per_class']:
            for metric in ['ap50','precision','recall','f1']:
                deviations[system+'/'+row['class']+'/'+metric] = row[metric]-float(classes[(system,row['class'])][metric])
    return {'passed':max(map(abs,deviations.values()))<1e-12,
            'verified_values':len(deviations),'max_absolute_deviation':max(map(abs,deviations.values())),
            'deviations':deviations,'measured':measured}


def attach_detector_confidence(images):
    for image in images:
        assert len(image['yolo'])==len(image['pipeline'])
        for detector,hybrid in zip(image['yolo'],image['pipeline']):
            assert detector['box']==hybrid['box']
            detector['detector_confidence'] = detector['confidence']
            hybrid['detector_confidence'] = detector['confidence']
    return images


def finalize(local_output, baseline):
    local_output,baseline = Path(local_output),Path(baseline)
    verification = verify_archive(baseline,local_output/'reference/test_labels',local_output/'reference/sortwaste_test.txt')
    (local_output/'evaluation/baseline_reproduction_exact.json').write_text(json.dumps(verification,indent=2)+'\n')
    if not verification['passed']:
        raise RuntimeError('Archived scoring does not reproduce exactly')
    path = local_output/'evaluation/metrics.json'
    original = json.loads(path.read_text())
    with gzip.open(local_output/'evaluation/predictions.json.gz','rt') as stream:
        images = attach_detector_confidence(json.load(stream))
    baseline_images = {row['image']:row for row in archived_images(baseline,
        local_output/'reference/test_labels',local_output/'reference/sortwaste_test.txt')}
    for image in images:
        reference = baseline_images[image['image']]
        assert reference['targets']==image['targets']
        image['baseline_yolo'] = reference['e2e_all']
        image['baseline_pipeline'] = reference['pipeline_all']
    result = summarize(images,systems=('yolo','pipeline','baseline_yolo','baseline_pipeline'))
    result.update(baseline_reproduction_passed=True,selected_epoch=original['selected_epoch'],
                  checkpoint_sha256=original['checkpoint_sha256'],test_list_sha256=original['test_list_sha256'])
    archived = local_output/'evaluation/job_metrics_before_protocol_reconciliation.json'
    if not archived.exists():
        archived.write_text(json.dumps(original,indent=2)+'\n')
    path.write_text(json.dumps(result,indent=2)+'\n')
    rows = [{'system':system,**row} for system,value in result['systems'].items() for row in value['per_class']]
    with (local_output/'evaluation/per_class.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    note = {'reason':'Use the archived test protocol, verified against all five baseline systems.',
            'baseline_values_reproduced':verification['verified_values'],
            'validation_selection_unchanged':True,
            'validation_scorer':'Training-job scorer: hybrid-score threshold 0.25; no carton-overlap ignore.',
            'test_scorer':result['protocol'],
            'model_training_or_checkpoint_selection_repeated':False}
    (local_output/'evaluation/protocol_reconciliation.json').write_text(json.dumps(note,indent=2)+'\n')
    return result
