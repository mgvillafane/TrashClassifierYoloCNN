"""Evaluate the three requested existing models on the original TACO test split."""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/taco_three_model_transfer'
os.environ.setdefault('YOLO_CONFIG_DIR',str(OUT/'runtime/yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'runtime/matplotlib'))

from . import sortwaste as base

SYSTEMS = ('taco_post','sortwaste_yolo','sortwaste_post')


def calculate(records):
    matched = [(r['truth'],r['prediction']) for r in records if r['matched']]
    counts = {label:sum(r['truth']==label for r in records) for label in base.RESNET_CLASSES}
    rows = []
    for label in base.RESNET_CLASSES:
        tp = sum(r['truth']==label and r['prediction']==label for r in records)
        predicted = sum(r['prediction']==label for r in records)
        support = counts[label]
        rows.append(dict(label=label,support=support,precision=tp/predicted if predicted else 0.,
                         recall=tp/support if support else 0.,f1=2*tp/(predicted+support) if predicted+support else 0.))
    return {'count':sum(counts.values()),'correct':sum(r['truth']==r['prediction'] for r in records if r['truth'] is not None),
            'macro_precision':sum(r['precision'] for r in rows)/6,
            'macro_recall':sum(r['recall'] for r in rows)/6,'macro_f1':sum(r['f1'] for r in rows)/6,
            'weighted_f1':sum(r['f1']*r['support'] for r in rows)/max(sum(counts.values()),1),'per_class':rows}


def run():
    import torch
    from PIL import Image
    from ultralytics import YOLO,settings

    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'predictions.json.gz').exists():
        raise RuntimeError('Existing completed predictions found; rebuild the graph without rerunning inference.')
    data = ROOT/'data/yolo/taco_trashnet'
    expected = '705bb0645f1a40046aa4ceacfe20141a8990e84263092d28b3879b77f391a416'
    assert base.sha256(data/'split_manifest.json')==expected
    splits = base.read_splits(data,'taco')
    assert len(splits['test'])==177
    torch.set_num_threads(4)
    torch.manual_seed(42)
    settings.update({'mlflow':False,'wandb':False,'tensorboard':False})
    definitions = [
        ('taco',ROOT/'app/models/YOLO_full_TACO.pt',OUT/'reference/taco_post_resnet.pt',['taco_post']),
        ('sortwaste',ROOT/'app/models/YOLO26m_SortWaste_7class_20261121.pt',
         ROOT/'app/models/sortwaste_yolo_resnet_post_training.pt',['sortwaste_yolo','sortwaste_post'])]
    records = {system:[] for system in SYSTEMS}
    predictions_export = []
    provenance = {}
    for detector_name,yolo_path,resnet_path,systems in definitions:
        yolo = YOLO(str(yolo_path))
        classifier = base.ResNetClassifier(resnet_path,'cpu')
        assert classifier.config['source_yolo_sha256']==base.sha256(yolo_path)
        assert classifier.config['training_datasets']==[detector_name]
        assert classifier.config['initial_experiment']=='C4'
        provenance[detector_name] = {'yolo':str(yolo_path.relative_to(ROOT)),'yolo_sha256':base.sha256(yolo_path),
            'resnet':str(resnet_path.relative_to(ROOT)),'resnet_sha256':base.sha256(resnet_path),
            'post_training_epochs':10,'selected_epoch':classifier.epoch,'post_training_yolo_frozen':True}
        completed = 0
        cache_path = OUT/(detector_name+'_evaluation_cache.json.gz')
        if cache_path.exists():
            with gzip.open(cache_path,'rt') as stream:
                cache = json.load(stream)
            assert cache['provenance']==provenance[detector_name]
            assert cache['split_manifest_sha256']==expected
            completed = cache['completed_images']
            for system in systems:
                records[system] = cache['records'][system]
            predictions_export.extend(cache['predictions'])
        for number,relative in enumerate(splits['test'],1):
            if number<=completed:
                continue
            with Image.open(data/relative) as source:
                image = source.convert('RGB')
            width,height = image.size
            targets = base.read_labels(data/'labels'/relative.relative_to('images').with_suffix('.txt'))
            prediction = yolo.predict(image,imgsz=640,conf=.35,max_det=300,device='cpu',verbose=False)[0]
            boxes = prediction.boxes.xyxy.cpu().tolist()
            labels = prediction.boxes.cls.cpu().int().tolist()
            normalized = [[x1/width,y1/height,x2/width,y2/height] for x1,y1,x2,y2 in boxes]
            crops = [image.crop((max(0,math.floor(x1)),max(0,math.floor(y1)),
                                min(width,max(math.floor(x1)+1,math.ceil(x2))),
                                min(height,max(math.floor(y1)+1,math.ceil(y2))))) for x1,y1,x2,y2 in boxes]
            alternate = classifier.predict(crops,batch_size=32)
            material_labels = [label for label,_ in alternate]
            matches = base.match_boxes(normalized,[t['box'] for t in targets],.5)
            by_target = {target:prediction for prediction,target,_ in matches}
            matched_predictions = {prediction for prediction,_,_ in matches}
            for system in systems:
                predicted = labels if system=='sortwaste_yolo' else material_labels
                for target_index,target in enumerate(targets):
                    index = by_target.get(target_index)
                    records[system].append({'image':relative.as_posix(),'truth':base.CLASSES[target['class_id']],
                        'prediction':base.CLASSES[predicted[index]] if index is not None else None,'matched':index is not None})
                for index,label in enumerate(predicted):
                    if index not in matched_predictions:
                        records[system].append({'image':relative.as_posix(),'truth':None,'prediction':base.CLASSES[label],'matched':False})
            predictions_export.append({'detector':detector_name,'image':relative.as_posix(),'targets':targets,
                'boxes':normalized,'yolo_labels':labels,'post_resnet_labels':material_labels,
                'yolo_confidence':prediction.boxes.conf.cpu().tolist(),'post_resnet_probability':[p for _,p in alternate]})
            if number%25==0 or number==177:
                status = {'state':'evaluating','detector':detector_name,'images_complete':number,'test_images':177}
                base.write_json(OUT/'status.json',status)
                cache = {'provenance':provenance[detector_name],'split_manifest_sha256':expected,
                         'completed_images':number,'records':{s:records[s] for s in systems},
                         'predictions':[r for r in predictions_export if r['detector']==detector_name]}
                temporary = cache_path.with_suffix('.tmp')
                with gzip.open(temporary,'wt') as stream:
                    json.dump(cache,stream)
                temporary.replace(cache_path)
                print(status,flush=True)
        del yolo,classifier
    result = {'test_images':177,'split_manifest_sha256':expected,'provenance':provenance,
              'protocol':{'image_size':640,'detector_confidence':.35,'max_detections':300,'matching_iou':.5,
                'matching':'class-independent descending IoU one-to-one; identical boxes for the SortWaste pair',
                'classes':list(base.RESNET_CLASSES),'device':'cpu','test_tuning':False},'systems':{}}
    for system in SYSTEMS:
        full = calculate(records[system])
        matched_records = [r for r in records[system] if r['matched']]
        conditional = calculate(matched_records)
        conditional['accuracy'] = conditional['correct']/conditional['count'] if conditional['count'] else 0.
        full['localization_recall'] = len(matched_records)/full['count']
        result['systems'][system] = {'matched_classification':conditional,'full_test_detection':full}
    reference = json.loads((ROOT/'reports/taco_yolo_resnet_test/metrics.json').read_text())
    expected_taco = reference['systems']['yolo_post_resnet']['primary_classifier_eligible']
    measured = result['systems']['taco_post']['matched_classification']
    assert provenance['taco']['resnet_sha256']==reference['config']['post_resnet_sha256']
    assert provenance['taco']['yolo_sha256']==reference['config']['yolo_sha256']
    result['original_taco_model_weights_verified'] = True
    result['original_taco_row_reproduced'] = (measured['count']==expected_taco['count'] and
                                           abs(measured['accuracy']-expected_taco['accuracy'])<1e-12)
    result['original_gpu_vs_fresh_cpu'] = {'original_matched_count':expected_taco['count'],
        'fresh_matched_count':measured['count'],'original_accuracy':expected_taco['accuracy'],
        'fresh_accuracy':measured['accuracy'],'all_new_rows_evaluated_on':'cpu',
        'note':'Fresh inference can differ numerically from the archived GPU run. Checkpoint bytes and test split are identical.'}
    base.write_json(OUT/'metrics.json',result)
    with gzip.open(OUT/'predictions.json.gz','wt') as stream:
        json.dump(predictions_export,stream)
    rows = [{'system':system,'scope':scope,**row} for system,value in result['systems'].items()
            for scope,scores in value.items() for row in scores['per_class']]
    with (OUT/'per_class.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert result['systems']['sortwaste_yolo']['matched_classification']['count']==result['systems']['sortwaste_post']['matched_classification']['count']
    assert len({v['full_test_detection']['count'] for v in result['systems'].values()})==1
    base.write_json(OUT/'status.json',{'state':'success','test_images':177,
                                     'original_taco_model_weights_verified':True})
    print(json.dumps({system:{scope:{k:v for k,v in scores.items() if k!='per_class'}
                     for scope,scores in value.items()} for system,value in result['systems'].items()},indent=2),flush=True)
    return result


if __name__=='__main__':
    run()
