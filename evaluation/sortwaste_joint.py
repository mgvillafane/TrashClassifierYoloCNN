"""Joint YOLO26m + C4 ResNet18 adaptation on the official SortWaste train split.

Both networks receive optimizer updates in every batch. Differentiable sampling
of predicted boxes lets the crop classification loss also reach YOLO's
one-to-many box branch and backbone. The standard dual-head detection loss
continues to train the YOLO one-to-one inference head.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import sortwaste as base
    from . import sortwaste_joint_metrics as metrics
else:
    import sortwaste as base
    import sortwaste_joint_metrics as metrics


def differentiable_crops(images, boxes, photo_ids, size=224):
    """Sample normalized xyxy boxes without detaching coordinates from YOLO."""
    axis = (torch.arange(size, device=images.device, dtype=images.dtype) + .5) / size
    x = boxes[:, 0, None] + (boxes[:, 2] - boxes[:, 0])[:, None] * axis
    y = boxes[:, 1, None] + (boxes[:, 3] - boxes[:, 1])[:, None] * axis
    grid = torch.stack((x[:, None, :].expand(-1, size, -1),
                        y[:, :, None].expand(-1, -1, size)), dim=-1) * 2 - 1
    return F.grid_sample(images[photo_ids], grid, mode='bilinear', padding_mode='border', align_corners=False)


def proposal_pairs(boxes, scores, truth, eligible, cap=16):
    """Discrete geometry-only assignment; selected box coordinates retain gradients."""
    from torchvision.ops import box_iou, nms
    with torch.no_grad():
        ids = scores.topk(min(300, len(scores))).indices
        ids = ids[nms(boxes[ids], scores[ids], .7)]
        ious = box_iou(boxes[ids].float(), truth.float())
        candidates = torch.nonzero((ious >= .5) & eligible[None], as_tuple=False)
        if not len(candidates):
            return [], []
        order = ious[candidates[:, 0], candidates[:, 1]].argsort(descending=True)
        seen_predictions, seen_targets, matches = set(), set(), []
        for pred, target in candidates[order].cpu().tolist():
            if pred not in seen_predictions and target not in seen_targets:
                matches.append((int(ids[pred]), target))
                seen_predictions.add(pred)
                seen_targets.add(target)
        if len(matches) > cap:
            matches = random.sample(matches, cap)
    return [p for p, _ in matches], [t for _, t in matches]


def crop_loss(detector, classifier, batch, predictions, max_crops=16):
    branch = predictions.get('one2many', predictions)
    head = detector.model[-1]
    boxes = head._get_decode_boxes(branch).transpose(1, 2)
    if not head.end2end and not head.xyxy:
        boxes = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2), -1)
    height, width = batch['img'].shape[-2:]
    scale = boxes.new_tensor([width, height, width, height])
    boxes = (boxes / scale).clamp(0, 1)
    scores = branch['scores'].detach().sigmoid().amax(dim=1)
    chosen_boxes, photo_ids, labels = [], [], []
    for photo in range(len(batch['img'])):
        indices = batch['batch_idx'] == photo
        truth = batch['bboxes'][indices]
        truth = torch.cat((truth[:, :2] - truth[:, 2:] / 2, truth[:, :2] + truth[:, 2:] / 2), -1)
        classes = batch['cls'][indices].flatten().long()
        eligible = batch['eligible'][indices] & (classes < 6)
        if not len(truth) or not eligible.any():
            continue
        selected, targets = proposal_pairs(boxes[photo], scores[photo], truth, eligible, max_crops)
        if selected:
            chosen_boxes.append(boxes[photo, selected])
            photo_ids.extend([photo] * len(selected))
            labels.extend(classes[targets].tolist())
    if not chosen_boxes:
        return sum(p.sum() * 0 for p in classifier.parameters()), 0, None
    selected_boxes = torch.cat(chosen_boxes)
    if selected_boxes.requires_grad:
        selected_boxes.retain_grad()
    crops = differentiable_crops(batch['img'], selected_boxes,
                                 torch.tensor(photo_ids, device=boxes.device))
    mean = crops.new_tensor([.485, .456, .406])[None, :, None, None]
    std = crops.new_tensor([.229, .224, .225])[None, :, None, None]
    logits = classifier((crops - mean) / std)
    loss = F.cross_entropy(logits, torch.tensor(labels, device=boxes.device))
    return loss, len(labels), selected_boxes


class TrainingPhotos(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import cv2
        record = self.records[index]
        image = cv2.imread(str(record['path']))
        if image is None:
            raise ValueError(record['path'])
        height, width = image.shape[:2]
        ratio = min(640 / height, 640 / width)
        new_width, new_height = round(width * ratio), round(height * ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        left, top = (640 - new_width) // 2, (640 - new_height) // 2
        image = cv2.copyMakeBorder(image, top, 640 - new_height - top,
                                  left, 640 - new_width - left, cv2.BORDER_CONSTANT, value=(114,114,114))
        image = image[:, :, ::-1].copy()
        targets = np.asarray([t['box'] for t in record['targets']], dtype=np.float32).reshape(-1, 4)
        targets[:, [0,2]] = (targets[:, [0,2]] * new_width + left) / 640
        targets[:, [1,3]] = (targets[:, [1,3]] * new_height + top) / 640
        if random.random() < .5:
            image = image[:, ::-1].copy()
            targets[:, [0,2]] = 1 - targets[:, [2,0]]
        image = image.astype(np.float32) / 255
        brightness, contrast = random.uniform(.9,1.1), random.uniform(.9,1.1)
        image = np.clip((image - .5) * contrast + .5, 0, 1) * brightness
        image = np.clip(image, 0, 1)
        xywh = np.concatenate(((targets[:, :2] + targets[:, 2:]) / 2,
                               targets[:, 2:] - targets[:, :2]), axis=1)
        return {'img': torch.from_numpy(image.transpose(2,0,1).copy()),
                'cls': torch.tensor([[t['class_id']] for t in record['targets']], dtype=torch.float32),
                'bboxes': torch.from_numpy(xywh),
                'eligible': torch.tensor(record['eligible'], dtype=torch.bool)}

    @staticmethod
    def collate(items):
        result = {'img': torch.stack([item['img'] for item in items])}
        for key in ['cls','bboxes','eligible']:
            result[key] = torch.cat([item[key] for item in items])
        result['batch_idx'] = torch.cat([torch.full((len(item['cls']),), i, dtype=torch.long)
                                         for i, item in enumerate(items)])
        return result


def stage(root, entries, destination, eligibility=None):
    def one(relative):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, path)
        targets = base.read_labels(root / 'labels' / relative.relative_to('images').with_suffix('.txt'))
        return {'image': relative.as_posix(), 'path': str(path), 'targets': targets,
                'eligible': [eligibility is None or (relative.stem,i) in eligibility for i in range(len(targets))]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, entries))


def save_pair(detector, classifier, config, epoch, output, prefix):
    snapshot = deepcopy(detector).cpu().float()
    snapshot.criterion = None
    torch.save({'model': snapshot, 'epoch': epoch, 'train_args': vars(detector.args),
                'joint_training': config}, output / (prefix + '_yolo.pt'))
    torch.save({'model_state_dict': {key: value.detach().cpu() for key,value in classifier.model.state_dict().items()},
                'config': {**classifier.config, 'experiment': config['run_name'], 'joint_training': config},
                'epoch': epoch}, output / (prefix + '_resnet.pt'))
    del snapshot


def evaluate(yolo_path, classifier, records, output=None):
    from PIL import Image
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    classifier.model.eval()
    images = []
    with torch.inference_mode():
        for number, record in enumerate(records, 1):
            with Image.open(record['path']) as source:
                image = source.convert('RGB')
            width, height = image.size
            prediction = yolo.predict(image, imgsz=640, conf=.01, max_det=100, agnostic_nms=True,
                                      iou=.7, device=classifier.device, verbose=False)[0]
            boxes = prediction.boxes.xyxy.cpu().tolist()
            labels = prediction.boxes.cls.cpu().int().tolist()
            confidences = prediction.boxes.conf.cpu().tolist()
            crops = [image.crop((max(0, math.floor(x1)), max(0, math.floor(y1)),
                                min(width, max(math.floor(x1)+1, math.ceil(x2))),
                                min(height, max(math.floor(y1)+1, math.ceil(y2))))) for x1,y1,x2,y2 in boxes]
            materials = classifier.predict(crops, batch_size=32)
            normalized = [[round(x1/width,5),round(y1/height,5),round(x2/width,5),round(y2/height,5)] for x1,y1,x2,y2 in boxes]
            images.append({'image': record['image'], 'targets': record['targets'],
                'yolo': [{'box': box,'class_id': label,'confidence': confidence}
                          for box,label,confidence in zip(normalized,labels,confidences)],
                'pipeline': [{'box': box,'class_id': label,'confidence': detector_score*probability}
                             for box,detector_score,(label,probability) in zip(normalized,confidences,materials)]})
            if number % 100 == 0 or number == len(records):
                print({'evaluation_images': number,'total': len(records)}, flush=True)
                if output:
                    base.write_json(output / 'status.json', {'state':'testing','images_complete':number,'test_images':len(records)})
    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images


def initialize(yolo_path, resnet_path, device, epochs):
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT
    detector = YOLO(str(yolo_path)).model.float().to(device)
    base.detector_classes(detector.names)
    detector.requires_grad_(True)
    detector.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, 'epochs': epochs})
    classifier = base.ResNetClassifier(resnet_path, device)
    if classifier.config['experiment'] != 'C4' or classifier.epoch != 14:
        raise ValueError('Expected the original selected C4 epoch-14 checkpoint')
    classifier.model.requires_grad_(True)
    criterion = detector.init_criterion()
    return detector, classifier, criterion


def train_mode(detector, classifier):
    detector.train()
    classifier.model.train()
    # Preserve C4's pretrained normalization statistics; affine weights train.
    for module in classifier.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train(args):
    if args.epochs != 10:
        raise ValueError('This experiment requires exactly ten epochs')
    if not torch.cuda.is_available():
        raise RuntimeError('The full training run requires a GPU')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    from ultralytics import settings
    settings.update({'mlflow':False,'wandb':False,'tensorboard':False})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output / 'status.json', {'state':'preparing','epochs_requested':10})
    root = Path(args.data_root)
    splits = base.read_splits(root)
    if {key:len(value) for key,value in splits.items()} != {'train':3705,'val':780,'test':776}:
        raise ValueError('Unexpected official split counts')
    test_hash = base.sha256(root / 'sortwaste_test.txt')
    if test_hash != args.test_hash:
        raise ValueError('Test membership differs from the original comparison')
    hashes = {'yolo':base.sha256(args.yolo),'resnet':base.sha256(args.resnet)}
    config = {'run_name':output.name, 'training':'joint detection plus differentiable-crop cross entropy',
              'source_yolo':args.yolo,'source_resnet':args.resnet,'source_sha256':hashes,
              'epochs_requested':10,'early_stopping':False,'batch_size':4,'image_size':640,'crop_size':224,
              'max_training_crops_per_image':16,'optimizer':'AdamW','yolo_lr':1e-5,
              'resnet_backbone_lr':1e-5,'resnet_head_lr':1e-4,'weight_decay':1e-4,
              'precision':'float32','seed':42,'all_yolo_weights_trainable':True,'all_resnet_weights_trainable':True,
              'resnet_batchnorm_running_statistics':'fixed; affine parameters trainable',
              'classification_gradient_path':'one2many predicted boxes -> grid_sample -> ResNet -> cross entropy; gradient reaches YOLO coordinates and backbone',
              'assignment':'class-independent greedy IoU >= 0.5 after class-agnostic NMS; assignment indices detached',
              'loss':'standard dual-head YOLO detection loss / image count + mean crop cross entropy',
              'training_augmentation':{'horizontal_flip':.5,'brightness':.1,'contrast':.1},
              'checkpoint_selection':'SortWaste validation pipeline macro-F1; mAP50 tiebreaker',
              'split_counts':{s:len(v) for s,v in splits.items()},'test_list_sha256':test_hash,
              'test_used_for_training_or_selection':False,'gpu':torch.cuda.get_device_name(0)}
    eligibility = {s:base.classifier_eligibility(args.classifier_index,splits,partition=s) for s in ['train','val']}
    with tempfile.TemporaryDirectory(prefix='sortwaste_joint_') as temporary:
        local = Path(temporary)
        detector, classifier, criterion = initialize(args.yolo,args.resnet,'cuda:0',10)
        if classifier.config['index_sha256'] != base.sha256(args.classifier_index):
            raise ValueError('Original classifier data index changed')
        initial_detector = next(detector.parameters()).detach().cpu().clone()
        initial_resnet = next(classifier.model.parameters()).detach().cpu().clone()
        train_records = stage(root,splits['train'],local,eligibility['train'])
        val_records = stage(root,splits['val'],local,eligibility['val'])
        dataset = TrainingPhotos(train_records)
        loader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,num_workers=2,
                     pin_memory=True,collate_fn=TrainingPhotos.collate,generator=torch.Generator().manual_seed(42))
        backbone = [p for name,p in classifier.model.named_parameters() if not name.startswith('fc.')]
        optimizer = torch.optim.AdamW([{'params':detector.parameters(),'lr':1e-5},
            {'params':backbone,'lr':1e-5},{'params':classifier.model.fc.parameters(),'lr':1e-4}],weight_decay=1e-4)
        summary = {'config':config,'epochs_completed':0,'history':[]}
        base.write_json(output / 'training_metrics.json',summary)
        best_rank, best_epoch = (-float('inf'),-float('inf')),None
        for epoch in range(1,11):
            started = perf_counter()
            train_mode(detector,classifier)
            total_loss,total_detection,total_classification,seen,crops = 0.,0.,0.,0,0
            for step,batch in enumerate(loader,1):
                batch = {key:value.to('cuda:0',non_blocking=True) for key,value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                predictions = detector(batch['img'])
                detection_loss,_ = criterion(predictions,batch)
                detection_loss = detection_loss.sum() / len(batch['img'])
                classification_loss,count,crop_boxes = crop_loss(detector,classifier.model,batch,predictions)
                loss = detection_loss + classification_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}, step {step}')
                loss.backward()
                if epoch == 1 and step == 1:
                    summary['first_batch_gradient_check'] = {
                        'yolo_first_layer_gradient_norm':float(next(detector.parameters()).grad.norm()),
                        'resnet_first_layer_gradient_norm':float(next(classifier.model.parameters()).grad.norm()),
                        'classification_to_box_gradient_norm':float(crop_boxes.grad.norm()) if crop_boxes is not None else 0.,
                        'matched_crops':count}
                    if count == 0 or min(summary['first_batch_gradient_check'][key] for key in [
                            'yolo_first_layer_gradient_norm','resnet_first_layer_gradient_norm','classification_to_box_gradient_norm']) <= 0:
                        raise RuntimeError('Both networks and differentiable boxes must receive gradients')
                nn.utils.clip_grad_norm_(list(detector.parameters())+list(classifier.model.parameters()),10.)
                optimizer.step()
                n = len(batch['img'])
                total_loss += float(loss.detach()) * n
                total_detection += float(detection_loss.detach()) * n
                total_classification += float(classification_loss.detach()) * n
                seen += n
                crops += count
                del loss,predictions,detection_loss,classification_loss,crop_boxes,batch
                if step % 100 == 0 or step == len(loader):
                    state = {'state':'training','epoch':epoch,'epochs_requested':10,'step':step,
                             'steps_per_epoch':len(loader),'images_seen':seen,'matched_crops':crops,
                             'mean_loss':total_loss/seen}
                    base.write_json(output / 'status.json',state)
                    print(state,flush=True)
            assert seen == 3705
            save_pair(detector,classifier,config,epoch,local,'validation')
            images = evaluate(local/'validation_yolo.pt',classifier,val_records)
            validation = metrics.summarize(images)['systems']
            rank = (validation['pipeline']['macro_f1'],validation['pipeline']['map50'])
            row = {'epoch':epoch,'train_images':seen,'matched_training_crops':crops,'train_loss':total_loss/seen,
                   'detection_loss':total_detection/seen,'classification_loss':total_classification/seen,
                   'val_pipeline_macro_f1':rank[0],'val_pipeline_map50':rank[1],
                   'val_yolo_map50':validation['yolo']['map50'],'seconds':perf_counter()-started}
            summary['history'].append(row)
            if rank > best_rank:
                best_rank,best_epoch = rank,epoch
                shutil.copyfile(local/'validation_yolo.pt',output/'best_yolo.pt')
                shutil.copyfile(local/'validation_resnet.pt',output/'best_resnet.pt')
            shutil.copyfile(local/'validation_yolo.pt',output/'last_yolo.pt')
            shutil.copyfile(local/'validation_resnet.pt',output/'last_resnet.pt')
            summary.update(epochs_completed=epoch,best_epoch=best_epoch,best_validation={'macro_f1':best_rank[0],'map50':best_rank[1]})
            base.write_json(output/'training_metrics.json',summary)
            with (output/'history.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(row))
                writer.writeheader()
                writer.writerows(summary['history'])
            if hasattr(criterion,'update'):
                criterion.update()
            print(row,flush=True)
        summary['weight_updates_verified'] = {
            'yolo_first_layer_changed':not torch.equal(initial_detector,next(detector.parameters()).detach().cpu()),
            'resnet_first_layer_changed':not torch.equal(initial_resnet,next(classifier.model.parameters()).detach().cpu())}
        if not all(summary['weight_updates_verified'].values()):
            raise RuntimeError('Both complete networks must have changed')
        if hashes != {'yolo':base.sha256(args.yolo),'resnet':base.sha256(args.resnet)}:
            raise RuntimeError('A source checkpoint changed')
        summary['source_checkpoints_preserved'] = True
        base.write_json(output/'training_metrics.json',summary)
        del optimizer,criterion,detector,classifier,loader
        torch.cuda.empty_cache()
        # Test pixels and labels are first opened only after all fitting/selection.
        test_records = stage(root,splits['test'],local)
        selected = base.ResNetClassifier(output/'best_resnet.pt','cuda:0')
        evaluation_images = evaluate(output/'best_yolo.pt',selected,test_records,output)
        evaluation_dir = output/'evaluation'
        evaluation_dir.mkdir()
        metrics.save_predictions(evaluation_dir/'predictions.json.gz',evaluation_images)
        baseline = metrics.baseline_reproduction(evaluation_images,
            Path(args.reference_eval)/'predictions_yolo_all.csv.gz',
            Path(args.reference_eval)/'summary.csv',Path(args.reference_eval)/'per_class.csv')
        base.write_json(evaluation_dir/'baseline_reproduction.json',baseline)
        result = metrics.summarize(evaluation_images,ap_method=baseline['selected_ap_method'],repetitions=1000)
        result.update(baseline_reproduction_passed=baseline['passed'],selected_epoch=best_epoch,
                      checkpoint_sha256={name:base.sha256(output/name) for name in ['best_yolo.pt','best_resnet.pt']},
                      test_list_sha256=test_hash)
        base.write_json(evaluation_dir/'metrics.json',result)
        rows = [{'system':system,**row} for system,value in result['systems'].items() for row in value['per_class']]
        with (evaluation_dir/'per_class.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        base.write_json(output/'status.json',{'state':'success' if baseline['passed'] else 'scorer_verification_required',
             'epochs_completed':10,'best_epoch':best_epoch,'test_images':len(evaluation_images),
             'baseline_reproduction_passed':baseline['passed']})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--yolo',required=True)
    parser.add_argument('--resnet',required=True)
    parser.add_argument('--data-root',default=base.DEFAULT_DATA)
    parser.add_argument('--classifier-index',default=base.DEFAULT_INDEX)
    parser.add_argument('--reference-eval',default='/Volumes/workspace/default/models/model_eval')
    parser.add_argument('--test-hash',required=True)
    parser.add_argument('--epochs',type=int,default=10)
    return train(parser.parse_args(argv))


if __name__ == '__main__':
    main()
