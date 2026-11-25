"""Exercise real YOLO/ResNet joint gradients on CPU before allocating a GPU."""
import json
from pathlib import Path
import random
import tempfile

import torch

from evaluation.sortwaste_joint import initialize, train_mode, crop_loss, save_pair
from evaluation.sortwaste import ResNetClassifier


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    root = Path(__file__).resolve().parents[1]
    detector,classifier,criterion = initialize(
        root/'reports/sortwaste_joint_pipeline_10epochs/reference/yolo_all.pt',
        root/'reports/sortwaste_yolo_resnet_test/archive/assets/resnet_C4_layer4_best.pt','cpu',10)
    train_mode(detector,classifier)
    images = torch.rand(2,3,128,128)
    with torch.no_grad():
        prediction = detector(images)
        from torchvision.ops import nms
        boxes = detector.model[-1]._get_decode_boxes(prediction['one2many']).transpose(1,2) / 128
        boxes = boxes.clamp(0,1)
        scores = prediction['one2many']['scores'].sigmoid().amax(1)
        chosen = []
        for i in range(2):
            ids = scores[i].topk(min(300,len(scores[i]))).indices
            ids = ids[nms(boxes[i,ids],scores[i,ids],.7)]
            areas = (boxes[i,ids,2:]-boxes[i,ids,:2]).prod(-1)
            interior = ((boxes[i,ids] > .001) & (boxes[i,ids] < .999)).sum(-1)
            rank = interior.float() + areas * .01
            chosen.append(ids[rank.argmax()])
        truth = torch.stack([boxes[i,chosen[i]] for i in range(2)])
        xywh = torch.cat(((truth[:,:2]+truth[:,2:])/2,truth[:,2:]-truth[:,:2]),1)
    batch = {'img':images,'bboxes':xywh,'cls':torch.tensor([[4.],[4.]]),
             'batch_idx':torch.arange(2),'eligible':torch.ones(2,dtype=torch.bool)}
    prediction = detector(images)
    classification,count,selected = crop_loss(detector,classifier.model,batch,prediction)
    if count == 0:
        raise RuntimeError('The real model produced no matched smoke-test crops')
    classifier_to_detector = torch.autograd.grad(classification,next(detector.parameters()),retain_graph=True)[0]
    detection,_ = criterion(prediction,batch)
    loss = detection.sum()/2 + classification
    loss.backward()
    result = {'loss':float(loss.detach()),'matched_crops':count,
              'classification_to_yolo_backbone_gradient_norm':float(classifier_to_detector.norm()),
              'yolo_gradient_norm':float(next(detector.parameters()).grad.norm()),
              'resnet_gradient_norm':float(next(classifier.model.parameters()).grad.norm()),
              'crop_coordinate_gradient_norm':float(selected.grad.norm())}
    print(json.dumps(result,indent=2),flush=True)
    assert all(result[key] > 0 for key in result if key.endswith('_norm'))
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory(prefix='joint_smoke_') as directory:
        destination = Path(directory)
        save_pair(detector,classifier,{'run_name':'smoke'},1,destination,'test')
        restored = YOLO(str(destination/'test_yolo.pt'))
        restored.model.eval()
        with torch.inference_mode():
            prediction = restored.model(images)
        assert prediction[0].shape[0] == 2
        resnet = ResNetClassifier(destination/'test_resnet.pt','cpu')
        assert resnet.epoch == 1
        result['checkpoint_pair_reload_verified'] = True
    output = root/'reports/sortwaste_joint_pipeline_10epochs/local_gradient_smoke.json'
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
