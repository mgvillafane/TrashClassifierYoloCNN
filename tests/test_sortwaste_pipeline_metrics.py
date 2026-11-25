"""Regression checks for the archived detector/head evaluation protocol."""
import unittest

from evaluation.sortwaste_legacy_metrics import prepare, score, summarize


def prediction(box,label,confidence=.9,detector_confidence=.9):
    return dict(box=box,class_id=label,confidence=confidence,detector_confidence=detector_confidence)


class SortWastePipelineMetricsTests(unittest.TestCase):
    def test_unmatched_carton_overlaps_are_ignored(self):
        a,b = [0,0,.4,.4],[.6,.6,1,1]
        images = [dict(targets=[dict(box=a,class_id=4),dict(box=b,class_id=6)],
                       pipeline=[prediction(a,4),prediction(b,4)])]
        result = score(prepare(images,'pipeline'))['per_class'][2]
        self.assertEqual(result['predicted_boxes'],1)
        self.assertEqual(result['precision'],1.)
        self.assertEqual(result['ap50'],1.)

    def test_correct_shared_match_is_retained_even_when_it_overlaps_carton(self):
        box = [0,0,1,1]
        images = [dict(targets=[dict(box=box,class_id=4),dict(box=box,class_id=6)],
                       pipeline=[prediction(box,4)])]
        result = score(prepare(images,'pipeline'))['per_class'][2]
        self.assertEqual(result['true_positives'],1)
        self.assertEqual(result['recall'],1.)

    def test_precision_uses_detector_threshold_for_every_head(self):
        box = [0,0,1,1]
        images = [dict(targets=[dict(box=box,class_id=4)],
                       pipeline=[prediction(box,4,confidence=.1,detector_confidence=.9)])]
        result = score(prepare(images,'pipeline'))['per_class'][2]
        self.assertEqual(result['predicted_boxes'],1)
        self.assertEqual(result['f1'],1.)

    def test_image_bootstrap_preserves_identical_perfect_predictions(self):
        boxes = [[0,0,.2,.2],[.4,.4,.6,.6],[.8,.8,1,1]]
        targets = [dict(box=box,class_id=label) for box,label in zip(boxes,[0,2,4])]
        predictions = [prediction(box,label) for box,label in zip(boxes,[0,2,4])]
        images = [dict(targets=targets,pipeline=predictions) for _ in range(3)]
        result = summarize(images,systems=('pipeline',),repetitions=20)['systems']['pipeline']
        self.assertEqual(result['macro_f1'],1.)
        self.assertEqual(result['intervals_95']['macro_f1'],[1.,1.])


if __name__=='__main__':
    unittest.main()
