"""Checks for the YOLO experiment scoring (no GPU or YOLO needed)."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_experiments import detection_metrics as dm  # noqa: E402


def image(gt, preds):
    """gt: [(label, box)], preds: [(label, score, box)] with det_conf = score."""
    return {"gt_boxes": np.array([b for _, b in gt], float).reshape(-1, 4), "gt_labels": np.array([l for l, _ in gt]),
            "boxes": np.array([b for _, _, b in preds], float).reshape(-1, 4),
            "labels": np.array([l for l, _, _ in preds]), "scores": np.array([s for _, s, _ in preds], float),
            "det_conf": np.array([s for _, s, _ in preds], float)}


A, B, C = [0, 0, .2, .2], [.5, .5, .7, .7], [.3, .3, .4, .4]


class MatchingTests(unittest.TestCase):
    def test_same_class_overlap_is_true_positive_and_duplicates_are_false(self):
        flags = dm.match([A, A], [4, 4], [.9, .8], [A], [4])
        self.assertEqual(flags.tolist(), [1, 0])

    def test_wrong_material_on_a_real_object_is_a_false_positive(self):
        self.assertEqual(dm.match([A], [0], [.9], [A], [4]).tolist(), [0])

    def test_predictions_on_a_carton_are_ignored_unless_correct(self):
        self.assertEqual(dm.match([B, B], [0, 6], [.9, .8], [B], [6]).tolist(), [-1, 1])

    def test_agnostic_nms_keeps_one_box_per_object(self):
        keep = dm.agnostic_nms(np.array([A, A, B]), np.array([.5, .9, .7]))
        self.assertEqual(sorted(keep.tolist()), [1, 2])


class ScoreTests(unittest.TestCase):
    def test_perfect_detector_scores_one(self):
        ims = [image([(4, A), (0, B)], [(4, .9, A), (0, .8, B)])]
        s = dm.Scored(ims, "labels", "scores")
        rows = s.per_class()
        self.assertAlmostEqual(rows[4]["ap50"], 1.0)
        self.assertAlmostEqual(rows[0]["f1"], 1.0)
        self.assertTrue(np.isnan(rows[2]["ap50"]))  # no metal in the test set: not averaged
        self.assertEqual(s.summary(dm.SHARED3)["classes"], 2)
        self.assertAlmostEqual(s.summary(dm.SHARED3)["map50"], 1.0)

    def test_average_precision_matches_hand_computation(self):
        # 2 true plastic boxes; ranked predictions: TP, FP, TP -> precision envelope 1 to r=.5, 2/3 to r=1
        ims = [image([(4, A), (4, B)], [(4, .9, A), (4, .8, C), (4, .7, B)])]
        ap = dm.Scored(ims, "labels", "scores").per_class()[4]["ap50"]
        expected = (51 * 1 + 50 * (2 / 3)) / 101
        self.assertAlmostEqual(ap, expected, places=6)

    def test_low_confidence_boxes_count_for_ap_but_not_f1(self):
        ims = [image([(4, A)], [(4, .1, A)])]
        row = dm.Scored(ims, "labels", "scores").per_class()[4]
        self.assertAlmostEqual(row["ap50"], 1.0)
        self.assertEqual(row["f1"], 0.0)  # the only true positive is below 0.25

    def test_photo_weights_equal_duplicated_photos(self):
        one = image([(4, A)], [(4, .9, A), (4, .6, B)])
        two = image([(0, B)], [(4, .8, B)])
        weighted = dm.Scored([one, two], "labels", "scores").per_class(np.array([2.0, 1.0]))[4]
        copied = dm.Scored([one, one, two], "labels", "scores").per_class()[4]
        for key in ("ap50", "precision", "recall", "f1", "n_gt"):
            self.assertAlmostEqual(weighted[key], copied[key])

    def test_bootstrap_weights_resample_all_photos(self):
        w = dm.bootstrap_weights(10, repeats=5)
        self.assertEqual(w.shape, (5, 10))
        self.assertTrue((w.sum(axis=1) == 10).all())


if __name__ == "__main__":
    unittest.main()
