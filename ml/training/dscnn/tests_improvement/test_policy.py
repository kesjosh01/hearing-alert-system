import unittest
import numpy as np

from dscnn_baseline.common import source
from improvement import policy
from improvement.protocol import verify_teacher


def row(sample, label, probabilities, group=None, domain="ko_target", split="validation"):
    return dict(sample_id=sample, split_group_id=group or sample, label=label,
                label_id=source.LABELS[label], target_domain=domain, split=split,
                source_dataset="synthetic", patch_count=1,
                **{f"p_{label}": p for label, p in zip(source.LABELS, probabilities)})


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("f", "fire_alarm", [.65, .1, .25]), row("b", "doorbell", [.1, .55, .35]),
                     row("l", "other", [.4, .1, .5]), row("o", "other", [.55, .1, .35], domain="general")]

    def test_target_thresholds_are_independent(self):
        actual = policy.apply(self.rows, {"fire_alarm": .6, "doorbell": .5})
        self.assertEqual([r["predicted_label"] for r in actual], ["fire_alarm", "doorbell", "other", "other"])
        self.assertEqual(actual[0]["p_fire_alarm"], .65)

    def test_fast_ranking_metrics_match_saved_prediction_metrics(self):
        for fire, bell in [(0, 0), (.6, .5), (1, 1)]:
            thresholds = {"fire_alarm": fire, "doorbell": bell}
            a, b = policy.fast_report(policy.prepare(self.rows), thresholds), policy.report(self.rows, thresholds)
            for key in ["domestic_target_macro_recall", "overseas_other_clip_fpr", "domestic_doorlock_false_positive_groups"]:
                self.assertAlmostEqual(a[key], b[key])
            self.assertAlmostEqual(a["domestic_groups"]["macro_f1"], b["domestic_groups"]["macro_f1"])
            self.assertAlmostEqual(a["groups"]["loss"], b["groups"]["loss"])

    def test_source_decision_happens_after_probability_mean(self):
        rows = [row("b1", "doorbell", [.0, .9, .1], "same"), row("b2", "doorbell", [.0, .3, .7], "same")]
        grouped = policy.apply(source.aggregate_groups(rows), {"fire_alarm": .7, "doorbell": .55})
        self.assertEqual(grouped[0]["predicted_label"], "doorbell")

    def test_test_and_train_cannot_calibrate_thresholds(self):
        for split in ["train", "test"]:
            with self.assertRaises(ValueError):
                policy.calibrate([[dict(r, split=split) for r in self.rows]], source.settings()[0]["validation_guards"])

    def test_duplicate_domestic_oof_is_rejected(self):
        with self.assertRaises(ValueError):
            policy.calibrate([self.rows] * 4, source.settings()[0]["validation_guards"])

    def test_calibration_can_retain_bell_while_rejecting_foreign_fire(self):
        thresholds, metrics, pooled, rank = policy.calibrate([self.rows], source.settings()[0]["validation_guards"])
        self.assertTrue(rank[0])
        self.assertEqual(metrics[0]["domestic_target_macro_recall"], 1.0)
        self.assertEqual(metrics[0]["overseas_other_clip_fpr"], 0.0)
        self.assertGreater(thresholds["fire_alarm"], .55)
        self.assertLessEqual(thresholds["doorbell"], .55)

    def test_invalid_thresholds_fail(self):
        for threshold in [-1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                policy.decide(np.array([[.2, .3, .5]]), {"fire_alarm": threshold, "doorbell": .5})

    def test_actual_fold_teachers_exclude_student_validation(self):
        for fold in range(4):
            self.assertFalse(verify_teacher(fold)["validation_group_leakage"])


if __name__ == "__main__":
    unittest.main()
