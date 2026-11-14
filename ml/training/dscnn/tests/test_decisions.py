from __future__ import annotations

import unittest
import numpy as np

from dscnn_baseline import decisions
from dscnn_baseline.common import source


def fixture(suffix=""):
    entries=[("fire","fire_alarm","ko_target",[.7,.1,.2]),
             ("bell","doorbell","ko_target",[.05,.9,.05]),
             ("lock","other","ko_target",[.55,.1,.35]),
             ("foreign","other","overseas",[.6,.1,.3])]
    return [source.prediction_row(dict(sample_id=name+suffix,split_group_id=name+suffix,
             label=label,label_id=source.LABELS[label],target_domain=domain,
             source_dataset="synthetic",split="validation"),p,1) for name,label,domain,p in entries]


class DecisionTests(unittest.TestCase):
    def test_rejection_changes_decisions_and_preserves_raw_probabilities(self):
        rows=fixture()
        result=decisions.apply(rows,.65)
        self.assertEqual([r["predicted_label"] for r in result],["fire_alarm","doorbell","other","other"])
        self.assertEqual(result[2]["p_fire_alarm"],.55)
        self.assertEqual(rows[2]["predicted_label"],"fire_alarm")
        for bad in (float("nan"),-1,1.1):
            with self.assertRaises(ValueError):decisions.apply(rows,bad)

    def test_confusion_accuracy_and_fpr_use_the_same_declared_policy(self):
        m=decisions.report(fixture(),.65)
        self.assertEqual(m["clips"]["accuracy"],1)
        self.assertEqual(m["domestic_target_macro_recall"],1)
        self.assertEqual(m["overseas_other_clip_fpr"],0)
        self.assertEqual(m["domestic_doorlock_false_positive_groups"],0)
        raw_loss=source.report(fixture())["groups"]["loss"]
        self.assertAlmostEqual(m["groups"]["loss"],raw_loss)

    def test_group_threshold_is_applied_after_equal_file_mean(self):
        rows=fixture()[:1]
        first=dict(rows[0],sample_id="one",p_fire_alarm=.95,p_doorbell=.02,p_other=.03)
        second=dict(rows[0],sample_id="two",p_fire_alarm=.5,p_doorbell=.1,p_other=.4)
        result=decisions.aggregate([first,second],.8)
        self.assertEqual(result[0]["predicted_label"],"other")
        self.assertAlmostEqual(result[0]["p_fire_alarm"],.725)

    def test_threshold_is_selected_only_on_development_validation(self):
        guards={"max_overseas_other_clip_fpr":.05,"max_domestic_doorlock_false_positive_groups":0}
        threshold,m,_=decisions.calibrate(fixture(),guards,[0,.65,.8])
        self.assertEqual(threshold,.65)
        self.assertEqual(m["domestic_target_macro_recall"],1)
        for role in ("train","test"):
            bad=fixture();bad[0]["split"]=role
            with self.assertRaisesRegex(ValueError,"validation only"):
                decisions.calibrate(bad,guards,[0,.65])

    def test_one_common_threshold_enforces_every_fold_guard(self):
        guards={"max_overseas_other_clip_fpr":.05,"max_domestic_doorlock_false_positive_groups":0}
        folds=[fixture(str(i)) for i in range(4)]
        threshold,metrics,pooled=decisions.calibrate_joint(folds,guards,[0,.65,.8])
        self.assertEqual(threshold,.65)
        self.assertTrue(all(source.checkpoint_rank(m,guards)[0] for m in metrics))
        self.assertEqual(pooled["domestic_groups"]["count"],12)
        with self.assertRaisesRegex(ValueError,"repeat"):
            decisions.calibrate_joint([fixture()]*4,guards,[0,.65])


if __name__=="__main__":unittest.main()
