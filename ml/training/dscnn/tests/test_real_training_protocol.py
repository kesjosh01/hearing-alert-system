from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dscnn_baseline.common import source
from dscnn_baseline.data import development_identity, development_refit_rows, development_splits
from dscnn_baseline.review import accept, merge_intervals, require_review_receipt
from dscnn_baseline.study import choose_weight
from dscnn_baseline.training import predict_clips


class ExtensionTests(unittest.TestCase):
    def test_new_sources_have_exactly_once_oof_and_cannot_enter_old_holdout(self):
        folds = development_splits()
        new_ids = {f"kr-user-db-{i:02}" for i in range(6, 11)}
        inventory = {r["sample_id"] for r in folds[0]}
        self.assertEqual(len(inventory), 1640)
        for sample in new_ids:
            self.assertEqual(sum(next(r for r in rows if r["sample_id"] == sample)["split"] == "validation"
                                 for rows in folds), 1)
        for rows in folds:
            self.assertEqual({r["sample_id"] for r in rows}, inventory)
            self.assertNotIn("test", {r["split"] for r in rows})
        refit = development_refit_rows()
        self.assertEqual({r["sample_id"] for r in refit}, inventory)
        self.assertTrue(all(r["split"] == "train" for r in refit if r["sample_id"] in new_ids))

    def test_human_receipt_is_required_without_importing_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                require_review_receipt(Path(directory) / "candidate.csv")


class ReviewTests(unittest.TestCase):
    def test_unreviewed_machine_suggestions_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = {"sample_id": "synthetic-test", "source_sha256": "a" * 64,
                    "label": "doorbell", "duration_s": 1.0}
            inventory = {"development_identity": {"test_fixture": True}, "clips": [item]}
            inventory["inventory_sha256"] = source.fingerprint(inventory)
            decisions = {"inventory_sha256": inventory["inventory_sha256"], "reviewer": "test-fixture",
                         "reviewed_at": "2026-11-14", "clips": [dict(item, intervals=[[0, 1]], reviewed=False)]}
            (root / "inventory.json").write_text(json.dumps(inventory))
            (root / "decisions.json").write_text(json.dumps(decisions))
            with patch("dscnn_baseline.review.development_identity", return_value={"test_fixture": True}):
                with self.assertRaisesRegex(ValueError, "Human event review incomplete"):
                    accept(root / "decisions.json", root / "inventory.json", root / "approved.csv")
            self.assertFalse((root / "approved.csv").exists())

    def test_intervals_are_merged_and_clipped_to_recording(self):
        self.assertEqual(merge_intervals([(-1, .5), (.6, 2), (4, 6)], 5), [[0, 2], [4, 5]])


class SelectionTests(unittest.TestCase):
    @staticmethod
    def summary(weight, recalls, feasible=True):
        return {"domestic_weight": weight,
                "folds": [{"validation_guards_passed": feasible, "domestic_target_recall": r} for r in recalls],
                "oof_metrics": {"domestic_groups": {"macro_f1": .8,
                    "classes": {label: {"precision": .8} for label in ("fire_alarm", "doorbell")}}}}

    def test_no_consistent_gain_keeps_smaller_weight(self):
        a = self.summary(1, [.8] * 4)
        b = self.summary(2, [.9, .9, .8, .8])
        c = self.summary(4, [.9, .9, .9, .7])
        self.assertEqual(choose_weight([a, b, c])["domestic_weight"], 1)

    def test_consistent_gain_and_guard_requirement(self):
        a = self.summary(1, [.8] * 4)
        b = self.summary(2, [.9, .9, .9, .8])
        self.assertEqual(choose_weight([a, b])["domestic_weight"], 2)
        with self.assertRaisesRegex(ValueError, "No weight passed"):
            choose_weight([self.summary(1, [.99] * 4, feasible=False)])


class AggregationTests(unittest.TestCase):
    def test_batching_preserves_equal_file_group_aggregation(self):
        class Tensor:
            def __init__(self, value): self.value = value
            def numpy(self): return self.value
        class FakeModel:
            def __call__(self, x, training=False): return Tensor(np.asarray(x, dtype=np.float32))
        rows = [dict(sample_id=str(i), split_group_id="same-source", label="doorbell", label_id=1,
                     target_domain="ko_target", source_dataset="synthetic", split="validation") for i in range(2)]
        features = {"0": np.tile([.1, .8, .1], (129, 1)), "1": np.array([[.8, .1, .1]])}
        predictions = predict_clips(FakeModel(), rows, features)
        grouped = source.aggregate_groups(predictions)
        self.assertAlmostEqual(grouped[0]["p_fire_alarm"], .45, places=5)
        self.assertAlmostEqual(grouped[0]["p_doorbell"], .45, places=5)


if __name__ == "__main__":
    unittest.main()
