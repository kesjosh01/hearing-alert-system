"""Guard corrected labels, source separation and actual group sampling mass."""
from collections import Counter
import unittest

import numpy as np

from dscnn_baseline.common import config, source
from dscnn_baseline.weak_labels import FileBagSampler
from expansion.data import read_rows, validate


class ExpansionDataTests(unittest.TestCase):
    def test_train_and_validation_sirens_have_consistent_positive_labels(self):
        for fold in range(4):
            rows = read_rows(fold)
            sirens = [r for r in rows if r["label_basis"] == "user_explicit_siren_positive_policy"]
            self.assertEqual(Counter(r["split"] for r in sirens), {"train": 24, "validation": 8})
            self.assertTrue(all(r["label"] == "fire_alarm" and int(r["label_id"]) == 0 and r["sampler_pool"] == "auxiliary_target" for r in sirens))

    def test_fixed_test_cannot_enter_the_pipeline(self):
        rows = read_rows(0)
        with self.assertRaisesRegex(ValueError, "fixed test"):
            validate([dict(rows[0], split="test")])
        parent, _ = source.load_splits(config()[1])
        held = {r["sample_id"] for r in parent[0] if r["split"] == "test"}
        self.assertFalse(held.intersection(r["sample_id"] for r in rows))

    def test_new_public_sources_are_train_only(self):
        for fold in range(4):
            public = [r for r in read_rows(fold) if r["audio_root"] == "dscnn"]
            self.assertEqual(len(public), 39)
            self.assertEqual({r["split"] for r in public}, {"train"})
            self.assertEqual(Counter(r["label"] for r in public), {"doorbell": 30, "fire_alarm": 9})

    def test_actual_sampler_keeps_class_balance_and_domestic_weight_four(self):
        rows = read_rows()
        fake = {r["sample_id"]: np.zeros((1, 2), np.float32) for r in rows}
        sampler = FileBagSampler(rows, fake, config()[1], 4, 42)
        probabilities = dict(zip(sampler.ids, sampler.probabilities))
        for label in source.LABELS:
            groups = [g for g, members in sampler.groups.items() if members[0]["label"] == label]
            self.assertAlmostEqual(sum(probabilities[g] for g in groups), 1 / 3)
            if label != "other":
                domestic = [probabilities[g] for g in groups if sampler.groups[g][0]["sampler_pool"] == "domestic_target"]
                overseas = [probabilities[g] for g in groups if sampler.groups[g][0]["sampler_pool"] == "auxiliary_target"]
                self.assertTrue(domestic and overseas)
                self.assertTrue(all(np.isclose(p / overseas[0], 4) for p in domestic))


if __name__ == "__main__":
    unittest.main()
