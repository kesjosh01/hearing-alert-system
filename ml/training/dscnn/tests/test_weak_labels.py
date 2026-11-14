from __future__ import annotations

import unittest

import numpy as np

from dscnn_baseline.common import ROOT, config, source
from dscnn_baseline.weak_labels import FileBagSampler, bag_arrays, require_file_label_approval


class FileLabelTests(unittest.TestCase):
    def test_variable_duration_files_keep_their_own_equal_window_means(self):
        first = np.array([[.2, .7, .1], [.4, .5, .1]], dtype=np.float32)
        second = np.tile([.8, .1, .1], (11, 1)).astype(np.float32)
        x, weights = bag_arrays([first, second])
        np.testing.assert_allclose(weights @ x, [first.mean(0), second.mean(0)], atol=1e-6)
        np.testing.assert_allclose(weights.sum(1), [1, 1], atol=1e-6)
        self.assertEqual(x.shape[0], 13)
        with self.assertRaises(ValueError):
            bag_arrays([np.zeros((0, 3), np.float32)])

    def test_sampler_retains_source_probabilities_and_never_labels_windows(self):
        _, protocol = config()
        descriptors = [("fire_alarm", "domestic_target"), ("doorbell", "domestic_target"),
                       ("other", "overseas_general"), ("other", "overseas_confuser"), ("other", "domestic_doorlock")]
        rows = [dict(sample_id=str(i), split_group_id=str(i), label=label, label_id=source.LABELS[label],
                     split="train", sampler_pool=pool) for i,(label,pool) in enumerate(descriptors)]
        features = {str(i): np.full((i+1,98,40,1),i,np.float32) for i in range(5)}
        sampler = FileBagSampler(rows,features,protocol,1,42)
        np.testing.assert_allclose(sampler.probabilities,[1/3,1/3,.6/3,.3/3,.1/3])
        count=0
        for x,w,y,ids in sampler.draw_bags(30,6):
            self.assertEqual(len(y),len(ids))
            self.assertLessEqual(len(y),6)
            self.assertEqual(w.shape,(len(y),len(x)))
            np.testing.assert_allclose(w.sum(1),1,atol=1e-6)
            for i,sample in enumerate(ids):
                self.assertEqual(int(np.count_nonzero(w[i])),len(features[sample]))
                self.assertEqual(int(y[i]),int(rows[int(sample)]["label_id"]))
            count+=len(ids)
        self.assertEqual(count,30)

    def test_real_approval_explicitly_excludes_interval_claims(self):
        value=require_file_label_approval(ROOT/"data/file_label_approval_20261114.json")
        self.assertEqual(value["mode"],"file_labels")
        self.assertFalse(value["window_labels_assigned"])
        self.assertFalse(value["event_intervals_used"])


if __name__=="__main__":
    unittest.main()
