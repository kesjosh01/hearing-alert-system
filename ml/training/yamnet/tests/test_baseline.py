import copy
import tempfile
import unittest
import wave
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sound_baseline.core import (ROOT, LABELS, GroupSampler, aggregate_groups,
                                 checkpoint_rank, load_splits, prediction_row, report, settings)
from sound_baseline.features import FeatureStore, decode_audio, read_wave


def row(sample_id, label, pool, role="train", domain="general", group=None):
    return {"sample_id": sample_id, "label": label, "label_id": str(LABELS[label]),
            "sampler_pool": pool, "split": role, "split_group_id": group or sample_id,
            "target_domain": domain, "source_dataset": "fixture"}


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.baseline, self.protocol = settings()
        self.rows = [row("fire_ko", "fire_alarm", "domestic_target", domain="ko_target"),
                     row("fire_aux", "fire_alarm", "auxiliary_target"),
                     row("bell_ko", "doorbell", "domestic_target", domain="ko_target"),
                     row("bell_aux", "doorbell", "auxiliary_target"),
                     row("general", "other", "overseas_general"),
                     row("confuser", "other", "overseas_confuser"),
                     row("lock", "other", "domestic_doorlock", domain="ko_target"),
                     row("heldout", "fire_alarm", "domestic_target", role="test", domain="ko_target")]
        self.features = {item["sample_id"]: np.zeros((100 if item["sample_id"] == "fire_aux" else 1, 1024), np.float32) for item in self.rows}

    def test_actual_draws_preserve_domain_ratio_despite_hundredfold_patch_count(self):
        sampler = GroupSampler(self.rows, self.features, self.protocol, seed=19)
        counts = Counter()
        for _ in range(100):
            _, _, selected = sampler.draw(300)
            counts.update(selected)
        self.assertNotIn("heldout", counts)
        self.assertAlmostEqual(counts["fire_ko"] / counts["fire_aux"], 4, delta=0.35)
        self.assertAlmostEqual((counts["fire_ko"] + counts["fire_aux"]) / 30000, 1 / 3, delta=0.02)
        total_other = sum(counts[key] for key in ("general", "confuser", "lock"))
        for key, expected in (("general", .6), ("confuser", .3), ("lock", .1)):
            self.assertAlmostEqual(counts[key] / total_other, expected, delta=.02)

    def test_empty_negative_pool_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Empty other"):
            GroupSampler([item for item in self.rows if item["sample_id"] != "lock"], self.features, self.protocol)

    def test_real_splits_and_default_sampler_match_frozen_csv(self):
        from sound_baseline.training import verify_sampler_csv
        folds, _ = load_splits(self.protocol)
        self.assertEqual(len(folds), 4)
        features = {item["sample_id"]: np.zeros((1, 1024), np.float32) for item in folds[0] if item["split"] == "train"}
        sampler = GroupSampler(folds[0], features, self.protocol)
        verify_sampler_csv(sampler, self.protocol, 0)


class MetricTests(unittest.TestCase):
    def test_group_aggregation_weights_clips_equally(self):
        first = row("a", "other", "overseas_general", group="shared")
        second = row("b", "other", "overseas_general", group="shared")
        predictions = [prediction_row(first, [.9, .05, .05], 100), prediction_row(second, [.1, .05, .85], 1)]
        combined = aggregate_groups(predictions)[0]
        self.assertAlmostEqual(combined["p_fire_alarm"], .5)
        self.assertEqual(report(predictions)["clips"]["count"], 2)
        self.assertEqual(report(predictions)["groups"]["count"], 1)

    def test_metrics_and_feasibility_guard(self):
        rows = [row("fire", "fire_alarm", "domestic_target", domain="ko_target"),
                row("bell", "doorbell", "domestic_target", domain="ko_target"),
                row("lock", "other", "domestic_doorlock", domain="ko_target"),
                row("foreign", "other", "overseas_general")]
        probabilities = [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.1, .1, .8]]
        metrics = report([prediction_row(item, value, 1) for item, value in zip(rows, probabilities)])
        self.assertEqual(metrics["domestic_target_macro_recall"], 1)
        self.assertEqual(metrics["groups"]["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        bad = copy.deepcopy(metrics)
        bad["overseas_other_clip_fpr"] = 1.0
        self.assertGreater(checkpoint_rank(metrics, settings()[0]["validation_guards"]), checkpoint_rank(bad, settings()[0]["validation_guards"]))


class AudioCacheTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".cache/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_signal(self, path, frequency=440):
        time = np.arange(44100) / 44100
        signal = (np.sin(2 * np.pi * frequency * time) * .2 * 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setparams((2, 2, 44100, len(signal), "NONE", "not compressed"))
            handle.writeframes(np.column_stack((signal, signal)).astype("<i2").tobytes())

    def test_real_ffmpeg_downmix_resample_and_amplitude(self):
        source, destination = self.directory / "한글 stereo.wav", self.directory / "mono.wav"
        self.write_signal(source)
        decode_audio(source, destination)
        result = read_wave(destination)
        self.assertAlmostEqual(len(result), 16000, delta=1)
        self.assertLess(float(np.max(np.abs(result))), .3)
        self.assertGreater(float(np.max(np.abs(result))), .15)

    def test_changed_source_invalidates_embedding_cache(self):
        source = self.directory / "source.wav"
        self.write_signal(source)
        baseline, _ = settings()
        baseline["processed_dir"] = str(self.directory / "processed")
        store = FeatureStore(baseline)
        sample = {"sample_id": "cache-test", "relative_path": str(source)}
        with patch.object(store, "embed_waveform", return_value=np.ones((2, 1024), dtype=np.float32)) as embed:
            store.extract(sample)
            store.extract(sample)
            self.assertEqual(embed.call_count, 1)
            self.write_signal(source, 660)
            store.extract(sample)
            self.assertEqual(embed.call_count, 2)
        np.testing.assert_array_equal(store.load(sample), np.ones((2, 1024), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
