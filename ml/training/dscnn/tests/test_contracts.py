from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dscnn_baseline.common import config, source
from dscnn_baseline.data import (ANNOTATION_FIELDS, FeatureStore, WindowView, positive_window_indices,
                                 read_annotations, require_development)
from dscnn_baseline.frontend import LogMel, quantize, rfft_power_float32, window_starts, windows


class FrontendTests(unittest.TestCase):
    def test_float32_fft_against_independent_numpy_fft(self):
        frames = np.random.default_rng(42).normal(0, 0.3, (10, 400)).astype(np.float32)
        actual = rfft_power_float32(frames)
        reference = np.abs(np.fft.rfft(frames, n=512)) ** 2
        relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        self.assertLess(relative_l2, 1e-6)
        self.assertGreaterEqual(float(actual.min()), 0)

    def test_silence_shape_and_floor(self):
        result = LogMel()(np.zeros(16000, dtype=np.int16))
        self.assertEqual(result.shape, (98, 40, 1))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, np.log(1e-6), atol=1e-6, rtol=0)

    def test_strict_input_contract(self):
        for value in (np.zeros(16000), np.zeros(15999, dtype=np.int16), np.zeros((16000, 1), dtype=np.int16)):
            with self.assertRaises(ValueError):
                LogMel()(value)

    def test_amplitude_is_not_peak_normalized(self):
        t = np.arange(16000) / 16000
        a = np.rint(1000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
        frontend = LogMel()
        low, high = frontend(a), frontend(a * 2)
        peak_bin = int(np.argmax(low.mean(axis=0)))
        np.testing.assert_allclose(high[:, peak_bin, 0] - low[:, peak_bin, 0], np.log(4), atol=2e-5, rtol=0)

    def test_window_coverage_and_padding(self):
        self.assertEqual(window_starts(8000), [0])
        self.assertEqual(window_starts(16000), [0])
        self.assertEqual(window_starts(25000), [0, 8000, 9000])
        pcm = np.arange(8000, dtype=np.int16)
        start, padded = next(windows(pcm))
        self.assertEqual(start, 0)
        np.testing.assert_array_equal(padded[:8000], pcm)
        self.assertTrue(np.all(padded[8000:] == 0))
        with self.assertRaises(ValueError):
            list(windows(np.zeros(0, dtype=np.int16)))

    def test_no_stft_end_padding(self):
        first = np.zeros(16000, dtype=np.int16)
        second = first.copy()
        second[-80:] = 32767
        np.testing.assert_array_equal(LogMel()(first), LogMel()(second))

    def test_quantization_rounding_clipping(self):
        values = np.array([-200, -1.5, -0.5, 0.5, 1.5, 200], dtype=np.float32)
        np.testing.assert_array_equal(quantize(values, 1, 0), [-128, -2, -1, 1, 2, 127])
        with self.assertRaises(ValueError):
            quantize([np.nan], 1, 0)
        with self.assertRaises(ValueError):
            quantize([0], 0, 0)


class SplitAndAnnotationTests(unittest.TestCase):
    def test_fixed_test_rejected_before_audio_read(self):
        row = {"sample_id": "held-out", "split": "test"}
        with self.assertRaisesRegex(ValueError, "Fixed test"):
            require_development(row)
        with self.assertRaisesRegex(ValueError, "Fixed test"):
            FeatureStore().load(row)

    def test_short_event_and_unannotated_windows(self):
        self.assertEqual(positive_window_indices([0, 8000, 16000], [(0.2, 0.3)]), [0])
        self.assertEqual(positive_window_indices([0, 8000, 16000], [(0.7, 0.9)]), [0, 1])
        self.assertEqual(positive_window_indices([0, 8000, 16000], [(0.1, 2.1)]), [0, 1, 2])
        self.assertEqual(positive_window_indices([0], []), [])
        with self.assertRaises(ValueError):
            positive_window_indices([0], [(0, 1)], 1.1)

    def test_annotation_hash_label_and_holdout_guards(self):
        row = {"sample_id": "dev", "split": "train", "label": "doorbell", "relative_path": "not-read.wav"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.csv"
            def write(sample="dev", label="doorbell", digest="a" * 64):
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(ANNOTATION_FIELDS)
                    writer.writerow([sample, 0, 0.2, label, digest])
            with patch.object(source, "sha256", return_value="a" * 64):
                write()
                self.assertEqual(read_annotations(path, [row]), {"dev": [(0, 0.2)]})
                write(digest="b" * 64)
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    read_annotations(path, [row])
                write(label="fire_alarm")
                with self.assertRaisesRegex(ValueError, "original target class"):
                    read_annotations(path, [row])
                write(sample="test")
                with self.assertRaisesRegex(ValueError, "held-out"):
                    read_annotations(path, [row])

    def test_unannotated_positive_is_blocked(self):
        row = {"sample_id": "dev", "split": "train", "label": "fire_alarm"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty.csv"
            path.write_text(",".join(ANNOTATION_FIELDS) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reviewed event intervals required"):
                read_annotations(path, [row])

    def test_window_view_does_not_include_unselected_windows(self):
        full = np.arange(20).reshape(5, 4)
        view = WindowView(full, [1, 4])
        self.assertEqual(len(view), 2)
        np.testing.assert_array_equal(view[1], full[4])
        np.testing.assert_array_equal(view[:], full[[1, 4]])

    def test_original_sampler_supports_spectrogram_windows(self):
        _, protocol = config()
        descriptors = [("fire_alarm", "domestic_target"), ("doorbell", "domestic_target"),
                       ("other", "overseas_general"), ("other", "overseas_confuser"), ("other", "domestic_doorlock")]
        rows = [{"sample_id": str(i), "split_group_id": str(i), "label": label, "label_id": source.LABELS[label],
                 "split": "train", "sampler_pool": pool} for i, (label, pool) in enumerate(descriptors)]
        features = {str(i): np.zeros((2, 98, 40, 1), dtype=np.float32) for i in range(5)}
        sampler = source.GroupSampler(rows, features, protocol, domestic_weight=1, seed=42)
        x, y, ids = sampler.draw(30)
        self.assertEqual(x.shape, (30, 98, 40, 1))
        self.assertEqual(y.shape, (30,))
        self.assertEqual(len(ids), 30)
        np.testing.assert_allclose(sampler.probabilities, [1/3, 1/3, 0.6/3, 0.3/3, 0.1/3])


if __name__ == "__main__":
    unittest.main()
