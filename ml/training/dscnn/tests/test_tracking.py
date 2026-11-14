from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dscnn_baseline import common, tracking, training
from dscnn_baseline.__main__ import main


def arguments(mode="disabled", **overrides):
    values = dict(wandb_mode=mode, wandb_project="synthetic-project", wandb_entity=None,
                  run_name="synthetic-run", fold=0, device="cpu", epochs=1,
                  annotations=Path("synthetic-annotations.csv"))
    values.update(overrides)
    return types.SimpleNamespace(**values)


def options():
    return dict(seed=42, epochs=1, patience=1, batch_size=1, steps_per_epoch=1,
                learning_rate=0.001, domestic_weight=1, minimum_event_coverage=0.8)


def history(epoch=1):
    return dict(epoch=epoch, train_loss=0.25, domestic_target_recall=0.5,
                overseas_other_fpr=0.1, domestic_doorlock_fp=0,
                validation_guards_passed=True, saved=True)


def fake_sdk(run=None):
    run = run if run is not None else Mock(id="synthetic-id", dir="synthetic-files")
    return types.SimpleNamespace(run=None, Settings=Mock(), init=Mock(return_value=run))


class ModeAndBoundaryTests(unittest.TestCase):
    def test_cli_training_defaults_to_requested_online_project(self):
        command = ["dscnn_baseline", "train", "--annotations", "synthetic.csv",
                   "--fold", "0", "--run-name", "synthetic"]
        with patch.object(sys, "argv", command), patch.object(training, "train_fold") as train:
            main()
        args = train.call_args.args[0]
        self.assertEqual(args.wandb_mode, "online")
        self.assertEqual(args.wandb_project, "hearing_alert_system_CNN")
        self.assertEqual(args.wandb_entity, "jhh0415-korea-university")

    def test_default_runtime_configuration_overrides_inherited_online_mode(self):
        with patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=True):
            common.configure_wandb()
            self.assertEqual(os.environ["WANDB_MODE"], "disabled")
            self.assertEqual(os.environ["WANDB_DISABLED"], "true")

    def test_selected_recording_clears_inherited_disabled_flag(self):
        for mode in ("offline", "online"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"WANDB_DISABLED": "true"}, clear=True):
                common.configure_wandb(mode)
                self.assertEqual(os.environ["WANDB_MODE"], mode)
                self.assertNotIn("WANDB_DISABLED", os.environ)

    def test_invalid_mode_does_not_mutate_environment(self):
        with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=True):
            with self.assertRaises(ValueError):
                common.configure_wandb("invalid")
            self.assertEqual(dict(os.environ), {"WANDB_MODE": "disabled"})

    def test_gpu_selection_guard_still_fails_before_tensorflow_import(self):
        self.assertNotIn("tensorflow", sys.modules)
        with tempfile.TemporaryDirectory() as temporary, patch.object(common, "ROOT", Path(temporary)):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "explicitly selected GPU UUID"):
                    common.runtime("gpu", wandb_mode="offline")
        self.assertNotIn("tensorflow", sys.modules)

    def test_online_requires_explicit_entity_and_session_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "--wandb-entity"):
                tracking.validate_options(arguments("online"))
            with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY"):
                tracking.validate_options(arguments("online", wandb_entity="synthetic-team"))
        with patch.dict(os.environ, {"WANDB_API_KEY": "synthetic-test-value"}, clear=True):
            tracking.validate_options(arguments("online", wandb_entity="synthetic-team"))

    def test_missing_online_credentials_stop_before_data_and_output_creation(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(training, "config", return_value=({"training": options()}, {})):
            with patch.object(training, "development_splits") as splits, patch.object(training, "new_directory") as output:
                with patch.object(training, "runtime") as runtime, patch.object(training, "start_run") as start:
                    with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY"):
                        training.train_fold(arguments("online", wandb_entity="synthetic-team"))
        for operation in (splits, output, runtime, start):
            operation.assert_not_called()

    def test_annotation_failure_precedes_runtime_and_wandb_start(self):
        with patch.object(training, "config", return_value=({"training": options()}, {})):
            with patch.object(training, "development_splits", return_value=[[]]):
                with patch.object(training, "require_review_receipt", return_value={}), patch.object(training, "read_annotations", side_effect=ValueError("synthetic missing annotations")):
                    with patch.object(training, "runtime") as runtime, patch.object(training, "start_run") as start:
                        with self.assertRaisesRegex(ValueError, "synthetic missing annotations"):
                            training.train_fold(arguments("offline"))
        runtime.assert_not_called()
        start.assert_not_called()


class RecordingTests(unittest.TestCase):
    def test_disabled_does_not_import_sdk(self):
        with patch.object(tracking.importlib, "import_module") as sdk_import:
            self.assertIsNone(tracking.start_run(arguments(), options(), Path("unused")))
        sdk_import.assert_not_called()

    def test_offline_start_excludes_unapproved_config_and_automatic_capture(self):
        sdk = fake_sdk()
        selected = options() | {"private_path": "/synthetic-private", "api_key": "synthetic-test-value"}
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"wandb": sdk}):
            run = tracking.start_run(arguments("offline"), selected, Path("synthetic-output"))
        self.assertIs(run, sdk.init.return_value)
        payload = sdk.init.call_args.kwargs
        self.assertEqual(payload["mode"], "offline")
        self.assertEqual(payload["dir"], "synthetic-output")
        self.assertEqual(set(payload["config"]), {"model", "fold", "seed", *tracking.CONFIG_FIELDS})
        self.assertNotIn("private_path", payload["config"])
        self.assertNotIn("api_key", payload["config"])
        settings = sdk.Settings.call_args.kwargs
        for key in ("disable_code", "disable_git", "disable_job_creation",
                    "x_disable_meta", "x_disable_stats", "x_disable_machine_info"):
            self.assertTrue(settings[key])
        self.assertEqual(settings["console"], "off")
        self.assertFalse(settings["save_code"])
        self.assertFalse(settings["x_save_requirements"])
        self.assertEqual(settings["capture_loggers"], {})
        run.define_metric.assert_any_call("*", step_metric="epoch")

    def test_existing_active_run_is_never_reused(self):
        sdk = fake_sdk()
        sdk.run = Mock()
        with patch.dict(sys.modules, {"wandb": sdk}):
            with self.assertRaisesRegex(RuntimeError, "W&B initialization failed"):
                tracking.start_run(arguments("offline"), options(), Path("unused"))
        sdk.init.assert_not_called()

    def test_epoch_is_committed_and_only_approved_fields_are_sent(self):
        run = Mock()
        row = history() | {"audio_path": "/synthetic-private.wav", "api_key": "synthetic-test-value"}
        tracking.log_epoch(run, row)
        run.log.assert_called_once_with(history(), step=1, commit=True)

    def test_sdk_errors_do_not_copy_arbitrary_text_to_training_errors(self):
        sdk = fake_sdk()
        sdk.init.side_effect = ValueError("synthetic-sensitive-text")
        with patch.dict(sys.modules, {"wandb": sdk}):
            with self.assertRaises(RuntimeError) as caught:
                tracking.start_run(arguments("offline"), options(), Path("unused"))
        self.assertNotIn("synthetic-sensitive-text", str(caught.exception))
        run = Mock()
        run.log.side_effect = ValueError("synthetic-sensitive-text")
        with self.assertRaises(RuntimeError) as caught:
            tracking.log_epoch(run, history())
        self.assertNotIn("synthetic-sensitive-text", str(caught.exception))

    def test_setup_failure_finishes_started_run_as_failed(self):
        run = Mock()
        run.define_metric.side_effect = RuntimeError("synthetic setup failure")
        with patch.dict(sys.modules, {"wandb": fake_sdk(run)}):
            with self.assertRaisesRegex(RuntimeError, "W&B initialization failed"):
                tracking.start_run(arguments("offline"), options(), Path("unused"))
        run.finish.assert_called_once_with(exit_code=1)

    def test_finish_distinguishes_success_failure_and_interrupt(self):
        for code in (0, 1, None):
            with self.subTest(exit_code=code):
                run = Mock()
                status = {"exit_code": code, "wandb": {"mode": "offline"}}
                tracking.finish_run(run, status)
                run.finish.assert_called_once_with(exit_code=0 if code == 0 else 1)
                self.assertTrue(status["wandb"]["finished"])

    def test_finish_error_preserves_training_first_fault(self):
        run = Mock()
        run.finish.side_effect = TimeoutError("synthetic-sensitive-text")
        status = {"exit_code": 1, "first_error": "original training fault", "wandb": {}}
        tracking.finish_run(run, status)
        self.assertEqual(status["first_error"], "original training fault")
        self.assertEqual(status["wandb"]["finish_error_type"], "TimeoutError")
        self.assertNotIn("synthetic-sensitive-text", json.dumps(status))


class MockTrainingIntegrationTests(unittest.TestCase):
    def test_epoch_reaches_wandb_and_local_csv_before_later_failure(self):
        """Exercise the real train_fold wiring with no TF, audio or model updates."""
        row = {"sample_id": "synthetic", "split": "train"}
        run = Mock(id="synthetic-id", dir="synthetic-files")
        run.finish.side_effect = TimeoutError("synthetic finish failure")
        sdk = fake_sdk(run)
        tf = Mock()
        tf.device.side_effect = lambda _: contextlib.nullcontext()
        tf.function.side_effect = lambda _: lambda x, y: types.SimpleNamespace(numpy=lambda: 0.25)
        keras = Mock()
        keras.models.load_model.side_effect = ValueError("synthetic post-epoch failure")
        sampler = Mock()
        sampler.table.return_value = []
        sampler.draw.return_value = (None, None, None)
        original_write_csv = training.source.write_csv

        def save_history_only(path, rows):
            if path.name == "history.csv":
                original_write_csv(path, rows)

        metrics = dict(domestic_target_macro_recall=0.5, overseas_other_clip_fpr=0.1,
                       domestic_doorlock_false_positive_groups=0)
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"wandb": sdk}))
            stack.enter_context(patch.object(training, "ROOT", Path(temporary)))
            stack.enter_context(patch.object(training, "config", return_value=({"training": options()}, {"version": "synthetic"})))
            stack.enter_context(patch.object(training, "development_splits", return_value=[[row]]))
            stack.enter_context(patch.object(training, "read_annotations", return_value={}))
            stack.enter_context(patch.object(training, "require_review_receipt", return_value={}))
            stack.enter_context(patch.object(training, "development_identity", return_value={"synthetic": True}))
            runtime = stack.enter_context(patch.object(training, "runtime", return_value=(tf, keras)))
            stack.enter_context(patch.object(training, "load_fold", return_value=({}, [])))
            stack.enter_context(patch.object(training, "build_model", return_value=Mock()))
            stack.enter_context(patch.object(training, "complexity", return_value={}))
            stack.enter_context(patch.object(training, "predict_clips", return_value=[]))
            stack.enter_context(patch.object(training.subprocess, "run", return_value=types.SimpleNamespace(stdout="synthetic==1\n")))
            stack.enter_context(patch.multiple(training.source, GroupSampler=Mock(return_value=sampler),
                                             settings=Mock(return_value=({"validation_guards": {}}, {})),
                                             split_path=Mock(return_value=Path("synthetic.csv")),
                                             sha256=Mock(return_value="synthetic-hash"),
                                             write_csv=Mock(side_effect=save_history_only),
                                             report=Mock(return_value=metrics),
                                             checkpoint_rank=Mock(return_value=(True, 0.5)),
                                             aggregate_groups=Mock(return_value=[])))
            with self.assertRaisesRegex(ValueError, "synthetic post-epoch failure"):
                training.train_fold(arguments("offline"))
            runtime.assert_called_once_with("cpu", wandb_mode="offline")
            directory = Path(temporary) / "runs/synthetic-run"
            with (directory / "history.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual(status["first_error"], "ValueError: synthetic post-epoch failure")
            self.assertEqual(status["wandb"]["finish_error_type"], "TimeoutError")
            run.log.assert_called_once_with(history(), step=1, commit=True)
            run.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
