from __future__ import annotations

import argparse
from collections import Counter
import time
import traceback
from pathlib import Path

import numpy as np

from dscnn_baseline.common import ROOT, runtime, config, source, write_json, new_directory
from dscnn_baseline.data import development_identity, development_splits, development_refit_rows, load_file_label_fold
from dscnn_baseline.weak_labels import FileBagSampler, require_file_label_approval
from dscnn_baseline.training import predict_clips
from dscnn_baseline.model import complexity
from dscnn_baseline.export import export_int8
from .protocol import CANDIDATES, OPTIONS, TEACHER_CACHE, REPORT, code_identity, verify_teacher
from .model import build_model, make_update
from . import policy


def train(args):
    method = CANDIDATES[args.candidate]
    options = dict(OPTIONS, domestic_weight=args.domestic_weight)
    if args.hard_label_only:
        options["hard_label_weight"] = 1.0
    output = new_directory(ROOT / "runs" / args.run_name)
    status = {"kind": "dscnn_distilled_refit" if args.refit_epochs else "dscnn_distilled_cv",
              "device": "gpu", "exit_code": None, "candidate": args.candidate, "fold": args.fold,
              "event_annotations_used": False, "test_evaluated": False, "deployment_ready": False}
    status["distillation_applied"] = not args.hard_label_only
    write_json(output / "status.json", status)
    started = time.monotonic()
    try:
        supervision = require_file_label_approval(ROOT / "data/file_label_approval_20261114.json")
        teacher_receipt = verify_teacher(None if args.refit_epochs else args.fold)
        preparation = source.read_json(REPORT / "teacher_preparation.json")
        if preparation.get("complete") is not True or preparation.get("fixed_test_files") != 0:
            raise ValueError("Aligned development-only teacher features required")
        tf, keras = runtime("gpu")
        tf.config.experimental.enable_op_determinism()
        seed = OPTIONS["seed"] + (0 if args.refit_epochs else args.fold)
        keras.utils.set_random_seed(seed)
        rows = development_refit_rows() if args.refit_epochs else development_splits()[args.fold]
        training = [r for r in rows if r["split"] == "train"]
        validation = [r for r in rows if r["split"] == "validation"]
        features, provenance = load_file_label_fold(rows)
        hashes = {r["sample_id"]: r for r in provenance}
        source.write_csv(output / "used_train_validation.csv", rows)
        write_json(output / "feature_provenance.json", provenance)
        with tf.device("/GPU:0"):
            teacher = keras.models.load_model(str(Path(teacher_receipt["directory"]) / "classifier.keras"), compile=False)
            teacher.trainable = False
        teacher_weights_before = [v.numpy().copy() for v in teacher.weights]
        targets = {}
        for row in training:
            sample = row["sample_id"]
            path = TEACHER_CACHE / f"{sample}.npy"
            receipt = source.read_json(path.with_suffix(".json"))
            if source.sha256(path) != receipt["embedding_sha256"] or receipt["source_sha256"] != hashes[sample]["source_sha256"] or len(features[sample]) != receipt["window_count"]:
                raise ValueError("Teacher cache identity/alignment changed")
            embeddings = np.load(path, allow_pickle=False)
            targets[sample] = teacher(embeddings, training=False).numpy().astype(np.float32)
        if any(not np.array_equal(a, b.numpy()) for a, b in zip(teacher_weights_before, teacher.weights)):
            raise ValueError("Teacher parameters changed")
        np.savez(output / "training_teacher_logits.npz", **targets)
        write_json(output / "teacher_provenance.json", dict(teacher_receipt,
                   output_kind="pre_softmax_logits", soft_targets_sha256=source.sha256(output / "training_teacher_logits.npz"),
                   student_training_sample_ids=sorted(targets), teacher_weights_unchanged=True))
        del teacher
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        with tf.device("/GPU:0"):
            model = build_model(keras, method, seed)
            optimizer = keras.optimizers.Adam(OPTIONS["learning_rate"])
        update = make_update(tf, keras, model, optimizer, options, method["specaugment"])
        _, protocol = config()
        sampler = FileBagSampler(training, features, protocol, options["domestic_weight"], seed)
        source.write_csv(output / "sampling_groups.csv", sampler.table())
        write_json(output / "run.json", {"candidate": args.candidate, "method": method, "options": options,
                   "supervision": supervision, "teacher": teacher_receipt, "development_identity": development_identity(),
                   "teacher_soft_window_targets_used": not args.hard_label_only, "human_window_labels_assigned": False,
                   "code_identity": code_identity(), "config": config()[0], "seed": seed,
                   "refit_epochs": args.refit_epochs, "complexity": complexity(model, keras)})
        epochs = args.refit_epochs or OPTIONS["epochs"]
        guards = source.settings()[0]["validation_guards"]
        best, stale, history, exposure = None, 0, [], Counter()
        for epoch in range(1, epochs + 1):
            losses = []
            for step in range(OPTIONS["steps_per_epoch"]):
                batches = []
                for x, weights, y, ids in sampler.draw_bags(OPTIONS["batch_size"], OPTIONS["microbatch_files"]):
                    exposure.update(ids)
                    batches.append((x, weights, y, np.concatenate([targets[s] for s in ids])))
                value = update(batches).numpy()
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite distillation loss")
                losses.append(value)
            predictions = predict_clips(model, validation, features)
            if args.refit_epochs:
                thresholds = source.read_json(args.decision)["decision_policy"]["thresholds"]
                metrics = policy.report(predictions, thresholds)
                rank, improved = (1, epoch), epoch == epochs
            else:
                thresholds, metrics_list, _, rank = policy.calibrate([predictions], guards)
                metrics = metrics_list[0]
                improved = best is None or rank > best
            if improved:
                best, stale, selected_epoch, selected_thresholds = rank, 0, epoch, thresholds
                model.save(str(output / "classifier.keras"))
                source.write_csv(output / "validation_clips.csv", policy.apply(predictions, thresholds))
                write_json(output / "validation_metrics.json", metrics)
            else:
                stale += 1
            average = np.mean(losses, axis=0)
            history.append({"epoch": epoch, "train_loss": float(average[0]), "hard_label_loss": float(average[1]),
                            "distillation_loss_T_squared": float(average[2]),
                            "domestic_target_recall": metrics["domestic_target_macro_recall"],
                            "overseas_other_fpr": metrics["overseas_other_clip_fpr"],
                            "domestic_doorlock_fp": metrics["domestic_doorlock_false_positive_groups"],
                            "fire_threshold": thresholds["fire_alarm"], "bell_threshold": thresholds["doorbell"], "saved": improved})
            source.write_csv(output / "history.csv", history)
            print("DISTILLATION_EPOCH", args.candidate, args.fold, history[-1], flush=True)
            if not args.refit_epochs and stale >= OPTIONS["patience"]:
                break
        restored = keras.models.load_model(str(output / "classifier.keras"), compile=False)
        actual = predict_clips(restored, validation, features)
        expected = source.read_csv(output / "validation_clips.csv")
        for a, b in zip(actual, expected):
            np.testing.assert_allclose([a[f"p_{l}"] for l in source.LABELS], [float(b[f"p_{l}"]) for l in source.LABELS], atol=1e-6, rtol=1e-5)
        source.write_csv(output / "sample_exposure.csv", [{"sample_id": r["sample_id"], "draws": exposure[r["sample_id"]]} for r in training])
        decision_policy = {"kind": "per_target_argmax_confidence_rejection", "thresholds": selected_thresholds,
                           "clip_aggregation": "all_window_mean_softmax", "group_aggregation": "equal_clip_mean_softmax_then_decide",
                           "selection_source": "development_only_fixed_before_refit"}
        write_json(output / "decision_policy.json", decision_policy)
        if args.refit_epochs:
            calibration, _, ids = sampler.draw(128)
            provenance = [{"split": "train", "sample_id": sample, "source_sha256": hashes[sample]["source_sha256"]} for sample in ids]
            export_int8(tf, restored, calibration, output / "export", provenance)
        status.update(exit_code=0, epochs_completed=len(history), selected_epoch=selected_epoch,
                      selected_thresholds=selected_thresholds, optimizer_steps=int(optimizer.iterations.numpy()),
                      model_sha256=source.sha256(output / "classifier.keras"),
                      model_variable_devices=sorted({v.device for v in model.variables}),
                      teacher_weights_unchanged=True, save_reload_verified=True,
                      distillation_optimizer_steps=0 if args.hard_label_only else int(optimizer.iterations.numpy()))
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--refit-epochs", type=int)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--hard-label-only", action="store_true")
    parser.add_argument("--domestic-weight", type=float, default=1.0)
    train(parser.parse_args())
