"""Matched-fold teacher heads and file-label distilled expanded students."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np

from dscnn_baseline.common import config, runtime, source, write_json, new_directory
from dscnn_baseline.export import export_int8
from dscnn_baseline.model import complexity
from dscnn_baseline.training import predict_clips
from dscnn_baseline.weak_labels import FileBagSampler
from improvement import policy
from .data import identity, load_features, read_rows
from .model import build_model, build_teacher_head, make_update, teacher_update
from .protocol import CANDIDATES, EMBEDDINGS, OPTIONS, REPORT, RUNS, code_identity
from .runtime import frozen_inputs


def run_directory(role, fold=None, candidate=None):
    suffix = "final" if fold is None else f"f{fold}"
    label = "teacher" if role == "teacher" else candidate
    return RUNS / f"dscnn_expanded_20261114_{label}_{suffix}"


def run(args):
    frozen_inputs()
    if not source.read_json(REPORT / "feature_preparation.json")["complete"]:
        raise ValueError("Complete development features required")
    fold = None if args.refit_epochs else args.fold
    output = new_directory(run_directory(args.role, fold, args.candidate))
    status = {"role": args.role, "candidate": args.candidate, "fold": fold, "exit_code": None,
              "device": "gpu", "test_evaluated": False, "deployment_ready": False,
              "distillation_applied": args.role == "student", "event_annotations_used": False}
    write_json(output / "status.json", status)
    started = time.monotonic()
    try:
        tf, keras = runtime("gpu")
        tf.config.experimental.enable_op_determinism()
        seed = OPTIONS["seed"] + (fold or 0)
        keras.utils.set_random_seed(seed)
        rows = read_rows(fold)
        training = [r for r in rows if r["split"] == "train"]
        validation = [r for r in rows if r["split"] == "validation"]
        features, provenance = load_features(rows)
        hashes = {r["sample_id"]: r for r in provenance}
        embeddings = {}
        for row in rows:
            sample = row["sample_id"]
            path = EMBEDDINGS / f"{sample}.npy"
            receipt = source.read_json(path.with_suffix(".json"))
            if source.sha256(path) != receipt["embedding_sha256"] or receipt["source_sha256"] != hashes[sample]["source_sha256"] or receipt["window_count"] != len(features[sample]):
                raise ValueError("Aligned embeddings changed")
            embeddings[sample] = np.load(path, mmap_mode="r", allow_pickle=False)
        source.write_csv(output / "used_train_validation.csv", rows)
        write_json(output / "feature_provenance.json", provenance)
        teacher_receipt, targets = None, {}
        if args.role == "student":
            teacher_path = run_directory("teacher", fold)
            teacher_status = source.read_json(teacher_path / "status.json")
            teacher_run = source.read_json(teacher_path / "run.json")
            if teacher_status["exit_code"] != 0 or teacher_run["dataset_identity"] != identity():
                raise ValueError("Matching corrected-label teacher incomplete or stale")
            teacher_rows = source.read_csv(teacher_path / "used_train_validation.csv")
            if {r["sample_id"] for r in teacher_rows if r["split"] == "train"} != {r["sample_id"] for r in training}:
                raise ValueError("Teacher/student training inventories differ")
            train_groups = {r["split_group_id"] for r in teacher_rows if r["split"] == "train"}
            if train_groups.intersection(r["split_group_id"] for r in validation):
                raise ValueError("Teacher fitting leaks student validation")
            with tf.device("/GPU:0"):
                teacher = keras.models.load_model(str(teacher_path / "classifier.keras"), compile=False)
                teacher.trainable = False
                before = [v.numpy().copy() for v in teacher.weights]
                for row in training:
                    targets[row["sample_id"]] = teacher(embeddings[row["sample_id"]], training=False).numpy()
            if any(not np.array_equal(a, b.numpy()) for a, b in zip(before, teacher.weights)):
                raise ValueError("Teacher changed during student soft-target preparation")
            np.savez(output / "training_teacher_logits.npz", **targets)
            teacher_receipt = {"directory": str(teacher_path), "model_sha256": source.sha256(teacher_path / "classifier.keras"),
                               "soft_logits_sha256": source.sha256(output / "training_teacher_logits.npz"),
                               "output": "pre_softmax_logits", "validation_group_leakage": False, "teacher_weights_unchanged": True,
                               "corrected_siren_labels_used": True}
            write_json(output / "teacher_provenance.json", teacher_receipt)
            del teacher
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        with tf.device("/GPU:0"):
            model = build_teacher_head(keras, seed) if args.role == "teacher" else build_model(keras, CANDIDATES[args.candidate], seed)
            optimizer = keras.optimizers.Adam(OPTIONS["learning_rate"])
            probability_model = keras.Model(model.input, keras.layers.Softmax()(model.output)) if args.role == "teacher" else model
        update = teacher_update(tf, keras, model, optimizer) if args.role == "teacher" else make_update(tf, keras, model, optimizer, OPTIONS, True)
        working = embeddings if args.role == "teacher" else features
        sampler = FileBagSampler(training, working, config()[1], OPTIONS["domestic_weight"], seed)
        group_members = sampler.groups
        sampling = []
        for row in sampler.table():
            members = group_members[row["split_group_id"]]
            sampling.append(dict(row, label=members[0]["label"], sampler_pool=members[0]["sampler_pool"],
                                 domain_weight=(4 if members[0]["sampler_pool"] == "domestic_target" else 1) if members[0]["label"] != "other" else "pool-specific"))
        source.write_csv(output / "sampling_groups.csv", sampling)
        for label in source.LABELS:
            if not np.isclose(sum(r["group_probability"] for r in sampling if r["label"] == label), 1 / 3):
                raise ValueError("Class sampling mass differs from1/3")
        for label in ("fire_alarm", "doorbell"):
            domestic = [r["group_probability"] for r in sampling if r["label"] == label and r["domain_weight"] == 4]
            foreign = [r["group_probability"] for r in sampling if r["label"] == label and r["domain_weight"] == 1]
            if not domestic or not foreign or not all(np.isclose(p / foreign[0], 4) for p in domestic):
                raise ValueError("Domestic positive source weight must be4 times overseas")
        write_json(output / "run.json", {"role": args.role, "candidate": args.candidate, "options": OPTIONS,
                   "dataset_identity": identity(), "code_identity": code_identity(), "teacher": teacher_receipt,
                   "seed": seed, "refit_epochs": args.refit_epochs, "supervision": "file-label bags; public catalog labels and explicit siren policy",
                   "human_window_labels_assigned": False, "complexity": None if args.role == "teacher" else complexity(model, keras)})
        guards = source.settings()[0]["validation_guards"]
        best, stale, history, exposure = None, 0, [], Counter()
        fixed = source.read_json(args.decision)["decision_policy"]["thresholds"] if args.refit_epochs else None
        for epoch in range(1, (args.refit_epochs or OPTIONS["epochs"]) + 1):
            losses = []
            for _ in range(OPTIONS["steps_per_epoch"]):
                batches = []
                for x, weights, y, ids in sampler.draw_bags(OPTIONS["batch_size"], OPTIONS["microbatch_files"]):
                    exposure.update(ids)
                    batches.append((x, weights, y, ids if args.role == "teacher" else np.concatenate([targets[s] for s in ids])))
                value = update(batches).numpy()
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite expanded training loss")
                losses.append(value)
            with tf.device("/GPU:0"):
                predictions = predict_clips(probability_model, validation, working)
            if fixed is None:
                thresholds, metric_list, _, rank = policy.calibrate([predictions], guards)
                metrics = metric_list[0]
                improved = best is None or rank > best
            else:
                thresholds = fixed
                metrics = policy.report(predictions, thresholds)
                rank, improved = (1, epoch), epoch == args.refit_epochs
            if improved:
                best, stale, selected_epoch = rank, 0, epoch
                model.save(str(output / "classifier.keras"))
                source.write_csv(output / "validation_clips.csv", policy.apply(predictions, thresholds))
                write_json(output / "validation_metrics.json", metrics)
                write_json(output / "decision_policy.json", {"thresholds": thresholds})
            else:
                stale += 1
            mean = np.mean(losses, axis=0)
            history.append({"epoch": epoch, "train_loss": float(mean[0]), "hard_label_loss": float(mean[1]),
                            "distillation_loss_T_squared": float(mean[2]), "domestic_target_recall": metrics["domestic_target_macro_recall"],
                            "overseas_other_fpr": metrics["overseas_other_clip_fpr"], "domestic_doorlock_fp": metrics["domestic_doorlock_false_positive_groups"],
                            "fire_threshold": thresholds["fire_alarm"], "bell_threshold": thresholds["doorbell"], "saved": improved})
            source.write_csv(output / "history.csv", history)
            print("EXPANDED_EPOCH", args.role, args.candidate, fold, history[-1], flush=True)
            if not args.refit_epochs and stale >= OPTIONS["patience"]:
                break
        with tf.device("/GPU:0"):
            restored = keras.models.load_model(str(output / "classifier.keras"), compile=False)
            check_model = keras.Model(restored.input, keras.layers.Softmax()(restored.output)) if args.role == "teacher" else restored
            actual = predict_clips(check_model, validation, working)
        expected = source.read_csv(output / "validation_clips.csv")
        for a, b in zip(actual, expected):
            np.testing.assert_allclose([a[f"p_{label}"] for label in source.LABELS], [float(b[f"p_{label}"]) for label in source.LABELS], atol=1e-6, rtol=1e-5)
        source.write_csv(output / "sample_exposure.csv", [{"sample_id": row["sample_id"], "label": row["label"], "draws": exposure[row["sample_id"]]} for row in training])
        if args.refit_epochs and args.role == "student":
            calibration, _, ids = sampler.draw(128)
            export_int8(tf, restored, calibration, output / "export", [{"split": "train", "sample_id": sid, "source_sha256": hashes[sid]["source_sha256"]} for sid in ids])
        status.update(exit_code=0, epochs_completed=len(history), selected_epoch=selected_epoch, optimizer_steps=int(optimizer.iterations.numpy()),
                      distillation_optimizer_steps=int(optimizer.iterations.numpy()) if args.role == "student" else 0,
                      model_sha256=source.sha256(output / "classifier.keras"), model_variable_devices=sorted({v.device for v in model.variables}),
                      save_reload_verified=True, training_siren_clips=sum(r["label_basis"] == "user_explicit_siren_positive_policy" for r in training))
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("teacher", "student"), required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, default="freq24")
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--refit-epochs", type=int)
    parser.add_argument("--decision", type=Path)
    run(parser.parse_args())
