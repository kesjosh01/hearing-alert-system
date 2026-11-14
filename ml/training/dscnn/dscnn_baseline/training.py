"""Annotated source-preserving development training, for a later approved run."""
from __future__ import annotations

import hashlib
from collections import Counter
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .common import ROOT, config, new_directory, runtime, source, write_json
from .data import development_identity, development_refit_rows, development_splits, load_file_label_fold, load_fold, read_annotations
from .export import export_int8, interpreter_for, predict_int8
from .model import build_model, complexity
from .tracking import finish_run, log_epoch, start_run, validate_options
from .review import require_review_receipt
from .weak_labels import FileBagSampler, file_label_update, require_file_label_approval
from . import decisions


def predict_clips(model, rows, features):
    # Batch across files without changing the order or equal-file aggregation.
    outputs = [[] for _ in rows]
    batch, owners = [], []
    def flush():
        if batch:
            probabilities = model(np.stack(batch), training=False).numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite classifier prediction")
            for owner, value in zip(owners, probabilities):
                outputs[owner].append(value)
            batch.clear(); owners.clear()
    for i, row in enumerate(rows):
        for value in features[row["sample_id"]]:
            batch.append(value); owners.append(i)
            if len(batch) == 128:
                flush()
    flush()
    return [source.prediction_row(row, np.asarray(values).mean(0), len(values))
            for row, values in zip(rows, outputs)]


def train_fold(args):
    cfg, protocol = config()
    options = dict(cfg["training"])
    validate_options(args)
    if args.epochs is not None:
        options["epochs"] = args.epochs
    if getattr(args, "domestic_weight", None) is not None:
        options["domestic_weight"] = args.domestic_weight
    is_refit = bool(getattr(args, "is_refit", False))
    if is_refit:
        options["patience"] = 0
    if options["epochs"] <= 0:
        raise ValueError("Epochs must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", "..") or "\\" in args.run_name:
        raise ValueError("Run name must be one directory name")
    folds = development_splits()
    # Select a truthful supervision contract before TensorFlow or audio access.
    mode = getattr(args, "supervision", "event_intervals")
    if mode == "file_labels":
        supervision = require_file_label_approval(args.file_label_approval)
        annotations = None
    elif mode == "event_intervals":
        require_review_receipt(args.annotations)
        annotations = read_annotations(args.annotations, folds[0])
        supervision = {"mode": mode, "annotations_sha256": source.sha256(args.annotations)}
    else:
        raise ValueError("Unknown supervision mode")
    directory = new_directory(ROOT / "runs" / args.run_name)
    status = {"kind": "dscnn_refit" if is_refit else "dscnn_development", "fold": args.fold, "device": args.device,
              "test_evaluated": False, "deployment_ready": False, "exit_code": None,
              "wandb": {"mode": args.wandb_mode}}
    started = time.monotonic()
    write_json(directory / "status.json", status)
    wandb_run = None
    try:
        tf, keras = runtime(args.device, wandb_mode=args.wandb_mode)
        rows = development_refit_rows() if is_refit else folds[args.fold]
        train = [r for r in rows if r["split"] == "train"]
        validation = [r for r in rows if r["split"] == "validation"]
        features, provenance = load_file_label_fold(rows) if mode == "file_labels" else load_fold(rows, annotations, options["minimum_event_coverage"])
        write_json(directory / "feature_provenance.json", provenance)
        source.write_csv(directory / "used_train_validation.csv", rows)
        seed = options["seed"] + args.fold
        sampler_class = FileBagSampler if mode == "file_labels" else source.GroupSampler
        sampler = sampler_class(train, features, protocol, options["domestic_weight"], seed)
        source.write_csv(directory / "sampling_groups.csv", sampler.table())
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        placement = "/GPU:0" if args.device == "gpu" else "/CPU:0"
        with tf.device(placement):
            model = build_model(keras, seed)
            optimizer = keras.optimizers.Adam(options["learning_rate"])
        baseline, _ = source.settings()
        write_json(directory / "run.json", {
            "config": cfg, "options": options, "protocol_version": protocol["version"],
            "split_sha256": source.sha256(source.split_path(protocol, args.fold)),
            "development_identity": development_identity(),
            "annotations_sha256": supervision.get("annotations_sha256"), "annotations": annotations,
            "supervision": supervision,
            "kind": status["kind"], "refit_source": getattr(args, "refit_source", None),
            "code_sha256": {p.name: source.sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "command": [sys.executable, *sys.argv], "complexity": complexity(model, keras),
            "validation_guards": baseline["validation_guards"],
            "note": "No fixed test or board access. Refit epoch is fixed before training; validation cannot change it."})
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        (directory / "packages.txt").write_text(freeze.stdout, encoding="utf-8")

        wandb_run = start_run(args, options, directory)
        if wandb_run is not None:
            status["wandb"].update(project=args.wandb_project, entity=args.wandb_entity,
                                   run_id=wandb_run.id, directory=wandb_run.dir)
            if args.wandb_mode == "online":
                status["wandb"]["url"] = wandb_run.url
                print(f"W&B run: {wandb_run.url}", flush=True)
            write_json(directory / "status.json", status)

        @tf.function
        def update(x, y):
            with tf.device(placement), tf.GradientTape() as tape:
                loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, model(x, training=True)))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        bag_update = file_label_update(tf, keras, model, optimizer, placement) if mode == "file_labels" else None

        best, best_epoch, stale, history = None, 0, 0, []
        best_threshold = float(getattr(args, "decision_threshold", 0.0))
        exposure = Counter()
        for epoch in range(1, options["epochs"] + 1):
            losses = []
            for _ in range(options["steps_per_epoch"]):
                if mode == "file_labels":
                    bags = list(sampler.draw_bags(options["batch_size"], options["file_label_microbatch_files"]))
                    for _, _, _, ids in bags:
                        exposure.update(ids)
                    loss = float(bag_update(bags).numpy())
                else:
                    x, y, ids = sampler.draw(options["batch_size"])
                    exposure.update(ids)
                    loss = float(update(x, y).numpy())
                if not np.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                losses.append(loss)
            predictions = predict_clips(model, validation, features)
            if is_refit:
                threshold = float(args.decision_threshold)
                metrics, rank = decisions.report(predictions, threshold), (1, epoch)
            elif "decision_threshold_grid" in options:
                threshold, metrics, rank = decisions.calibrate(predictions, baseline["validation_guards"], options["decision_threshold_grid"])
            else:
                threshold = 0.0
                metrics = source.report(predictions)
                rank = source.checkpoint_rank(metrics, baseline["validation_guards"])
            predictions = decisions.apply(predictions, threshold)
            improved = epoch == options["epochs"] if is_refit else best is None or rank > best
            if improved:
                best, best_epoch, stale = rank, epoch, 0
                best_threshold = threshold
                model.save(str(directory / "classifier.keras"))
                source.write_csv(directory / "validation_clips.csv", predictions)
                source.write_csv(directory / "validation_groups.csv", decisions.aggregate(predictions, threshold))
                write_json(directory / "validation_metrics.json", metrics)
            else:
                stale += 1
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "domestic_target_recall": metrics["domestic_target_macro_recall"],
                            "overseas_other_fpr": metrics["overseas_other_clip_fpr"],
                            "domestic_doorlock_fp": metrics["domestic_doorlock_false_positive_groups"],
                            "validation_guards_passed": None if is_refit else bool(rank[0]), "saved": improved,
                            "decision_threshold": threshold})
            source.write_csv(directory / "history.csv", history)
            log_epoch(wandb_run, history[-1])
            print(f"DS-CNN fold {args.fold} epoch {epoch}: {history[-1]}", flush=True)
            if options["patience"] and stale >= options["patience"]:
                break
        restored = keras.models.load_model(str(directory / "classifier.keras"), compile=False)
        reloaded_predictions = predict_clips(restored, validation, features)
        expected = source.read_csv(directory / "validation_clips.csv")
        values = lambda items: np.asarray([[float(r[f"p_{label}"]) for label in source.LABELS] for r in items])
        np.testing.assert_allclose(values(reloaded_predictions), values(expected), rtol=1e-5, atol=1e-6)
        source.write_csv(directory / "sample_exposure.csv", [
            {"sample_id": r["sample_id"], "split_group_id": r["split_group_id"],
             "draws": exposure[r["sample_id"]]} for r in train])
        status.update(exit_code=0, selected_epoch=best_epoch, epochs_completed=len(history),
                      validation_guards_passed=None if is_refit else bool(best[0]),
                      model_variable_devices=sorted({v.device for v in model.variables}),
                      optimizer_steps=int(optimizer.iterations.numpy()),
                      model_sha256=source.sha256(directory / "classifier.keras"),
                      selected_threshold=best_threshold,
                      save_reload_verified=True)
        if getattr(args, "skip_export", False):
            status["int8_export_skipped_for_float_cv"] = True
            return directory
        calibration, _, ids = sampler.draw(128)
        source_hashes = {p["sample_id"]: p["source_sha256"] for p in provenance}
        calibration_provenance = [{"split": "train", "sample_id": sample, "source_sha256": source_hashes[sample],
                                   "logmel_sha256": hashlib.sha256(feature.tobytes()).hexdigest()}
                                  for sample, feature in zip(ids, calibration)]
        path, metadata = export_int8(tf, restored, calibration, directory / "export", calibration_provenance)
        write_json(directory / "decision_policy.json", {"kind":"argmax_with_target_confidence_rejection",
                   "threshold":best_threshold,"clip_aggregation":"all_window_mean_softmax",
                   "group_aggregation":"equal_clip_mean_softmax_then_decide",
                   "threshold_selection":"development_only_fixed_before_refit_and_test"})
        integer = interpreter_for(tf, path)
        int_predictions = []
        max_error = 0.0
        for row in validation:
            feature = features[row["sample_id"]]
            float_prob = restored(feature[:], training=False).numpy().mean(0)
            quant_prob = predict_int8(integer, feature).mean(0)
            max_error = max(max_error, float(np.max(np.abs(float_prob - quant_prob))))
            int_predictions.append(source.prediction_row(row, quant_prob, len(feature)))
        int_predictions = decisions.apply(int_predictions, best_threshold)
        source.write_csv(directory / "int8_validation_clips.csv", int_predictions)
        write_json(directory / "int8_validation_metrics.json", decisions.report(int_predictions,best_threshold))
        status.update(exit_code=0, selected_epoch=best_epoch, epochs_completed=len(history),
                      validation_guards_passed=None if is_refit else bool(best[0]), int8_export=metadata,
                      float_int8_clip_probability_max_abs=max_error)
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (directory / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        finish_run(wandb_run, status)
        write_json(directory / "status.json", status)
    return directory


def train_refit(args):
    summary = source.read_json(args.cv_summary)
    if summary.get("kind") != "dscnn_four_fold_cv" or len(summary.get("folds", [])) != 4:
        raise ValueError("Refit requires a completed four-fold DS-CNN CV")
    if not all(f["validation_guards_passed"] for f in summary["folds"]):
        raise ValueError("Refit is blocked because the selected CV did not pass every validation guard")
    if summary["development_identity"] != development_identity() or summary["config"] != config()[0]:
        raise ValueError("CV data or configuration changed before refit")
    mode = getattr(args, "supervision", "event_intervals")
    if mode != summary["supervision"]["mode"]:
        raise ValueError("Refit supervision differs from CV")
    if mode == "file_labels":
        if require_file_label_approval(args.file_label_approval) != summary["supervision"]:
            raise ValueError("Approved file-label supervision changed before refit")
    elif summary["annotations_sha256"] != source.sha256(args.annotations):
        raise ValueError("Reviewed event annotations changed before refit")
    from .study import code_identity
    if summary["code_identity"] != code_identity():
        raise ValueError("DS-CNN code changed after CV")
    for fold in summary["folds"]:
        path = Path(fold["run"])
        if source.sha256(path / "classifier.keras") != fold["model_sha256"]:
            raise ValueError("A selected CV checkpoint changed before refit")
    args.epochs = summary["refit_epochs"]
    args.decision_threshold = summary["decision_policy"]["threshold"]
    args.domestic_weight = summary["domestic_weight"]
    args.fold, args.is_refit, args.skip_export = 0, True, False
    args.refit_source = {"summary": str(Path(args.cv_summary).resolve()),
                         "sha256": source.sha256(args.cv_summary)}
    return train_fold(args)
