"""Small legacy-Keras head, source-level validation, CV, and fixed-epoch refit."""
from __future__ import annotations

import csv
import json
import statistics
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .core import (ROOT, LABELS, GroupSampler, aggregate_groups, checkpoint_rank, fingerprint,
                   prediction_row, read_csv, read_json, report, sha256, softmax, split_path, write_csv, write_json)
from .features import load_classifier, package_versions, runtime_cache, save_classifier, tensorflow_runtime


def new_run(baseline, prefix, name=None):
    name = name or f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if Path(name).name != name or any(char in name for char in ("/", "\\", ":")):
        raise ValueError("Run name must be one directory name")
    directory = ROOT / baseline["runs_dir"] / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def run_options(args, baseline, protocol):
    options = {
        "seed": args.seed if args.seed is not None else protocol["seed"],
        "domestic_weight": args.domestic_weight if args.domestic_weight is not None else protocol["sampling"]["target_class_domestic_group_weight"],
        "epochs": args.epochs if args.epochs is not None else baseline["max_epochs"],
        "patience": args.patience if args.patience is not None else baseline["patience"],
        "batch_size": args.batch_size if args.batch_size is not None else protocol["sampling"]["suggested_batch_size"],
        "steps_per_epoch": args.steps_per_epoch if args.steps_per_epoch is not None else protocol["sampling"]["suggested_steps_per_epoch"],
        "learning_rate": args.learning_rate if args.learning_rate is not None else baseline["learning_rate"],
        "wandb_mode": args.wandb_mode or baseline["wandb_mode"],
        "validation_guards": dict(baseline["validation_guards"]),
    }
    if getattr(args, "max_other_fpr", None) is not None:
        options["validation_guards"]["max_overseas_other_clip_fpr"] = args.max_other_fpr
    if any(options[key] <= 0 for key in ("domestic_weight", "epochs", "batch_size", "steps_per_epoch", "learning_rate")) or options["patience"] < 0:
        raise ValueError("Epochs, batch size, steps, learning rate and domain weight must be positive; patience must be nonnegative")
    if not 0 <= options["validation_guards"]["max_overseas_other_clip_fpr"] <= 1:
        raise ValueError("max-other-fpr must be between 0 and 1")
    return options


def predict_clips(model, rows, features):
    predictions = []
    for row in rows:
        values = features[row["sample_id"]]
        chunks = [softmax(model(values[start:start + 256], training=False).numpy()) for start in range(0, len(values), 256)]
        predictions.append(prediction_row(row, np.concatenate(chunks).mean(axis=0), len(values)))
    return predictions


def save_evaluation(directory, prefix, predictions):
    metrics = report(predictions)
    write_csv(directory / f"{prefix}_clips.csv", predictions)
    write_csv(directory / f"{prefix}_groups.csv", aggregate_groups(predictions))
    write_json(directory / f"{prefix}_metrics.json", metrics)
    return metrics


def load_features(store, rows):
    features = {}
    for i, row in enumerate(rows, 1):
        features[row["sample_id"]] = store.load(row)
        if i % 400 == 0:
            print(f"Loaded cached features: {i}/{len(rows)}", flush=True)
    return features


def verify_sampler_csv(sampler, protocol, fold, refit=False):
    filename = "sampling_groups_refit.csv" if refit else f"sampling_groups_fold_{fold}.csv"
    expected = {row["split_group_id"]: float(row["group_probability"]) for row in read_csv(split_path(protocol).parent / filename)}
    actual = {row["split_group_id"]: row["group_probability"] for row in sampler.table()}
    if set(expected) != set(actual) or any(not np.isclose(expected[key], actual[key], rtol=1e-10, atol=1e-12) for key in expected):
        raise ValueError("Training sampler differs from the frozen 4x sampling design")


def fit_head(rows, features, baseline, protocol, options, directory, fold=0, kind="development"):
    tf, keras = tensorflow_runtime()
    keras.backend.clear_session()
    # A new head and optimizer are constructed for every fold/weight experiment.
    seed = options["seed"] + fold
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    train_rows = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    is_refit = kind == "refit"
    sampler = GroupSampler(train_rows, features, protocol, options["domestic_weight"], seed)
    if kind != "smoke" and options["domestic_weight"] == protocol["sampling"]["target_class_domestic_group_weight"]:
        verify_sampler_csv(sampler, protocol, fold, refit=is_refit)
    write_csv(directory / "sampling_groups.csv", sampler.table())
    write_csv(directory / "used_train_validation.csv", [row for row in rows if row["split"] != "test"])
    model = keras.Sequential([
        keras.layers.Input(shape=(baseline["embedding_dimension"],)),
        keras.layers.Dense(baseline["hidden_units"], activation="relu", kernel_initializer=keras.initializers.GlorotUniform(seed=seed)),
        keras.layers.Dropout(baseline["dropout"], seed=seed + 1),
        keras.layers.Dense(len(LABELS), kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 2)),
    ], name="yamnet_embedding_classifier")
    optimizer = keras.optimizers.Adam(learning_rate=options["learning_rate"])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_batch(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    provenance = [{"sample_id": row["sample_id"], "source_sha256": sha256(ROOT / row["relative_path"]),
                   "embedding_sha256": sha256(ROOT / baseline["processed_dir"] / "embeddings" / f"{row['sample_id']}.npy")}
                  for row in rows if row["split"] != "test"]
    metadata = {"kind": kind, "fold": fold, "labels": LABELS, "baseline": baseline, "protocol_version": protocol["version"],
                "options": options, "versions": package_versions(), "feature_provenance_fingerprint": fingerprint(provenance),
                "code_fingerprint": fingerprint({p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}),
                "split_sha256": sha256(split_path(protocol, fold, refit=is_refit)),
                "evaluation_policy": "native patch -> mean softmax per clip -> equal clip mean per source; argmax, no event threshold tuning"}
    write_json(directory / "run.json", metadata)
    write_json(directory / "feature_provenance.json", provenance)
    with (directory / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (directory / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    wandb_run = None
    if options["wandb_mode"] != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Optional W&B missing. Install wandb or keep --wandb-mode disabled.") from error
        wandb_run = wandb.init(project="capstone-sound", name=directory.name, mode=options["wandb_mode"],
                               dir=str(directory), config={"protocol_version": protocol["version"], "kind": kind, "fold": fold, **options}, save_code=False)
    log_directory = runtime_cache() / "logs" / fingerprint(str(directory))
    summary_writer = tf.summary.create_file_writer(str(log_directory))
    best_key, best_epoch, best_metrics, stale = None, 0, None, 0
    exposure = Counter()
    history = []
    try:
        for epoch in range(1, options["epochs"] + 1):
            losses = []
            for _ in range(options["steps_per_epoch"]):
                x, y, sample_ids = sampler.draw(options["batch_size"])
                losses.append(float(train_batch(x, y).numpy()))
                exposure.update(sample_ids)
            predictions = predict_clips(model, validation, features)
            metrics = report(predictions)
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_group_loss": metrics["groups"]["loss"],
                      "val_domestic_target_recall": metrics["domestic_target_macro_recall"], "val_macro_f1": metrics["groups"]["macro_f1"],
                      "val_overseas_other_fpr": metrics["overseas_other_clip_fpr"], "val_doorlock_fp": metrics["domestic_doorlock_false_positive_groups"]}
            if is_refit:
                improved = epoch == options["epochs"]
                key = None
            else:
                key = checkpoint_rank(metrics, options["validation_guards"])
                improved = best_key is None or key > best_key
                record["validation_guards_passed"] = bool(key[0])
            if improved:
                best_key, best_epoch, best_metrics, stale = key, epoch, metrics, 0
                save_classifier(model, directory / "classifier.keras")
                save_evaluation(directory, "validation", predictions)
            else:
                stale += 1
            history.append(record)
            write_csv(directory / "history.csv", history)
            with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            values = {key: value for key, value in record.items() if value is not None and isinstance(value, (int, float))}
            with summary_writer.as_default():
                for key, value in values.items():
                    tf.summary.scalar(key, value, step=epoch)
            summary_writer.flush()
            if wandb_run:
                wandb_run.log(values)
            print(f"{directory.name} epoch {epoch}: loss={record['train_loss']:.4f} "
                  f"domestic_recall={record['val_domestic_target_recall']} other_fpr={record['val_overseas_other_fpr']} "
                  f"saved={improved}", flush=True)
            if not is_refit and options["patience"] and stale >= options["patience"]:
                break
    finally:
        summary_writer.close()
        if log_directory.is_dir():
            shutil.copytree(log_directory, directory / "tensorboard", dirs_exist_ok=True)
        if wandb_run:
            wandb_run.finish()
    write_csv(directory / "sample_exposure.csv", [{"sample_id": sample_id, "draws": count} for sample_id, count in sorted(exposure.items())])
    result = {"kind": kind, "best_epoch": best_epoch, "epochs_completed": len(history), "validation": best_metrics,
              "validation_guards_passed": None if is_refit else bool(best_key[0]),
              "notice": "SMOKE: pipeline verification only" if kind == "smoke" else "No held-out test was evaluated."}
    write_json(directory / "result.json", result)
    # Check that the saved classifier can be reloaded and produces matching scores.
    restored = load_classifier(directory / "classifier.keras")
    restored_predictions = predict_clips(restored, validation, features)
    if report(restored_predictions) != best_metrics:
        # Hardware kernels can differ by a few ulps; labels/metrics must still match.
        if report(restored_predictions)["groups"]["confusion_matrix"] != best_metrics["groups"]["confusion_matrix"]:
            raise RuntimeError("Saved/reloaded classifier evaluation changed")
    return result


def train_command(args, baseline, protocol, folds, store):
    options = run_options(args, baseline, protocol)
    selected = list(range(len(folds))) if args.fold == "all" else [int(args.fold)]
    development = [row for row in folds[0] if row["split"] != "test"]
    features = load_features(store, development)
    directory = new_run(baseline, f"cv_w{options['domestic_weight']}_s{options['seed']}" if len(selected) > 1 else f"fold{selected[0]}_w{options['domestic_weight']}", args.run_name)
    if len(selected) == 1:
        fit_head(folds[selected[0]], features, baseline, protocol, options, directory, selected[0])
    else:
        results, oof = [], []
        for fold in selected:
            child = directory / f"fold_{fold}"
            child.mkdir()
            result = fit_head(folds[fold], features, baseline, protocol, options, child, fold)
            results.append({"fold": fold, "best_epoch": result["best_epoch"], "validation_guards_passed": result["validation_guards_passed"],
                            "domestic_target_recall": result["validation"]["domestic_target_macro_recall"], "overseas_other_fpr": result["validation"]["overseas_other_clip_fpr"]})
            for row in read_csv(child / "validation_clips.csv"):
                if row["target_domain"] == "ko_target":
                    oof.append(row)
        if len({row["sample_id"] for row in oof}) != len(oof):
            raise ValueError("Domestic OOF samples repeated")
        write_csv(directory / "cv_folds.csv", results)
        write_csv(directory / "oof_domestic_clips.csv", oof)
        write_json(directory / "oof_domestic_metrics.json", report(oof))
        values = [row["domestic_target_recall"] for row in results]
        summary = {"kind": "cv", "protocol_version": protocol["version"], "baseline": baseline, "options": options,
                   "feature_signature": store.signature, "folds": results,
                   "domestic_target_recall_mean": statistics.mean(values), "domestic_target_recall_std": statistics.stdev(values),
                   "refit_epochs": max(1, int(statistics.median(row["best_epoch"] for row in results) + 0.5)),
                   "notice": "Select weights using development metrics only; overseas validation is reused and is not OOF."}
        write_json(directory / "cv_summary.json", summary)
    print(f"Run saved: {directory}", flush=True)
    return directory


def refit_command(args, baseline, protocol, refit_rows, store):
    cv_directory = Path(args.cv_run).resolve()
    summary = read_json(cv_directory / "cv_summary.json")
    if summary.get("kind") != "cv" or len(summary["folds"]) != protocol["split"]["development_folds"]:
        raise ValueError("Refit requires a completed four-fold CV run")
    if summary["protocol_version"] != protocol["version"] or summary["feature_signature"] != store.signature or summary["baseline"] != baseline:
        raise ValueError("CV protocol, features or classifier settings changed")
    options = summary["options"].copy()
    options.update(epochs=summary["refit_epochs"], wandb_mode=args.wandb_mode or "disabled")
    features = load_features(store, [row for row in refit_rows if row["split"] != "test"])
    # Prevent a changed raw/cache dataset from being silently used for final refit.
    provenance = {row["sample_id"]: row for row in read_json(cv_directory / "fold_0/feature_provenance.json")}
    for row in refit_rows:
        if row["split"] == "test":
            continue
        previous = provenance[row["sample_id"]]
        if previous["source_sha256"] != sha256(ROOT / row["relative_path"]) or previous["embedding_sha256"] != sha256(store.paths(row["sample_id"])["embedding"]):
            raise ValueError("Development data changed since CV; review before refit")
    directory = new_run(baseline, f"refit_w{options['domestic_weight']}_s{options['seed']}", args.run_name)
    result = fit_head(refit_rows, features, baseline, protocol, options, directory, kind="refit")
    write_json(directory / "refit_source.json", {"cv_run": str(cv_directory), "cv_summary_sha256": sha256(cv_directory / "cv_summary.json")})
    print(f"Refit saved: {directory} (epochs={result['best_epoch']})", flush=True)
    return directory
