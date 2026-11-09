"""Run with: python -m sound_baseline --help."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .core import (ROOT, LABELS, load_splits, read_json, settings, sha256, softmax, split_path,
                   write_csv, write_json, prediction_row)
from .features import FeatureStore, decode_audio, ffmpeg_binary, load_classifier, package_versions, read_wave, runtime_cache, tensorflow_runtime
from .training import (fit_head, load_features, new_run, predict_clips, refit_command,
                       run_options, save_evaluation, train_command)


def process_rows(store, rows, stage):
    index = []
    for i, row in enumerate(rows, 1):
        if stage == "preprocess":
            meta = store.preprocess(row)
            item = {"sample_id": row["sample_id"], "duration_s": meta["duration_s"], "samples": meta["samples"],
                    "rms": meta["rms"], "near_clipping_fraction": meta["near_clipping_fraction"], "source_sha256": meta["source_sha256"]}
        else:
            item = store.extract(row)
        index.append(item)
        if i == 1 or i % 50 == 0 or i == len(rows):
            print(f"{stage}: {i}/{len(rows)} ({row['sample_id']})", flush=True)
    return index


def smoke(args, baseline, protocol, folds, store):
    # Cover every training pool and every validation domain/class without test.
    buckets = defaultdict(list)
    for row in folds[0]:
        if row["split"] != "test":
            buckets[(row["split"], row["label"], row["sampler_pool"])].append(row)
    selected = []
    for key in sorted(buckets):
        selected.extend(sorted(buckets[key], key=lambda row: row["sample_id"])[:1])
    process_rows(store, selected, "extract")
    features = load_features(store, selected)
    options = run_options(args, baseline, protocol)
    options.update(epochs=2, steps_per_epoch=2, patience=0)
    directory = new_run(baseline, "SMOKE", args.run_name)
    fit_head(selected, features, baseline, protocol, options, directory, kind="smoke")
    print(f"SMOKE PASS: real audio -> real YAMNet -> head training -> checkpoint reload.\n{directory}", flush=True)


def evaluate(args, baseline, protocol, folds, refit_rows, store):
    directory = Path(args.run).resolve()
    metadata = read_json(directory / "run.json")
    if metadata["baseline"] != baseline or metadata["protocol_version"] != protocol["version"]:
        raise ValueError("Run preprocessing/protocol differs from current configuration")
    if metadata["kind"] == "smoke":
        raise ValueError("Smoke models are pipeline tests; use a development or refit run")
    if args.split == "test" and metadata["kind"] != "refit":
        raise ValueError("Test evaluation is reserved for a final refit model. Complete four-fold CV and refit first.")
    rows = refit_rows if metadata["kind"] == "refit" else folds[metadata["fold"]]
    expected_hash = sha256(split_path(protocol, metadata["fold"], refit=metadata["kind"] == "refit"))
    if metadata["split_sha256"] != expected_hash:
        raise ValueError("Split file changed after this run")
    rows = [row for row in rows if row["split"] == args.split]
    output = directory / f"{args.split}_evaluation"
    if (output / "metrics.json").exists():
        previous = read_json(output / "evaluation.json")
        if previous["classifier_sha256"] != sha256(directory / "classifier.keras") or previous["run_sha256"] != sha256(directory / "run.json"):
            raise ValueError("Saved evaluation belongs to a different classifier/run; retain it and use a new run directory")
        print(f"Existing evaluation: {output / 'metrics.json'}")
        print(json.dumps(read_json(output / "metrics.json"), indent=2))
        return
    features = load_features(store, rows)
    model = load_classifier(directory / "classifier.keras")
    output.mkdir()
    metrics = save_evaluation(output, "predictions", predict_clips(model, rows, features))
    write_json(output / "metrics.json", metrics)
    write_json(output / "evaluation.json", {"split": args.split, "classifier_sha256": sha256(directory / "classifier.keras"),
               "run_sha256": sha256(directory / "run.json"), "protocol_version": protocol["version"],
               "notice": "Clip/source classification only; no event latency or deployment false alarms/hour measured."})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Evaluation saved: {output}")


def predict(args, baseline, store):
    directory = Path(args.run).resolve()
    metadata = read_json(directory / "run.json")
    if metadata["baseline"] != baseline:
        raise ValueError("Run preprocessing differs from current configuration")
    source = Path(args.audio).resolve()
    if not source.is_file():
        raise ValueError(f"Audio file does not exist: {source}")
    # External user-selected recordings are decoded locally, never uploaded.
    destination = ROOT / ".cache/predict" / f"{sha256(source)}.wav"
    decode_audio(source, destination, baseline["sample_rate"])
    features = store.embed_waveform(read_wave(destination, baseline["sample_rate"]))
    model = load_classifier(directory / "classifier.keras")
    probabilities = softmax(model(features, training=False).numpy()).mean(axis=0)
    output = {"predicted_label": list(LABELS)[int(probabilities.argmax())],
              "probabilities": {label: float(probabilities[index]) for label, index in LABELS.items()},
              "native_patch_count": len(features), "rule": baseline["clip_prediction"]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parser():
    cli = argparse.ArgumentParser(description="Capstone YAMNet baseline: local audio, frozen source splits, group-balanced training.")
    cli.add_argument("--config", type=Path, help="Alternative baseline JSON; default config/yamnet_baseline_v1.json")
    sub = cli.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check environment, all frozen splits and optional YAMNet download")
    doctor.add_argument("--load-yamnet", action="store_true")
    for name in ("preprocess", "extract", "prepare"):
        command = sub.add_parser(name, help={"preprocess": "Decode audio to 16 kHz mono PCM16", "extract": "Create/cache frozen YAMNet native embeddings", "prepare": "Preprocess and extract (resumable)"}[name])
        command.add_argument("--include-test", action="store_true", help="Also prepare frozen test audio; never used by train")
        command.add_argument("--limit", type=int, help="Only prepare first N selected files for a quick check")
    def training_options(command):
        command.add_argument("--run-name")
        command.add_argument("--seed", type=int)
        command.add_argument("--domestic-weight", type=float)
        command.add_argument("--epochs", type=int)
        command.add_argument("--patience", type=int)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--steps-per-epoch", type=int)
        command.add_argument("--learning-rate", type=float)
        command.add_argument("--max-other-fpr", type=float)
        command.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    train = sub.add_parser("train", help="Train a fresh head on one fold or all four folds; never evaluate test")
    train.add_argument("--fold", choices=("0", "1", "2", "3", "all"), default="0")
    training_options(train)
    smoke_parser = sub.add_parser("smoke", help="Real small-subset integration test, 2 epochs x 2 steps, no test data")
    training_options(smoke_parser)
    refit_parser = sub.add_parser("refit", help="Final training on all domestic development groups, with CV-derived epoch count")
    refit_parser.add_argument("--cv-run", required=True)
    refit_parser.add_argument("--run-name")
    refit_parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    evaluation = sub.add_parser("evaluate", help="Explicit validation or final test evaluation")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    inference = sub.add_parser("predict", help="Predict one local audio recording")
    inference.add_argument("--run", required=True)
    inference.add_argument("--audio", required=True)
    return cli


def main():
    args = parser().parse_args()
    baseline, protocol = settings(args.config)
    folds, refit_rows = load_splits(protocol)
    store = FeatureStore(baseline)
    if args.command == "doctor":
        tf, _ = tensorflow_runtime()
        details = {"python": platform.python_version(), "executable": sys.executable, "ffmpeg": ffmpeg_binary(),
                   "versions": package_versions(), "tensorflow_cache": str(runtime_cache()), "tensorflow_devices": [device.name for device in tf.config.list_physical_devices()],
                   "validated_folds": len(folds), "records_per_fold": len(folds[0]),
                   "development_records": sum(row["split"] != "test" for row in folds[0]),
                   "test_records": sum(row["split"] == "test" for row in folds[0]), "labels": LABELS}
        if args.load_yamnet:
            store.yamnet()
            details["yamnet_loaded"] = True
        print(json.dumps(details, ensure_ascii=False, indent=2))
    elif args.command in ("preprocess", "extract", "prepare"):
        rows = [row for row in folds[0] if args.include_test or row["split"] != "test"]
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("limit must be positive")
            rows = rows[:args.limit]
        stage = "preprocess" if args.command == "preprocess" else "extract"
        index = process_rows(store, rows, stage)
        name = f"{stage}_{'with_test' if args.include_test else 'development'}{'_limited' if args.limit else ''}.csv"
        write_csv(store.directory / name, index)
        print(f"Prepared {len(rows)} records. Index: {store.directory / name}")
    elif args.command == "smoke":
        smoke(args, baseline, protocol, folds, store)
    elif args.command == "train":
        train_command(args, baseline, protocol, folds, store)
    elif args.command == "refit":
        refit_command(args, baseline, protocol, refit_rows, store)
    elif args.command == "evaluate":
        evaluate(args, baseline, protocol, folds, refit_rows, store)
    elif args.command == "predict":
        predict(args, baseline, store)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
