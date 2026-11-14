"""Predeclared 1/2/4 x four-fold comparison, fixed refit and frozen evaluation."""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import ROOT, config, new_directory, source, write_json
from .data import development_identity, development_splits
from .review import require_review_receipt
from .weak_labels import require_file_label_approval
from . import decisions


def code_identity():
    return {p.name: source.sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def choose_weight(summaries):
    """Prefer the smallest feasible weight without a consistent paired-fold gain."""
    eligible = [s for s in summaries if all(f["validation_guards_passed"] for f in s["folds"])]
    if not eligible:
        raise ValueError("No weight passed all four development guards; final refit is blocked")
    selected = min(eligible, key=lambda s: s["domestic_weight"])
    for candidate in sorted(eligible, key=lambda s: s["domestic_weight"]):
        if candidate["domestic_weight"] <= selected["domestic_weight"]:
            continue
        differences = [b["domestic_target_recall"] - a["domestic_target_recall"]
                       for a, b in zip(selected["folds"], candidate["folds"])]
        old = selected["oof_metrics"]["domestic_groups"]
        new = candidate["oof_metrics"]["domestic_groups"]
        non_degrading_precision = all(new["classes"][label]["precision"] >= old["classes"][label]["precision"] - 1e-12
                                     for label in ("fire_alarm", "doorbell"))
        if min(differences) >= -1e-12 and sum(v > 1e-12 for v in differences) >= 3 and non_degrading_precision and new["macro_f1"] >= old["macro_f1"] - 1e-12:
            selected = candidate
    return selected


def summarize(weight, directories, supervision, output):
    folds, oof, validation_rows = [], [], []
    identity = development_identity()
    for i, path in enumerate(directories):
        status = source.read_json(path / "status.json")
        run = source.read_json(path / "run.json")
        if status["exit_code"] != 0 or run["development_identity"] != identity:
            raise ValueError("Incomplete or incompatible fold")
        if run["supervision"] != supervision:
            raise ValueError("Fold supervision contracts differ")
        metrics = source.read_json(path / "validation_metrics.json")
        folds.append({"fold": i, "run": str(path), "selected_epoch": status["selected_epoch"],
                      "model_sha256": source.sha256(path / "classifier.keras"),
                      "fold_selected_threshold": status["selected_threshold"],
                      "validation_guards_passed": status["validation_guards_passed"],
                      "domestic_target_recall": metrics["domestic_target_macro_recall"],
                      "overseas_other_fpr": metrics["overseas_other_clip_fpr"],
                      "domestic_doorlock_fp": metrics["domestic_doorlock_false_positive_groups"]})
        predictions = source.read_csv(path / "validation_clips.csv")
        validation_rows.append(predictions)
        oof += [r for r in predictions if r["target_domain"] == "ko_target"]
    expected = {r["sample_id"] for r in development_splits()[0] if r["target_domain"] == "ko_target"}
    if len(oof) != len(expected) or {r["sample_id"] for r in oof} != expected:
        raise ValueError("Domestic OOF coverage must be exactly once for every development sample")
    baseline, _ = source.settings()
    threshold, calibrated_metrics, pooled_metrics = decisions.calibrate_joint(
        validation_rows, baseline["validation_guards"], config()[0]["training"]["decision_threshold_grid"])
    for fold, metrics, predictions in zip(folds, calibrated_metrics, validation_rows):
        fold.update(domestic_target_recall=metrics["domestic_target_macro_recall"],
                    overseas_other_fpr=metrics["overseas_other_clip_fpr"],
                    domestic_doorlock_fp=metrics["domestic_doorlock_false_positive_groups"],
                    validation_guards_passed=bool(source.checkpoint_rank(metrics, baseline["validation_guards"])[0]),
                    common_threshold=threshold)
        source.write_csv(output / f"w{weight}_fold{fold['fold']}_common_threshold_clips.csv", decisions.apply(predictions,threshold))
    oof = decisions.apply(oof, threshold)
    values = [f["domestic_target_recall"] for f in folds]
    summary = {"kind": "dscnn_four_fold_cv", "domestic_weight": weight,
               "config": config()[0], "development_identity": identity, "code_identity": code_identity(),
               "annotations_sha256": supervision.get("annotations_sha256"),
               "supervision": supervision, "folds": folds,
               "domestic_target_recall_mean": statistics.mean(values),
               "domestic_target_recall_std": statistics.stdev(values),
               "refit_epochs": max(1, int(statistics.median(f["selected_epoch"] for f in folds) + .5)),
               "oof_metrics": pooled_metrics,
               "decision_policy":{"kind":"argmax_with_target_confidence_rejection","threshold":threshold,
                                  "selection_source":"domestic_development_OOF_and_reused_overseas_validation_only",
                                  "selection_grid":config()[0]["training"]["decision_threshold_grid"]},
               "test_accessed": False, "overseas_validation_note": "Same overseas validation reused across folds; it is not OOF.",
               "supervision_note": "Five new domestic doorbells are added. File-label mode uses clip-level loss without time annotations; supervision and training budget differ from YAMNet."}
    source.write_csv(output / f"w{weight}_folds.csv", folds)
    source.write_csv(output / f"w{weight}_oof_domestic.csv", oof)
    write_json(output / f"w{weight}_cv_summary.json", summary)
    return summary


def run_child(command, logfile):
    print("STUDY_STAGE", " ".join(command[3:]), flush=True)
    with logfile.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"First child failure exit={result.returncode}; stopped without retry. Log: {logfile}")


def supervision_arguments(args):
    if args.supervision == "file_labels":
        return ["--supervision", "file_labels", "--file-label-approval", str(args.file_label_approval.resolve())]
    return ["--supervision", "event_intervals", "--annotations", str(args.annotations.resolve())]


def run(args):
    output = new_directory(args.output)
    status = {"kind": "dscnn_real_audio_study", "stage": "preflight", "exit_code": None,
              "completed_folds": 0, "completed_weights": [], "test_evaluations": 0,
              "installation_executed": False, "cpu_training_fallback": False}
    write_json(output / "status.json", status)
    started = time.monotonic()
    try:
        preflight = source.read_json(args.gpu_preflight)
        if not preflight.get("passed") or preflight.get("exit_code") != 0:
            raise ValueError("A successful DS-CNN GPU preflight is required")
        export_preflight = source.read_json(args.gpu_preflight.parent.parent / "gpu_export_preflight/status.json")
        if not export_preflight.get("passed") or export_preflight.get("exit_code") != 0:
            raise ValueError("A successful GPU-origin int8 export preflight is required")
        if args.supervision == "file_labels":
            supervision = require_file_label_approval(args.file_label_approval)
            bag_preflight = source.read_json(args.gpu_preflight.parent.parent / "weak_label_gpu_preflight/status.json")
            if not bag_preflight.get("passed") or bag_preflight["weak_labels_code_sha256"] != source.sha256(Path(__file__).with_name("weak_labels.py")):
                raise ValueError("Current file-label optimizer requires its own successful GPU preflight")
        else:
            status["stage"] = "waiting_for_human_event_review"
            write_json(output / "status.json", status)
            if args.wait_review:
                print("WAITING_FOR_HUMAN_REVIEW", args.annotations, flush=True)
                while not args.annotations.with_suffix(".review.json").is_file():
                    time.sleep(2)
            require_review_receipt(args.annotations)
            supervision = {"mode": "event_intervals", "annotations_sha256": source.sha256(args.annotations)}
        cfg, _ = config()
        identity, code = development_identity(), code_identity()
        write_json(output / "execution_plan.json", {
            "weights": [1, 2, 4], "folds": [0, 1, 2, 3], "config": cfg,
            "run_prefix": args.run_prefix,
            "development_identity": identity, "code_identity": code,
            "supervision": supervision,
            "checkpoint_rule": "Existing source.checkpoint_rank with unchanged guards, patience and maximum epochs",
            "threshold_rule": "Select a rejection threshold per development checkpoint; then one common threshold from domestic OOF and four overseas validation predictions, enforcing each fold guard. Freeze before refit/test.",
            "weight_rule": "Smallest all-guard-feasible weight; increase only for >=3 fold wins with no fold recall loss, no target precision or pooled macro-F1 loss",
            "refit_rule": "New model and Adam, seed42, rounded median best epoch of selected four-fold CV; no validation epoch selection",
            "evaluation_rule": "Freeze checkpoint first; evaluate old 411-clip test exactly once as historical reused holdout, not a fresh independent test",
            "tracking": "Local CSV/JSON and local HTML graphs, approved by user; W&B disabled",
            "quantization_rule": "Full int8, training windows only, same frontend, report float/int8 differences without tuning on test"})
        summaries = []
        for weight in (1, 2, 4):
            directories = []
            for fold in range(4):
                if identity != development_identity() or code != code_identity():
                    raise ValueError("Data or code changed during the frozen study")
                status.update(stage="cross_validation", weight=weight, fold=fold)
                write_json(output / "status.json", status)
                name = f"{args.run_prefix}_w{weight}_f{fold}"
                command = [sys.executable, "-B", "-m", "dscnn_baseline", "train",
                           *supervision_arguments(args), "--fold", str(fold),
                           "--run-name", name, "--device", "gpu", "--domestic-weight", str(weight),
                           "--wandb-mode", "disabled", "--skip-export"]
                run_child(command, output / f"w{weight}_fold{fold}.log")
                directories.append(ROOT / "runs" / name)
                status["completed_folds"] += 1
                write_json(output / "status.json", status)
            summaries.append(summarize(weight, directories, supervision, output))
            status["completed_weights"].append(weight)
            write_json(output / "status.json", status)
        selected = choose_weight(summaries)
        selected_path = output / f"w{selected['domestic_weight']}_cv_summary.json"
        decision = {"selected_weight": selected["domestic_weight"], "refit_epochs": selected["refit_epochs"],
                    "decision_threshold":selected["decision_policy"]["threshold"],
                    "summary": str(selected_path), "summary_sha256": source.sha256(selected_path),
                    "test_seen_for_this_selection": False}
        write_json(output / "decision.json", decision)
        status["stage"] = "fixed_epoch_refit_and_int8_export"
        write_json(output / "status.json", status)
        refit_name = f"{args.run_prefix}_refit_w{selected['domestic_weight']}"
        run_child([sys.executable, "-B", "-m", "dscnn_baseline", "refit", "--cv-summary", str(selected_path),
                   *supervision_arguments(args), "--run-name", refit_name,
                   "--device", "gpu", "--wandb-mode", "disabled"], output / "refit.log")
        refit = ROOT / "runs" / refit_name
        frozen = {"model_sha256": source.sha256(refit / "classifier.keras"),
                  "int8_sha256": source.sha256(refit / "export/dscnn_int8.tflite"),
                  "development_identity": identity, "decision_sha256": source.sha256(output / "decision.json"),
                  "decision_policy":selected["decision_policy"],
                  "prediction_policy": "All 1s windows (0.5s hop + final complete window) mean softmax per file, equal-file mean per source, then argmax and fixed target confidence rejection",
                  "test_note": "Previously evaluated for YAMNet; historical comparison only, no fresh-test claim"}
        write_json(output / "frozen_model.json", frozen)
        status["stage"] = "historical_holdout_evaluation"
        write_json(output / "status.json", status)
        run_child([sys.executable, "-B", "-m", "dscnn_baseline.evaluation", "--refit", str(refit),
                   "--frozen", str(output / "frozen_model.json"), "--output", str(output / "historical_holdout")],
                  output / "historical_holdout.log")
        status.update(stage="complete", exit_code=0, refit=str(refit), decision=decision,
                      test_evaluations=1, model_development_and_training_complete=True,
                      board_validated=False, deployment_ready=False)
        print("DSCNN_REAL_AUDIO_STUDY_COMPLETE", output, flush=True)
    except Exception as error:
        status.update(stage="stopped_on_first_failure", exit_code=1,
                      first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--supervision", choices=("event_intervals", "file_labels"), default="event_intervals")
    parser.add_argument("--file-label-approval", type=Path)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--wait-review", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
