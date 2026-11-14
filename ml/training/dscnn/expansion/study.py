"""Recorded, sequential teacher/data/architecture comparison and fixed refit."""
from __future__ import annotations

import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

from dscnn_baseline.common import ROOT, source, write_json
from improvement import policy
from .protocol import CANDIDATES, OPTIONS, REPORT
from .runtime import frozen_inputs
from .train import run_directory


def child(command, log):
    if log.exists():
        raise FileExistsError(f"Preserve prior execution log: {log}")
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"First failed child: exit{completed.returncode}; inspect {log}")


def summarize(role, candidate):
    directories = [run_directory(role, i, candidate) for i in range(4)]
    rows = [source.read_csv(directory / "validation_clips.csv") for directory in directories]
    guards = source.settings()[0]["validation_guards"]
    thresholds, metrics, oof, _ = policy.calibrate(rows, guards)
    details = []
    for i, (directory, predictions, metric) in enumerate(zip(directories, rows, metrics)):
        status = source.read_json(directory / "status.json")
        if status["exit_code"] != 0 or status["test_evaluated"] is not False:
            raise ValueError("Incomplete or test-contaminated CV contributor")
        applied = policy.apply(predictions, thresholds)
        foreign_targets = [r for r in applied if r["target_domain"] != "ko_target" and r["label"] != "other"]
        details.append({"fold": i, "run": str(directory), "selected_epoch": status["selected_epoch"],
                        "model_sha256": status["model_sha256"], "domestic_target_recall": metric["domestic_target_macro_recall"],
                        "overseas_other_fpr": metric["overseas_other_clip_fpr"], "overseas_other_count": metric["overseas_other_clip_count"],
                        "overseas_other_fp": metric["overseas_other_clip_false_positives"],
                        "domestic_doorlock_fp": metric["domestic_doorlock_false_positive_groups"],
                        "validation_guards_passed": bool(source.checkpoint_rank(metric, guards)[0]),
                        "foreign_positive_correct": sum(r["label"] == r["predicted_label"] for r in foreign_targets),
                        "foreign_positive_support": len(foreign_targets)})
    run = source.read_json(directories[0] / "run.json")
    result = {"role": role, "candidate": candidate, "folds": details,
              "domestic_target_recall_mean": float(np.mean([f["domestic_target_recall"] for f in details])),
              "domestic_target_recall_std": float(np.std([f["domestic_target_recall"] for f in details], ddof=1)),
              "oof_metrics": oof, "parameters": 131587 if role == "teacher" else run["complexity"]["parameters_including_bn_statistics"],
              "refit_epochs": int(statistics.median(f["selected_epoch"] for f in details) + .5),
              "decision_policy": {"kind": "per_target_argmax_confidence_rejection", "thresholds": thresholds,
                                  "clip_aggregation": "all_window_mean_softmax", "group_aggregation": "equal_clip_mean_softmax_then_decide",
                                  "selection_source": "domestic_OOF_and_corrected_foreign_validation_only"},
              "new_public_files_role": "train_only", "fixed_test_used": False}
    name = "teacher" if role == "teacher" else candidate
    write_json(REPORT / f"{name}_cv_summary.json", result)
    oof_rows = [r for predictions in rows for r in predictions if r["target_domain"] == "ko_target"]
    source.write_csv(REPORT / f"{name}_oof_domestic.csv", policy.apply(oof_rows, thresholds))
    return result


def selection_rank(summary):
    folds = summary["folds"]
    domestic = summary["oof_metrics"]["domestic_groups"]
    return (int(all(f["validation_guards_passed"] for f in folds)), summary["domestic_target_recall_mean"],
            domestic["macro_f1"], min(domestic["classes"][c]["precision"] for c in ("fire_alarm", "doorbell")),
            -summary["parameters"], -float(np.mean([f["overseas_other_fpr"] for f in folds])))


def execute():
    plan = frozen_inputs()
    status_path = REPORT / "status.json"
    if status_path.exists():
        raise FileExistsError("Coordinator already started; inspect its actual process and receipts before resuming")
    started = time.monotonic()
    status = {"stage": "corrected_teacher_cv", "coordinator_pid": os.getpid(), "exit_code": None,
              "completed_teacher_folds": 0, "completed_student_folds": 0, "historical_test_evaluations": 0,
              "new_public_files": plan["dataset"]["accepted_new_public_files"], "apps_or_packages_installed": False}
    write_json(status_path, status)
    prefix = [sys.executable, "-B", "-m", "expansion.train"]
    try:
        for fold in range(4):
            status.update(fold=fold)
            write_json(status_path, status)
            child(prefix + ["--role", "teacher", "--fold", str(fold)], REPORT / f"teacher_fold{fold}.log")
            status["completed_teacher_folds"] += 1
            write_json(status_path, status)
        teacher = summarize("teacher", "freq24")
        teacher_decision = {"refit_epochs": teacher["refit_epochs"], "decision_policy": teacher["decision_policy"]}
        write_json(REPORT / "teacher_decision.json", teacher_decision)
        status["stage"] = "corrected_teacher_refit"
        write_json(status_path, status)
        child(prefix + ["--role", "teacher", "--refit-epochs", str(teacher["refit_epochs"]), "--decision", str(REPORT / "teacher_decision.json")], REPORT / "teacher_refit.log")
        summaries = []
        status["stage"] = "architecture_cv"
        for candidate in CANDIDATES:
            for fold in range(4):
                status.update(candidate=candidate, fold=fold)
                write_json(status_path, status)
                child(prefix + ["--role", "student", "--candidate", candidate, "--fold", str(fold)], REPORT / f"{candidate}_fold{fold}.log")
                status["completed_student_folds"] += 1
                write_json(status_path, status)
            summaries.append(summarize("student", candidate))
        chosen = max(summaries, key=selection_rank)
        if not selection_rank(chosen)[0]:
            raise ValueError("No architecture passed all development guards")
        decision = {"candidate": chosen["candidate"], "refit_epochs": chosen["refit_epochs"], "decision_policy": chosen["decision_policy"],
                    "domestic_weight": OPTIONS["domestic_weight"], "selection_rank": selection_rank(chosen),
                    "summary_sha256": source.sha256(REPORT / f"{chosen['candidate']}_cv_summary.json"), "fixed_test_used": False}
        write_json(REPORT / "decision.json", decision)
        status.update(stage="fixed_student_refit", decision=decision)
        write_json(status_path, status)
        child(prefix + ["--role", "student", "--candidate", chosen["candidate"], "--refit-epochs", str(chosen["refit_epochs"]),
                        "--decision", str(REPORT / "decision.json")], REPORT / "student_refit.log")
        final = run_directory("student", None, chosen["candidate"])
        write_json(REPORT / "frozen_model.json", {"refit": str(final), "decision": decision,
                   "model_sha256": source.sha256(final / "classifier.keras"), "int8_sha256": source.sha256(final / "export/dscnn_int8.tflite"),
                   "code_identity": plan["code_identity"], "dataset_identity": plan["dataset_identity"]})
        status.update(stage="training_complete", exit_code=0, teacher_retrained_for_new_labels=True, distillation_applied=True,
                      model_development_and_training_complete=True, deployment_ready=False)
        print("EXPANDED_TRAINING_COMPLETE", decision, flush=True)
    except Exception as error:
        status.update(stage="stopped_on_first_failure", exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (REPORT / "first_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(status_path, status)


if __name__ == "__main__":
    execute()
