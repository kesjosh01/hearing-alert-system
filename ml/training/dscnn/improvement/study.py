"""Frozen development-only distillation ablations, then one fixed refit."""
from __future__ import annotations

import statistics
import sys
import time
import traceback
from pathlib import Path

from dscnn_baseline.common import ROOT, source, write_json
from dscnn_baseline.data import development_identity, development_splits
from dscnn_baseline.study import run_child
from .protocol import REPORT, CANDIDATES, code_identity, plan
from . import policy


def summarize(name, directories, domestic_weight=1):
    predictions = [source.read_csv(path / "validation_clips.csv") for path in directories]
    expected = {r["sample_id"] for r in development_splits()[0] if r["target_domain"] == "ko_target"}
    oof = [r for rows in predictions for r in rows if r["target_domain"] == "ko_target"]
    if len(oof) != len(expected) or {r["sample_id"] for r in oof} != expected:
        raise ValueError("Every domestic development sample must occur in OOF exactly once")
    guards = source.settings()[0]["validation_guards"]
    thresholds, metrics, pooled, rank = policy.calibrate(predictions, guards)
    folds = []
    for i, (path, metric) in enumerate(zip(directories, metrics)):
        status = source.read_json(path / "status.json")
        run = source.read_json(path / "run.json")
        if status["exit_code"] != 0 or run["development_identity"] != development_identity():
            raise ValueError("Incomplete/incompatible improvement fold")
        folds.append({"fold": i, "run": str(path), "selected_epoch": status["selected_epoch"],
                      "model_sha256": source.sha256(path / "classifier.keras"),
                      "domestic_target_recall": metric["domestic_target_macro_recall"],
                      "overseas_other_fpr": metric["overseas_other_clip_fpr"],
                      "domestic_doorlock_fp": metric["domestic_doorlock_false_positive_groups"],
                      "validation_guards_passed": bool(source.checkpoint_rank(metric, guards)[0])})
    values = [f["domestic_target_recall"] for f in folds]
    parameters = source.read_json(directories[0] / "run.json")["complexity"]["parameters_including_bn_statistics"]
    summary = {"candidate": name, "domestic_weight": domestic_weight, "folds": folds,
               "development_identity": development_identity(), "code_identity": code_identity(),
               "domestic_target_recall_mean": statistics.mean(values), "domestic_target_recall_std": statistics.stdev(values),
               "refit_epochs": max(1, int(statistics.median(f["selected_epoch"] for f in folds) + .5)),
               "oof_metrics": pooled, "parameters": parameters,
               "decision_policy": {"kind": "per_target_argmax_confidence_rejection", "thresholds": thresholds,
                                   "clip_aggregation": "all_window_mean_softmax", "group_aggregation": "equal_clip_mean_softmax_then_decide",
                                   "selection_source": "domestic_OOF_and_reused_foreign_validation_only"},
               "test_accessed_for_selection": False}
    write_json(REPORT / f"{name}_cv_summary.json", summary)
    source.write_csv(REPORT / f"{name}_oof_domestic.csv", policy.apply(oof, thresholds))
    return summary


def selection_rank(summary):
    classes = summary["oof_metrics"]["domestic_groups"]["classes"]
    return (int(all(f["validation_guards_passed"] for f in summary["folds"])),
            summary["domestic_target_recall_mean"], summary["oof_metrics"]["domestic_groups"]["macro_f1"],
            min(classes[label]["precision"] for label in ("fire_alarm", "doorbell")),
            -summary["parameters"], -statistics.mean(f["overseas_other_fpr"] for f in summary["folds"]))


def run():
    status_path = REPORT / "status.json"
    if status_path.exists():
        raise FileExistsError("Improvement study already exists; inspect before resuming")
    status = {"kind": "dscnn_distillation_improvement_study", "stage": "validation", "exit_code": None,
              "completed_candidates": [], "completed_folds": 0, "test_evaluations": 0,
              "installation_executed": False, "cpu_training_fallback": False}
    write_json(status_path, status)
    started = time.monotonic()
    try:
        if "\nOK\n" not in (REPORT / "policy_tests_final_v2.log").read_text():
            raise ValueError("Policy/leakage unit checks must pass before real training")
        preflight = source.read_json(REPORT / "student_preflight/status.json")
        current = code_identity()
        if not preflight.get("passed") or preflight["code_identity"]["improvement/model.py"] != current["improvement/model.py"]:
            raise ValueError("Current optimizer/model requires a successful synthetic GPU preflight")
        frozen = plan()
        write_json(REPORT / "execution_plan.json", frozen)
        previous = ROOT / "reports/real_training_20261114/study_calibrated_v2_resumed"
        baseline_plan = source.read_json(previous / "execution_plan.json")
        if any(current[f"dscnn_baseline/{name}"] != digest for name, digest in baseline_plan["code_identity"].items()):
            raise ValueError("Preserved baseline code changed")
        summaries = []
        for weight in (1, 2, 4):
            old = source.read_json(previous / f"w{weight}_cv_summary.json")
            directories = [Path(f["run"]) for f in old["folds"]]
            if any(source.sha256(p / "classifier.keras") != f["model_sha256"] for p, f in zip(directories, old["folds"])):
                raise ValueError("Baseline checkpoint changed")
            summaries.append(summarize(f"baseline_w{weight}", directories, weight))
        for candidate in CANDIDATES:
            directories = []
            for fold in range(4):
                if frozen["code_identity"] != code_identity() or frozen["development_identity"] != development_identity():
                    raise ValueError("Frozen improvement inputs changed during study")
                status.update(stage="distillation_cross_validation", candidate=candidate, fold=fold)
                write_json(status_path, status)
                name = f"dscnn_improvement_20261114_{candidate}_f{fold}"
                run_child([sys.executable, "-B", "-m", "improvement.train", "--candidate", candidate,
                           "--fold", str(fold), "--run-name", name], REPORT / f"{candidate}_fold{fold}.log")
                directories.append(ROOT / "runs" / name)
                status["completed_folds"] += 1
                write_json(status_path, status)
            summaries.append(summarize(candidate, directories))
            status["completed_candidates"].append(candidate)
            write_json(status_path, status)
        chosen = max(summaries, key=selection_rank)
        if not selection_rank(chosen)[0]:
            raise ValueError("No candidate passed all development guards")
        decision = {"candidate": chosen["candidate"], "domestic_weight": chosen["domestic_weight"],
                    "refit_epochs": chosen["refit_epochs"], "decision_policy": chosen["decision_policy"],
                    "summary_sha256": source.sha256(REPORT / f"{chosen['candidate']}_cv_summary.json"),
                    "selection_rank": selection_rank(chosen), "test_used_for_selection": False,
                    "distillation_selected": chosen["candidate"] in CANDIDATES}
        write_json(REPORT / "decision.json", decision)
        status["stage"] = "fixed_refit_and_int8_export"; write_json(status_path, status)
        candidate = chosen["candidate"] if decision["distillation_selected"] else "kd24"
        name = "dscnn_improvement_20261114_final"
        command = [sys.executable, "-B", "-m", "improvement.train", "--candidate", candidate,
                   "--run-name", name, "--refit-epochs", str(chosen["refit_epochs"]),
                   "--decision", str(REPORT / "decision.json"), "--domestic-weight", str(chosen["domestic_weight"])]
        if not decision["distillation_selected"]:
            command.append("--hard-label-only")
        run_child(command, REPORT / "refit.log")
        refit = ROOT / "runs" / name
        write_json(REPORT / "frozen_model.json", {"refit": str(refit), "decision": decision,
                   "model_sha256": source.sha256(refit / "classifier.keras"),
                   "int8_sha256": source.sha256(refit / "export/dscnn_int8.tflite"),
                   "code_identity": frozen["code_identity"], "development_identity": frozen["development_identity"]})
        status.update(stage="training_complete", exit_code=0, decision=decision, refit=str(refit),
                      distillation_cv_executions=16, model_development_and_training_complete=True,
                      deployment_ready=False, historical_holdout_repeated_during_search=False)
        print("DSCNN_IMPROVEMENT_TRAINING_COMPLETE", flush=True)
    except Exception as error:
        status.update(stage="stopped_on_first_failure", exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (REPORT / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(status_path, status)


if __name__ == "__main__":
    run()
