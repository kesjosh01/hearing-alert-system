"""Audit saved evidence for improvement, without rerunning training or test."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, source, write_json
from dscnn_baseline.data import development_identity
from improvement.protocol import REPORT, CANDIDATES, OPTIONS, code_identity, verify_teacher
from improvement.study import selection_rank
from improvement import policy


def audit():
    status = source.read_json(REPORT / "status.json")
    frozen = source.read_json(REPORT / "frozen_model.json")
    decision = source.read_json(REPORT / "decision.json")
    refit = Path(frozen["refit"])
    refit_status, run = source.read_json(refit / "status.json"), source.read_json(refit / "run.json")
    summaries = [source.read_json(p) for p in sorted(REPORT.glob("*_cv_summary.json"))]
    selected = next(s for s in summaries if s["candidate"] == decision["candidate"])
    checks = {}
    checks["all_16_distillation_cv_executions_completed"] = status["exit_code"] == 0 and status["completed_folds"] == 16 and status["completed_candidates"] == list(CANDIDATES)
    checks["training_code_and_data_frozen"] = frozen["code_identity"] == code_identity() and frozen["development_identity"] == development_identity() == run["development_identity"]
    checks["best_development_candidate_selected_without_test"] = decision["candidate"] == max(summaries, key=selection_rank)["candidate"] and decision["test_used_for_selection"] is False
    checks["selected_candidate_passed_all_original_guards"] = all(f["validation_guards_passed"] for f in selected["folds"])
    checks["selected_summary_unchanged"] = source.sha256(REPORT / f"{selected['candidate']}_cv_summary.json") == decision["summary_sha256"]
    guard = source.settings()[0]["validation_guards"]
    all_kd_folds = [f for s in summaries if s["candidate"] in CANDIDATES for f in s["folds"]]
    original_folds, _ = source.load_splits(config()[1])
    test_ids = {r["sample_id"] for r in original_folds[0] if r["split"] == "test"}
    cv_valid = True
    optimizer_steps = 0
    for fold in all_kd_folds:
        path = Path(fold["run"])
        state = source.read_json(path / "status.json")
        record = source.read_json(path / "run.json")
        teacher = source.read_json(path / "teacher_provenance.json")
        used = source.read_csv(path / "used_train_validation.csv")
        train_ids = {r["sample_id"] for r in used if r["split"] == "train"}
        arrays = np.load(path / "training_teacher_logits.npz", allow_pickle=False)
        history = source.read_csv(path / "history.csv")
        cv_valid &= (state["exit_code"] == 0 and state["distillation_applied"] is True and
                     state["distillation_optimizer_steps"] == state["optimizer_steps"] == len(history) * OPTIONS["steps_per_epoch"] and
                     state["save_reload_verified"] is True and all("GPU:0" in d for d in state["model_variable_devices"]) and
                     record["code_identity"] == frozen["code_identity"] and record["teacher"] == verify_teacher(fold["fold"]) and
                     set(arrays.files) == train_ids == set(teacher["student_training_sample_ids"]) and
                     teacher["teacher_weights_unchanged"] is True and teacher["soft_targets_sha256"] == source.sha256(path / "training_teacher_logits.npz") and
                     source.sha256(path / "classifier.keras") == state["model_sha256"] == fold["model_sha256"] and
                     not test_ids.intersection(r["sample_id"] for r in used) and
                     all(np.isfinite(float(r["distillation_loss_T_squared"])) and float(r["distillation_loss_T_squared"]) >= -1e-5 for r in history))
        arrays.close()
        optimizer_steps += state["optimizer_steps"]
    checks["actual_KL_training_gpu_and_fold_teacher_provenance_verified"] = bool(cv_valid)
    common_metrics_valid = True
    for summary in summaries:
        thresholds = summary["decision_policy"]["thresholds"]
        for fold in summary["folds"]:
            metrics = policy.report(source.read_csv(Path(fold["run"]) / "validation_clips.csv"), thresholds)
            common_metrics_valid &= abs(metrics["domestic_target_macro_recall"] - fold["domestic_target_recall"]) < 1e-10
            common_metrics_valid &= bool(source.checkpoint_rank(metrics, guard)[0]) == fold["validation_guards_passed"]
    checks["all_reported_development_metrics_match_predictions"] = bool(common_metrics_valid)
    expected_epochs = max(1, int(statistics.median(f["selected_epoch"] for f in selected["folds"]) + .5))
    checks["refit_epochs_fixed_from_selected_cv"] = expected_epochs == decision["refit_epochs"] == run["refit_epochs"] == refit_status["epochs_completed"] == refit_status["selected_epoch"]
    checks["refit_training_and_distillation_role_truthful"] = refit_status["exit_code"] == 0 and refit_status["distillation_applied"] == decision["distillation_selected"] and refit_status["optimizer_steps"] == expected_epochs * OPTIONS["steps_per_epoch"]
    checks["refit_teacher_excludes_test"] = run["teacher"] == verify_teacher()
    used = source.read_csv(refit / "used_train_validation.csv")
    checks["test_excluded_from_refit"] = not test_ids.intersection(r["sample_id"] for r in used)
    exposure = {r["sample_id"]: int(r["draws"]) for r in source.read_csv(refit / "sample_exposure.csv")}
    new_ids = [f"kr-user-db-{i:02}" for i in range(6, 11)]
    checks["all_new_doorbells_actually_sampled"] = all(exposure.get(sample, 0) > 0 for sample in new_ids)
    train_ids = {r["sample_id"] for r in used if r["split"] == "train"}
    calibration = source.read_json(refit / "export/calibration_provenance.json")
    checks["int8_calibration_training_only"] = len(calibration) == 128 and all(r["split"] == "train" and r["sample_id"] in train_ids for r in calibration)
    metadata = source.read_json(refit / "export/model_metadata.json")
    checks["full_integer_trained_model_contract"] = metadata["pipeline_only"] is False and metadata["calibration_role"] == "train" and set(metadata["all_tensor_types"]) <= {"int8", "int32"} and metadata["input"]["shape"] == [1, 98, 40, 1] and metadata["output"]["shape"] == [1, 3]
    holdout = source.read_json(REPORT / "historical_holdout/status.json")
    checks["one_frozen_retrospective_comparison_after_selection"] = holdout["exit_code"] == 0 and holdout["test_evaluation_invocations"] == 1 and holdout["is_fresh_independent_test"] is False and holdout["evaluated_clips"] == len(test_ids)
    checks["model_hashes_fixed_before_and_after_comparison"] = source.sha256(refit / "classifier.keras") == frozen["model_sha256"] == holdout["model_sha256_before"] == holdout["model_sha256_after"] and source.sha256(refit / "export/dscnn_int8.tflite") == frozen["int8_sha256"] == holdout["int8_sha256_before"] == holdout["int8_sha256_after"] == metadata["model_sha256"]
    frontend = source.read_json(REPORT / "trained_frontend_host/status.json")
    checks["trained_cpp_frontend_host_parity"] = frontend["passed"] is True and frontend["exit_code"] == 0 and len(frontend["signals"]) == 27 and all(r["int8_model_output_identical"] for r in frontend["signals"])
    protected = source.read_json(REPORT / "preserved_baseline_before.json")
    checks["previous_baseline_artifacts_preserved"] = all(Path(p).is_file() and source.sha256(p) == digest for p, digest in protected.items())
    checks["originals_yamnet_splits_and_all_environments_preserved"] = source.read_json(REPORT / "integrity.json")["unchanged"] is True
    checks["policy_and_leakage_tests_passed"] = "\nOK\n" in (REPORT / "policy_tests_final_v2.log").read_text()
    checks["synthetic_KL_formula_and_gpu_export_preflight_passed"] = source.read_json(REPORT / "student_preflight/status.json")["passed"] is True
    original = source.read_json(ROOT / "reports/real_training_20261114/study_calibrated_v2_resumed/w1_cv_summary.json")
    old_test = source.read_json(ROOT / "reports/real_training_20261114/study_calibrated_v2_resumed/historical_holdout/int8_metrics.json")
    new_test = source.read_json(REPORT / "historical_holdout/int8_metrics.json")
    checks["development_recall_improved_vs_original_selected_student"] = selected["domestic_target_recall_mean"] > original["domestic_target_recall_mean"] + 1e-12
    result = {"kind": "dscnn_improvement_artifact_audit", "model_development_and_training_complete": all(checks.values()),
              "checks": checks, "distillation_applied_to_selected_model": decision["distillation_selected"],
              "actual_distillation_cv_runs": 16, "actual_distillation_cv_optimizer_steps": optimizer_steps,
              "original_selected_development_recall": original["domestic_target_recall_mean"],
              "selected_development_recall": selected["domestic_target_recall_mean"],
              "selected_candidate": decision["candidate"], "new_doorbell_refit_draws": {s: exposure[s] for s in new_ids},
              "trained_model": str(refit / "classifier.keras"), "trained_int8_model": str(refit / "export/dscnn_int8.tflite"),
              "trained_int8_bytes": metadata["model_bytes"], "event_intervals_reviewed": False,
              "independent_test_available": False, "board_executions": 0, "deployment_ready": False,
              "practical_model_replacement_recommended": False,
              "retrospective_performance_tradeoff": {
                  "old_target_recall": old_test["domestic_target_macro_recall"],
                  "new_target_recall": new_test["domestic_target_macro_recall"],
                  "old_foreign_other_false_positives": old_test["overseas_other_clip_false_positives"],
                  "new_foreign_other_false_positives": new_test["overseas_other_clip_false_positives"],
                  "foreign_other_clips": new_test["overseas_other_clip_count"],
                  "new_foreign_other_fpr": new_test["overseas_other_clip_fpr"],
                  "new_historical_fpr_below_development_limit": new_test["overseas_other_clip_fpr"] <= guard["max_overseas_other_clip_fpr"],
                  "note": "Recall increased but false positives increased. Development guards passed; retrospective foreign FPR did not meet the same 5% number. Do not describe this as an across-metric improvement or deployable detector."},
              "limitations": ["Model and thresholds were selected on the same small development corpus; OOF model-selection scores can be optimistic.",
                              "Old baseline checkpoints were selected with a common threshold before retrospective class-specific recalibration. Distillation comparisons do not isolate KD alone from checkpoint selection.",
                              "Final old-test comparison is retrospective because this holdout was already observed. It is not a fresh independent test.",
                              "Live microphone, event timing, false alarms/hour, full ESP-IDF build and board RAM/latency remain unverified."]}
    write_json(REPORT / "FINAL_VERIFICATION.json", result)
    if not result["model_development_and_training_complete"]:
        raise ValueError(f"Improvement audit failed: {[key for key, passed in checks.items() if not passed]}")
    print("DSCNN_IMPROVEMENT_ARTIFACT_AUDIT_PASS")
    return result


if __name__ == "__main__":
    audit()
