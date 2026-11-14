"""Requirement-level audit of the collected data and actual expanded training."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, source, write_json
from expansion.data import identity, read_rows
from expansion.protocol import CANDIDATES, DATA, OPTIONS, REPORT, code_identity
from expansion.study import selection_rank
from expansion.train import run_directory
from improvement import policy


def audit(final=False):
    plan = source.read_json(REPORT / "frozen_plan.json")
    status = source.read_json(REPORT / "status.json")
    frozen = source.read_json(REPORT / "frozen_model_coverage.json")
    coverage_amendment = source.read_json(REPORT / "refit_coverage_amendment.json")
    coverage_status = source.read_json(REPORT / "coverage_refit_status.json")
    decision = source.read_json(REPORT / "decision.json")
    public = source.read_json(DATA / "accepted_public.json")
    dataset = source.read_json(DATA / "dataset.json")
    summaries = [source.read_json(REPORT / f"{c}_cv_summary.json") for c in CANDIDATES]
    selected = next(s for s in summaries if s["candidate"] == decision["candidate"])
    refit = Path(frozen["refit"])
    checks = {}
    checks["public_audio_actually_collected_and_licensed"] = len(public) == 39 and all(
        source.sha256(ROOT / r["relative_path"]) == r["source_sha256"] and r["license_id"] in ("CC-BY-4.0", "CC0-1.0")
        and any(s.get("codec_type") == "audio" for s in r["probe"]["streams"]) for r in public)
    checks["korean_first_and_overseas_shortfall_supplemented"] = Counter((r["origin"], r["label"]) for r in public) == {
        ("KR_public_catalog", "doorbell"): 24, ("overseas_public_catalog", "fire_alarm"): 9,
        ("overseas_public_catalog", "doorbell"): 6}
    checks["positive_class_data_ratio_and_source_counts_recorded"] = .8 <= dataset["fire_to_doorbell_training_clip_ratio"] <= 1.25 and dataset["training_groups_by_class"]["fire_alarm"] == 31 and dataset["training_groups_by_class"]["doorbell"] == 32
    checks["code_and_dataset_frozen_before_real_training"] = code_identity() == plan["code_identity"] == frozen["code_identity"] and identity() == plan["dataset_identity"] == frozen["dataset_identity"]
    overrides = source.read_csv(DATA / "label_overrides.csv")
    checks["training24_and_validation8_sirens_positive"] = Counter(r["split"] for r in overrides) == {"train": 24, "validation": 8} and all(r["new_label"] == "fire_alarm" for r in overrides)
    original_folds, _ = source.load_splits(config()[1])
    test_ids = {r["sample_id"] for r in original_folds[0] if r["split"] == "test"}
    test_groups = {r["split_group_id"] for r in original_folds[0] if r["split"] == "test"}
    runs = [run_directory("teacher", fold) for fold in range(4)] + [run_directory("teacher")]
    runs += [run_directory("student", fold, candidate) for candidate in CANDIDATES for fold in range(4)]
    runs += [Path(source.read_json(REPORT / "frozen_model.json")["refit"]), refit]
    run_valid, sampler_valid, teacher_valid, loss_valid = [], [], [], []
    teacher_steps = student_steps = 0
    for directory in runs:
        state, run = source.read_json(directory / "status.json"), source.read_json(directory / "run.json")
        used = source.read_csv(directory / "used_train_validation.csv")
        history = source.read_csv(directory / "history.csv")
        expected = read_rows(state["fold"])
        fitting = [r for r in used if r["split"] == "train"]
        validating = [r for r in used if r["split"] == "validation"]
        fitting_groups = {r["split_group_id"] for r in fitting}
        run_valid.append(state["exit_code"] == 0 and state["optimizer_steps"] == len(history) * 30 and state["save_reload_verified"]
                         and source.fingerprint(used) == source.fingerprint(expected)
                         and run["code_identity"] == plan["code_identity"] and run["dataset_identity"] == identity()
                         and state["training_siren_clips"] == 24 and all("GPU:0" in device for device in state["model_variable_devices"])
                         and not test_ids.intersection(r["sample_id"] for r in used) and not test_groups.intersection(fitting_groups)
                         and not fitting_groups.intersection(r["split_group_id"] for r in validating)
                         and source.sha256(directory / "classifier.keras") == state["model_sha256"])
        sampling = source.read_csv(directory / "sampling_groups.csv")
        okay = all(np.isclose(sum(float(r["group_probability"]) for r in sampling if r["label"] == label), 1 / 3) for label in source.LABELS)
        for label in ("fire_alarm", "doorbell"):
            domestic = [float(r["group_probability"]) for r in sampling if r["label"] == label and r["domain_weight"] == "4"]
            foreign = [float(r["group_probability"]) for r in sampling if r["label"] == label and r["domain_weight"] == "1"]
            okay = okay and bool(domestic and foreign) and all(np.isclose(p / foreign[0], 4) for p in domestic)
        sampler_valid.append(okay)
        if state["role"] == "teacher":
            teacher_steps += state["optimizer_steps"]
        else:
            student_steps += state["optimizer_steps"]
            receipt = source.read_json(directory / "teacher_provenance.json")
            teacher = run_directory("teacher", state["fold"])
            teacher_used = source.read_csv(teacher / "used_train_validation.csv")
            teacher_valid.append(receipt["directory"] == str(teacher) and receipt["model_sha256"] == source.sha256(teacher / "classifier.keras")
                                 and receipt["soft_logits_sha256"] == source.sha256(directory / "training_teacher_logits.npz")
                                 and receipt["teacher_weights_unchanged"] and not receipt["validation_group_leakage"]
                                 and {r["sample_id"] for r in teacher_used if r["split"] == "train"} == {r["sample_id"] for r in fitting})
            loss_valid.append(state["distillation_optimizer_steps"] == state["optimizer_steps"] > 0 and all(
                float(r["distillation_loss_T_squared"]) >= 0 and np.isclose(float(r["train_loss"]), .5 * float(r["hard_label_loss"]) + .5 * float(r["distillation_loss_T_squared"]), atol=2e-6) for r in history))
    checks["all15_actual_training_runs_gpu_and_split_scope_verified"] = len(runs) == 15 and all(run_valid) and status["completed_teacher_folds"] == 4 and status["completed_student_folds"] == 8
    amended_run = source.read_json(refit / "run.json")
    schedule = {r["sample_id"]: int(r["expected_draws"]) for r in source.read_csv(REPORT / "refit_coverage_schedule.csv")}
    actual_schedule = {r["sample_id"]: int(r["draws"]) for r in source.read_csv(refit / "sample_exposure.csv")}
    checks["coverage_seed_amended_before_test_without_score_selection"] = coverage_status["exit_code"] == 0 and not coverage_status["test_used_to_select_seed"] and coverage_amendment["new_test_evaluations_before_amendment"] == 0 and actual_schedule == schedule and amended_run["seed"] == coverage_amendment["selected_refit_seed"] and amended_run["options"] == dict(OPTIONS, seed=coverage_amendment["selected_refit_seed"]) and amended_run["refit_coverage_amendment"]["sha256"] == source.sha256(REPORT / "refit_coverage_amendment.json") and coverage_amendment["wrapper_sha256"] == source.sha256(ROOT / "tools/refit_expanded_with_coverage.py")
    checks["class_mass_thirds_and_domestic_to_foreign_group_weight4_in_every_run"] = all(sampler_valid)
    checks["corrected_label_teachers_and_aligned_KL_actually_used"] = all(teacher_valid) and all(loss_valid)
    checks["architecture_comparison_and_selection_without_test"] = decision["candidate"] == max(summaries, key=selection_rank)["candidate"] and not decision["fixed_test_used"] and all(f["validation_guards_passed"] for f in selected["folds"])
    metric_valid = []
    for summary in summaries:
        for fold in summary["folds"]:
            metric = policy.report(source.read_csv(Path(fold["run"]) / "validation_clips.csv"), summary["decision_policy"]["thresholds"])
            metric_valid.append(np.isclose(metric["domestic_target_macro_recall"], fold["domestic_target_recall"]) and np.isclose(metric["overseas_other_clip_fpr"], fold["overseas_other_fpr"]) and metric["overseas_other_clip_count"] == 390)
    checks["all_reported_cv_metrics_match_saved_predictions"] = all(metric_valid)
    epochs = int(statistics.median(f["selected_epoch"] for f in selected["folds"]) + .5)
    final_status = source.read_json(refit / "status.json")
    checks["final_refit_epochs_fixed_from_cv"] = decision["refit_epochs"] == epochs == final_status["epochs_completed"] == final_status["selected_epoch"]
    exposure = {r["sample_id"]: int(r["draws"]) for r in source.read_csv(refit / "sample_exposure.csv")}
    checks["all39_public_files_actually_sampled_in_final_training"] = all(exposure.get(r["sample_id"], 0) > 0 for r in public)
    checks["all24_siren_files_actually_sampled_as_fire_alarm"] = all(exposure.get(r["sample_id"], 0) > 0 for r in overrides if r["split"] == "train")
    model, integer = refit / "classifier.keras", refit / "export/dscnn_int8.tflite"
    metadata = source.read_json(refit / "export/model_metadata.json")
    calibration = source.read_json(refit / "export/calibration_provenance.json")
    train_ids = {r["sample_id"] for r in read_rows() if r["split"] == "train"}
    checks["trained_float_and_full_int8_models_match_freeze"] = source.sha256(model) == frozen["model_sha256"] and source.sha256(integer) == frozen["int8_sha256"] == metadata["model_sha256"] and metadata["pipeline_only"] is False and set(metadata["all_tensor_types"]) <= {"int8", "int32"}
    checks["int8_calibration_training_only"] = len(calibration) == 128 and all(r["split"] == "train" and r["sample_id"] in train_ids for r in calibration)
    comparison = source.read_json(REPORT / "historical_holdout/status.json")
    scope = source.read_json(REPORT / "historical_holdout/evaluation_scope.json")
    checks["one_retrospective_comparison_after_freeze_with_sirens_excluded"] = comparison["exit_code"] == 0 and comparison["test_evaluation_invocations"] == 1 and comparison["evaluated_clips"] == 403 and comparison["model_unchanged"] and scope["foreign_negative_count"] == 394 and len(scope["excluded_ids"]) == 8 and not scope["eligibility_depends_on_model_predictions"]
    frontend = source.read_json(REPORT / "trained_frontend_host/status.json")
    checks["cpp_frontend_and_actual_int8_host_outputs_match"] = frontend["passed"] and len(frontend["signals"]) == 27 and all(r["int8_model_output_identical"] for r in frontend["signals"])
    preserved = source.read_json(REPORT / "preserved_previous_before.json")
    checks["previous_dscnn_code_models_and_data_preserved"] = all(Path(p).is_file() and source.sha256(p) == digest for p, digest in preserved.items())
    checks["originals_yamnet_and_all_environments_preserved"] = source.read_json(REPORT / "integrity.json")["unchanged"] is True
    checks["synthetic_gpu_integer_and_data_contract_gates_passed"] = source.read_json(REPORT / "preflight.json")["passed"] and "\nOK\n" in (REPORT / "data_tests.log").read_text()
    if final:
        bundle = ROOT.parents[1] / "models/dscnn_expanded_20261114"
        manifest = source.read_json(bundle / "release_manifest.json")
        cli = source.read_json(REPORT / "offline_bundle_check/status.json")
        checks["local_model_bundle_hashes_and_offline_cli_verified"] = all(source.sha256(bundle / name) == digest for name, digest in manifest["files"].items()) and cli["passed"] and cli["bundle_files_unchanged"]
    result = {"kind": "expanded_dscnn_requirement_audit", "model_development_and_training_complete": all(checks.values()),
              "checks": checks, "public_audio_files_added": len(public), "training_sirens_reclassified": 24,
              "domestic_positive_group_weight": 4, "overseas_positive_group_weight": 1,
              "actual_teacher_training_runs": 5, "actual_student_distillation_training_runs": 10,
              "initial_refit_preserved_for_target_coverage_gap": True,
              "final_refit_seed": coverage_amendment["selected_refit_seed"],
              "actual_teacher_optimizer_steps": teacher_steps, "actual_student_optimizer_steps": student_steps,
              "selected_architecture": decision["candidate"], "structure_changed_from_previous": decision["candidate"] == "temporal24",
              "selected_cv_target_recall": selected["domestic_target_recall_mean"], "refit_epochs": epochs,
              "trained_model": str(model), "trained_int8_model": str(integer), "trained_int8_bytes": integer.stat().st_size,
              "new_public_refit_draws": {r["sample_id"]: exposure.get(r["sample_id"], 0) for r in public},
              "independent_new_test_available": False, "device_country_of_public_catalog_recordings_verified": False,
              "public_file_labels_basis": "Publisher category/title; no human listening verification claimed",
              "event_intervals_human_reviewed": False, "apps_packages_or_drivers_installed": False,
              "deployment_ready": False, "board_executions": 0}
    historical = source.read_json(REPORT / "historical_holdout/int8_metrics.json")
    previous = next(r for r in source.read_json(REPORT / "historical_holdout/previous_models_same_scope.json") if r["model"] == "previous_distilled" and r["backend"] == "int8")
    result.update(practical_model_replacement_recommended=False,
                  retrospective_performance={"previous_target_recall": previous["domestic_target_macro_recall"],
                     "current_target_recall": historical["domestic_target_macro_recall"],
                     "previous_foreign_false_positives": previous["false_positives"],
                     "current_foreign_false_positives": historical["overseas_other_clip_false_positives"],
                     "eligible_foreign_negatives": historical["overseas_other_clip_count"],
                     "current_foreign_fpr": historical["overseas_other_clip_fpr"],
                     "across_metric_improvement_demonstrated": historical["domestic_target_macro_recall"] >= previous["domestic_target_macro_recall"] and historical["overseas_other_clip_fpr"] <= previous["fpr"],
                     "note": "Training completion is separate from practical performance. Report regressions; no test-driven retuning or replacement recommendation."})
    refit_validation = source.read_json(refit / "validation_metrics.json")
    result["final_refit_validation_guard"] = {"model_backend": "float", "foreign_false_positives": refit_validation["overseas_other_clip_false_positives"],
               "foreign_negative_count": refit_validation["overseas_other_clip_count"],
               "foreign_fpr": refit_validation["overseas_other_clip_fpr"], "development_limit": .05,
               "passes_same_fpr_number": refit_validation["overseas_other_clip_fpr"] <= .05,
               "note": "Selected CV checkpoints passed all four guards. This is a separate fixed-refit validation result; do not claim the delivered model passes that guard."}
    write_json(REPORT / ("FINAL_VERIFICATION.json" if final else "MODEL_VERIFICATION.json"), result)
    if not result["model_development_and_training_complete"]:
        raise ValueError(f"Requirement audit incomplete: {[key for key, passed in checks.items() if not passed]}")
    print("EXPANDED_MODEL_AUDIT_PASS", len(checks), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    audit(parser.parse_args().final)
