"""One post-freeze retrospective comparison under the approved FPR scope."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, new_directory, runtime, source, write_json
from dscnn_baseline.evaluation import HistoricalHoldoutStore, compare
from dscnn_baseline.export import interpreter_for, predict_int8
from dscnn_baseline.training import predict_clips
from expansion.protocol import REPORT
from expansion.runtime import frozen_inputs
from improvement import policy


def run():
    frozen_inputs()
    status = source.read_json(REPORT / "status.json")
    frozen = source.read_json(REPORT / "frozen_model_coverage.json")
    if not source.read_json(REPORT / "coverage_refit_status.json")["all100_training_targets_sampled"]:
        raise ValueError("Final target coverage correction incomplete")
    if status["stage"] != "training_complete" or status["exit_code"] != 0:
        raise ValueError("Training and development selection must be complete")
    refit = Path(frozen["refit"])
    model_path, int8_path = refit / "classifier.keras", refit / "export/dscnn_int8.tflite"
    if source.sha256(model_path) != frozen["model_sha256"] or source.sha256(int8_path) != frozen["int8_sha256"]:
        raise ValueError("Model hashes differ from pre-comparison freeze")
    output = new_directory(REPORT / "historical_holdout")
    receipt = {"kind": "expanded_model_historical_comparison", "exit_code": None,
               "test_evaluation_invocations": 1, "is_fresh_independent_test": False,
               "model_sha256_before": frozen["model_sha256"], "int8_sha256_before": frozen["int8_sha256"],
               "model_selection_and_thresholds_fixed_before_comparison": True, "deployment_ready": False}
    write_json(output / "status.json", receipt)
    started = time.monotonic()
    try:
        folds, _ = source.load_splits(config()[1])
        all_rows = [r for r in folds[0] if r["split"] == "test"]
        meta = {r["sample_id"]: r for r in source.read_csv(ROOT.parent / "yamnet/data/metadata/esc50_manifest.csv")}
        excluded = [r for r in all_rows if r["source_dataset"] == "ESC-50" and meta[r["sample_id"]]["confuser_tags"] == "siren"]
        excluded_ids = {r["sample_id"] for r in excluded}
        rows = [r for r in all_rows if r["sample_id"] not in excluded_ids]
        if len(all_rows) != 411 or len(excluded_ids) != 8 or len(rows) != 403:
            raise ValueError("Historical membership/exclusion policy changed")
        write_json(output / "evaluation_scope.json", {"original_frozen_test_count": len(all_rows), "evaluated_count": len(rows),
                   "foreign_negative_count": 394, "excluded_original_category": "siren", "excluded_ids": sorted(excluded_ids),
                   "excluded_audio_decoded": False, "eligibility_depends_on_model_predictions": False,
                   "raw_test_labels_and_membership_unchanged": True, "foreign_siren_subtype_not_inferred": True})
        tf, keras = runtime("gpu")
        tf.config.experimental.enable_op_determinism()
        store, features, provenance = HistoricalHoldoutStore(rows, frozen["model_sha256"]), {}, []
        for index, row in enumerate(rows, 1):
            values, metadata = store.load(row)
            features[row["sample_id"]] = values
            provenance.append({"sample_id": row["sample_id"], "source_sha256": metadata["source_sha256"],
                               "feature_sha256": metadata["feature_sha256"], "window_count": len(values)})
            if index % 100 == 0:
                print("EXPANDED_HISTORICAL_FEATURES", index, len(rows), flush=True)
        write_json(output / "feature_provenance.json", provenance)
        thresholds = frozen["decision"]["decision_policy"]["thresholds"]
        with tf.device("/GPU:0"):
            model = keras.models.load_model(str(model_path), compile=False)
            floating = policy.apply(predict_clips(model, rows, features), thresholds)
        interpreter = interpreter_for(tf, int8_path)
        quantized = []
        for index, row in enumerate(rows, 1):
            values = features[row["sample_id"]]
            probabilities = predict_int8(interpreter, values).mean(0)
            quantized.append(source.prediction_row(row, probabilities, len(values)))
            if index % 100 == 0:
                print("EXPANDED_HISTORICAL_INT8", index, len(rows), flush=True)
        quantized = policy.apply(quantized, thresholds)
        for name, predictions in (("float", floating), ("int8", quantized)):
            source.write_csv(output / f"{name}_clips.csv", predictions)
            source.write_csv(output / f"{name}_groups.csv", policy.apply(source.aggregate_groups(predictions), thresholds))
            metrics = policy.report(predictions, thresholds)
            if metrics["overseas_other_clip_count"] != 394:
                raise ValueError("False-positive denominator includes excluded target-like sounds")
            write_json(output / f"{name}_metrics.json", metrics)
        write_json(output / "float_int8_comparison.json", {"clips": compare(floating, quantized), "thresholds_recalibrated": False})
        previous = []
        directories = {"initial": ROOT / "reports/real_training_20261114/study_calibrated_v2_resumed/historical_holdout",
                       "previous_distilled": ROOT / "reports/improvement_20261114/historical_holdout"}
        for name, directory in directories.items():
            for backend in ("float", "int8"):
                path = directory / f"{backend}_clips.csv"
                old = [r for r in source.read_csv(path) if r["sample_id"] not in excluded_ids]
                if {r["sample_id"] for r in old} != {r["sample_id"] for r in rows}:
                    raise ValueError("Old/new retrospective populations differ")
                negatives = [r for r in old if r["target_domain"] != "ko_target" and r["label"] == "other"]
                metrics = source.read_json(directory / f"{backend}_metrics.json")
                fp = sum(r["predicted_label"] != "other" for r in negatives)
                previous.append({"model": name, "backend": backend, "domestic_target_macro_recall": metrics["domestic_target_macro_recall"],
                                 "foreign_negative_count": len(negatives), "false_positives": fp, "fpr": fp / len(negatives),
                                 "prediction_sha256": source.sha256(path), "new_model_inference": False})
        write_json(output / "previous_models_same_scope.json", previous)
        if source.sha256(model_path) != frozen["model_sha256"] or source.sha256(int8_path) != frozen["int8_sha256"]:
            raise ValueError("Frozen model changed during comparison")
        receipt.update(exit_code=0, evaluated_clips=len(rows), foreign_sirens_excluded=8,
                       model_sha256_after=source.sha256(model_path), int8_sha256_after=source.sha256(int8_path), model_unchanged=True)
        print("EXPANDED_RETROSPECTIVE_COMPARISON_COMPLETE", flush=True)
    except Exception as error:
        receipt.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", receipt)


if __name__ == "__main__":
    run()
