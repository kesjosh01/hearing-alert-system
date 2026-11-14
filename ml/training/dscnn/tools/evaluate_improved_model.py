"""One retrospective comparison after improvement selection/refit are frozen."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import runtime, config, source, new_directory, write_json
from dscnn_baseline.data import development_identity
from dscnn_baseline.evaluation import HistoricalHoldoutStore, compare
from dscnn_baseline.training import predict_clips
from dscnn_baseline.export import interpreter_for, predict_int8
from improvement.protocol import REPORT, code_identity
from improvement import policy


def run():
    study = source.read_json(REPORT / "status.json")
    frozen = source.read_json(REPORT / "frozen_model.json")
    if study.get("stage") != "training_complete" or study.get("exit_code") != 0:
        raise ValueError("Improvement selection and fixed refit must finish before historical comparison")
    if frozen["code_identity"] != code_identity() or frozen["development_identity"] != development_identity():
        raise ValueError("Frozen improvement inputs changed")
    refit = Path(frozen["refit"])
    model_path, integer_path = refit / "classifier.keras", refit / "export/dscnn_int8.tflite"
    if source.sha256(model_path) != frozen["model_sha256"] or source.sha256(integer_path) != frozen["int8_sha256"]:
        raise ValueError("Frozen model hashes differ")
    output = new_directory(REPORT / "historical_holdout")
    status = {"kind": "improved_model_retrospective_holdout_comparison", "exit_code": None,
              "is_fresh_independent_test": False, "test_evaluation_invocations": 1,
              "previous_student_test_comparisons": 1, "not_used_for_improvement_selection": True,
              "model_sha256_before": frozen["model_sha256"], "int8_sha256_before": frozen["int8_sha256"],
              "deployment_ready": False, "completed_clips": 0}
    write_json(output / "status.json", status)
    started = time.monotonic()
    try:
        _, protocol = config()
        folds, _ = source.load_splits(protocol)
        rows = [r for r in folds[0] if r["split"] == "test"]
        tf, keras = runtime("gpu")
        tf.config.experimental.enable_op_determinism()
        store = HistoricalHoldoutStore(rows, frozen["model_sha256"])
        features, provenance = {}, []
        for i, row in enumerate(rows, 1):
            values, metadata = store.load(row)
            features[row["sample_id"]] = values
            provenance.append(dict(sample_id=row["sample_id"], source_sha256=metadata["source_sha256"],
                                   feature_sha256=metadata["feature_sha256"], window_count=len(values)))
            if i % 50 == 0:
                print("IMPROVED_HISTORICAL_FEATURES", i, len(rows), flush=True)
        write_json(output / "feature_provenance.json", provenance)
        with tf.device("/GPU:0"):
            model = keras.models.load_model(str(model_path), compile=False)
        thresholds = frozen["decision"]["decision_policy"]["thresholds"]
        floating = policy.apply(predict_clips(model, rows, features), thresholds)
        integer = interpreter_for(tf, integer_path)
        quantized = []
        for i, row in enumerate(rows, 1):
            values = features[row["sample_id"]]
            probabilities = predict_int8(integer, values).mean(0)
            quantized.append(source.prediction_row(row, probabilities, len(values)))
            if i % 50 == 0:
                status["completed_clips"] = i; write_json(output / "status.json", status)
                print("IMPROVED_HISTORICAL_INT8", i, len(rows), flush=True)
        quantized = policy.apply(quantized, thresholds)
        for name, predictions in (("float", floating), ("int8", quantized)):
            source.write_csv(output / f"{name}_clips.csv", predictions)
            source.write_csv(output / f"{name}_groups.csv", policy.apply(source.aggregate_groups(predictions), thresholds))
            write_json(output / f"{name}_metrics.json", policy.report(predictions, thresholds))
        write_json(output / "float_int8_comparison.json", {"clips": compare(floating, quantized),
                   "source_groups": compare(policy.apply(source.aggregate_groups(floating), thresholds),
                                            policy.apply(source.aggregate_groups(quantized), thresholds)),
                   "note": "Observed differences only; no test-driven recalibration or model reselection."})
        if source.sha256(model_path) != frozen["model_sha256"] or source.sha256(integer_path) != frozen["int8_sha256"]:
            raise ValueError("Model changed during historical comparison")
        status.update(exit_code=0, completed_clips=len(rows), evaluated_clips=len(rows),
                      model_sha256_after=source.sha256(model_path), int8_sha256_after=source.sha256(integer_path),
                      model_unchanged=True)
        print("IMPROVED_RETROSPECTIVE_COMPARISON_COMPLETE", flush=True)
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)


if __name__ == "__main__":
    run()
