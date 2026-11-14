"""One frozen-model comparison on the historically used holdout; never tune here."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from .common import ROOT, config, new_directory, runtime, source, write_json
from .data import FeatureStore, development_identity
from .export import interpreter_for, predict_int8
from .training import predict_clips
from . import decisions


class HistoricalHoldoutStore(FeatureStore):
    """Separate cache and exact membership gate, usable only by explicit evaluation."""
    def __init__(self, rows, model_hash):
        super().__init__(ROOT / ".cache/historical_holdout" / model_hash)
        self.allowed = {r["sample_id"]: r for r in rows}
        if not rows or any(r["split"] != "test" for r in rows):
            raise ValueError("Historical holdout inventory must contain test rows only")

    def load(self, row):
        if row["split"] != "test" or self.allowed.get(row["sample_id"]) != row:
            raise ValueError("Holdout feature extraction requires exact frozen test membership")
        return self._load_audio(row)


def compare(float_rows, integer_rows):
    if [r["sample_id"] for r in float_rows] != [r["sample_id"] for r in integer_rows]:
        raise ValueError("Float/int8 evaluation inventories differ")
    a = np.asarray([[float(r[f"p_{label}"]) for label in source.LABELS] for r in float_rows])
    b = np.asarray([[float(r[f"p_{label}"]) for label in source.LABELS] for r in integer_rows])
    return {"count": len(a), "probability_max_abs": float(np.max(np.abs(a - b))),
            "argmax_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
            "declared_decision_agreement": float(np.mean([a["predicted_label"]==b["predicted_label"] for a,b in zip(float_rows,integer_rows)])),
            "argmax_disagreements": int(np.sum(a.argmax(1) != b.argmax(1)))}


def evaluate(refit, frozen_path, output):
    refit = Path(refit).resolve()
    frozen = source.read_json(frozen_path)
    refit_status = source.read_json(refit / "status.json")
    if refit_status.get("kind") != "dscnn_refit" or refit_status.get("exit_code") != 0:
        raise ValueError("Evaluation requires completed fixed-epoch refit")
    model_path, int8_path = refit / "classifier.keras", refit / "export/dscnn_int8.tflite"
    if source.sha256(model_path) != frozen["model_sha256"] or source.sha256(int8_path) != frozen["int8_sha256"]:
        raise ValueError("Frozen model hash changed before evaluation")
    if frozen["development_identity"] != development_identity():
        raise ValueError("Development selection changed before evaluation")
    directory = new_directory(output)
    status = {"kind": "historical_reused_holdout_comparison", "is_fresh_independent_test": False,
              "test_evaluation_invocations": 1, "exit_code": None, "deployment_ready": False,
              "model_sha256_before": frozen["model_sha256"], "completed_clips": 0}
    write_json(directory / "status.json", status)
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
            array, meta = store.load(row)
            features[row["sample_id"]] = array
            provenance.append({"sample_id": row["sample_id"], "source_sha256": meta["source_sha256"],
                               "feature_sha256": meta["feature_sha256"], "window_count": len(array)})
            if i % 50 == 0 or i == len(rows):
                print(f"HISTORICAL_HOLDOUT_FEATURES {i}/{len(rows)}", flush=True)
        write_json(directory / "feature_provenance.json", provenance)
        with tf.device("/GPU:0"):
            model = keras.models.load_model(str(model_path), compile=False)
        floating = predict_clips(model, rows, features)
        threshold = frozen["decision_policy"]["threshold"]
        floating = decisions.apply(floating,threshold)
        integer = interpreter_for(tf, int8_path)
        quantized = []
        for i, row in enumerate(rows, 1):
            values = features[row["sample_id"]]
            probabilities = predict_int8(integer, values).mean(0)
            quantized.append(source.prediction_row(row, probabilities, len(values)))
            if i % 50 == 0 or i == len(rows):
                status["completed_clips"] = i
                write_json(directory / "status.json", status)
                print(f"HISTORICAL_HOLDOUT_INT8 {i}/{len(rows)}", flush=True)
        quantized = decisions.apply(quantized,threshold)
        for name, predictions in (("float", floating), ("int8", quantized)):
            source.write_csv(directory / f"{name}_clips.csv", predictions)
            source.write_csv(directory / f"{name}_groups.csv", decisions.aggregate(predictions,threshold))
            write_json(directory / f"{name}_metrics.json", decisions.report(predictions,threshold))
        parity = {"clips": compare(floating, quantized),
                  "source_groups": compare(decisions.aggregate(floating,threshold), decisions.aggregate(quantized,threshold)),
                  "note": "Observed quantization differences, not a post-hoc tuning or acceptance threshold."}
        write_json(directory / "float_int8_comparison.json", parity)
        model_after, int8_after = source.sha256(model_path), source.sha256(int8_path)
        if model_after != frozen["model_sha256"] or int8_after != frozen["int8_sha256"]:
            raise ValueError("Frozen model changed during evaluation")
        status.update(exit_code=0, evaluated_clips=len(rows), model_sha256_after=model_after,
                      int8_sha256_after=int8_after, model_unchanged=True)
        print("HISTORICAL_HOLDOUT_COMPARISON_COMPLETE", flush=True)
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (directory / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.refit, args.frozen, args.output)


if __name__ == "__main__":
    main()
