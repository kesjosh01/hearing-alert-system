"""Reference OOF scores of frozen fold teachers on aligned development windows."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import runtime, source, write_json
from dscnn_baseline.data import development_splits
from improvement.protocol import REPORT, TEACHER_CACHE, verify_teacher
from improvement import policy


def run():
    path = REPORT / "teacher_oof_benchmark.json"
    if path.exists():
        raise FileExistsError(path)
    tf, keras = runtime("cpu")
    all_predictions = []
    for fold, rows in enumerate(development_splits()):
        receipt = verify_teacher(fold)
        head = keras.models.load_model(str(Path(receipt["directory"]) / "classifier.keras"), compile=False)
        predictions = []
        for row in rows:
            if row["split"] != "validation":
                continue
            feature_path = TEACHER_CACHE / f"{row['sample_id']}.npy"
            metadata = source.read_json(feature_path.with_suffix(".json"))
            if source.sha256(feature_path) != metadata["embedding_sha256"]:
                raise ValueError("Teacher benchmark feature hash mismatch")
            embeddings = np.load(feature_path, allow_pickle=False)
            probabilities = source.softmax(head(embeddings, training=False).numpy()).mean(0)
            predictions.append(source.prediction_row(row, probabilities, len(embeddings)))
        source.write_csv(REPORT / f"teacher_fold{fold}_validation_clips.csv", predictions)
        all_predictions.append(predictions)
    thresholds, metrics, pooled, rank = policy.calibrate(all_predictions, source.settings()[0]["validation_guards"])
    write_json(path, {"kind": "aligned_teacher_development_OOF_reference", "host_only": True,
                     "fixed_test_accessed": False, "thresholds": thresholds,
                     "folds": metrics, "pooled_domestic": pooled,
                     "mean_domestic_target_recall": float(np.mean([m["domestic_target_macro_recall"] for m in metrics])),
                     "note": "Fold teachers exclude their student-validation groups. Same aligned 0.975s view as distillation. A reference score, not a guaranteed student upper bound."})
    print("FROZEN_TEACHER_OOF_REFERENCE_COMPLETE")


if __name__ == "__main__":
    run()
