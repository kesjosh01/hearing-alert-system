"""Frozen YAMNet features aligned to student windows; development files only."""
from __future__ import annotations

import argparse
import hashlib
import time
import wave
from pathlib import Path

import numpy as np

from dscnn_baseline.common import ROOT, runtime, source, write_json
from dscnn_baseline.data import FeatureStore, development_splits, require_development
from dscnn_baseline.frontend import windows
from .protocol import BACKBONE, REPORT, TEACHER_CACHE, verify_teacher


def backbone_runtime():
    tf, keras = runtime("gpu")
    tf.config.experimental.enable_op_determinism()
    import tensorflow_hub as hub
    with tf.device("/GPU:0"):
        backbone = hub.load(str(BACKBONE))

    @tf.function(input_signature=[tf.TensorSpec([15600], tf.float32)], autograph=False)
    def embed(waveform):
        with tf.device("/GPU:0"):
            return backbone(waveform)[1][0]

    return tf, keras, backbone, embed


def preflight():
    output = REPORT / "teacher_preflight.json"
    if output.exists():
        raise FileExistsError(output)
    tf, keras, backbone, embed = backbone_runtime()
    start = time.monotonic()
    signal = np.sin(np.arange(15600, dtype=np.float32) * (2 * np.pi * 1000 / 16000)) * .1
    values = embed(signal).numpy()
    repeated = embed(signal).numpy()
    if values.shape != (1024,) or not np.isfinite(values).all():
        raise ValueError("YAMNet aligned embedding failed")
    np.testing.assert_allclose(repeated, values, atol=1e-6, rtol=1e-6)
    receipts = []
    for fold in range(4):
        receipt = verify_teacher(fold)
        with tf.device("/GPU:0"):
            model = keras.models.load_model(str(Path(receipt["directory"]) / "classifier.keras"), compile=False)
            model.trainable = False
            logits = model(values[None], training=False).numpy()
        if logits.shape != (1, 3) or not np.isfinite(logits).all():
            raise ValueError("Teacher logits have an invalid contract")
        receipts.append(receipt)
    write_json(output, {"passed": True, "exit_code": 0, "pipeline_only": True, "real_audio_files": 0,
                        "device": "gpu", "embedding_shape": list(values.shape), "teacher_output": "logits, not probabilities",
                        "backbone_pb_sha256": source.sha256(BACKBONE / "saved_model.pb"),
                        "teachers": receipts, "elapsed_seconds": time.monotonic() - start})
    print("ALIGNED_TEACHER_GPU_PREFLIGHT_PASS", flush=True)


def prepare():
    if source.read_json(REPORT / "teacher_preflight.json")["passed"] is not True:
        raise ValueError("Teacher preflight required")
    tf, _, backbone, embed = backbone_runtime()
    TEACHER_CACHE.mkdir(parents=True, exist_ok=True)
    store = FeatureStore()
    rows = development_splits()[0]
    backbone_hash = source.sha256(BACKBONE / "saved_model.pb")
    records = []
    started = time.monotonic()
    for i, row in enumerate(rows, 1):
        require_development(row)
        features, metadata = store.load(row)
        directory = store.cache / source.fingerprint([metadata["source_sha256"], store.signature])
        pcm_path = directory / "audio.wav"
        target = TEACHER_CACHE / f"{row['sample_id']}.npy"
        target_meta = target.with_suffix(".json")
        if target.exists() or target_meta.exists():
            receipt = source.read_json(target_meta)
            if receipt["source_sha256"] != metadata["source_sha256"] or receipt["backbone_pb_sha256"] != backbone_hash or receipt["embedding_sha256"] != source.sha256(target):
                raise ValueError("Incomplete/changed teacher cache; preserve before retry")
        else:
            with wave.open(str(pcm_path)) as handle:
                if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Unexpected decoded PCM contract")
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            if len(pcm) != metadata["samples"]:
                raise ValueError("Decoded cache duration differs from source features")
            vectors = np.stack([embed(chunk[:15600].astype(np.float32) / 32768).numpy() for _, chunk in windows(pcm)])
            if vectors.shape != (len(features), 1024) or not np.isfinite(vectors).all():
                raise ValueError("Student/teacher window alignment differs")
            np.save(target, vectors, allow_pickle=False)
            receipt = {"sample_id": row["sample_id"], "source_sha256": metadata["source_sha256"],
                       "decoded_pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                       "student_feature_sha256": metadata["feature_sha256"], "window_starts": metadata["starts"],
                       "embedding_sha256": source.sha256(target), "backbone_pb_sha256": backbone_hash,
                       "window_count": len(vectors), "event_annotation": False}
            write_json(target_meta, receipt)
        records.append(receipt)
        if i % 50 == 0 or i == len(rows):
            write_json(REPORT / "teacher_preparation.json", {"completed_files": i, "total_files": len(rows),
                        "completed_windows": sum(r["window_count"] for r in records), "fixed_test_files": 0,
                        "complete": i == len(rows), "elapsed_seconds": time.monotonic() - started})
            print(f"ALIGNED_TEACHER_FEATURES {i}/{len(rows)}", flush=True)
    write_json(REPORT / "teacher_feature_provenance.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "prepare"))
    arguments = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    preflight() if arguments.mode == "preflight" else prepare()
