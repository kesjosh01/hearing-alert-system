"""Synthetic GPU gate and aligned development features; no downloads/install."""
from __future__ import annotations

import argparse
import hashlib
import time
import wave

import numpy as np

from dscnn_baseline.common import ROOT, runtime, source, write_json
from dscnn_baseline.export import export_int8
from dscnn_baseline.frontend import windows
from dscnn_baseline.model import complexity
from .data import ExpandedFeatureStore, identity, read_rows
from .model import build_model, build_teacher_head, make_update, teacher_update
from .protocol import BACKBONE, CANDIDATES, EMBEDDINGS, OPTIONS, REPORT, code_identity


def frozen_inputs():
    plan = source.read_json(REPORT / "frozen_plan.json")
    if plan["code_identity"] != code_identity() or plan["dataset_identity"] != identity():
        raise ValueError("Code or expanded dataset changed after freezing")
    return plan


def backbone_embed(tf):
    import tensorflow_hub as hub
    with tf.device("/GPU:0"):
        backbone = hub.load(str(BACKBONE))

    @tf.function(input_signature=[tf.TensorSpec([15600], tf.float32)], autograph=False)
    def embed(pcm):
        with tf.device("/GPU:0"):
            return backbone(pcm)[1][0]
    return backbone, embed


def preflight():
    frozen_inputs()
    if (REPORT / "preflight.json").exists():
        raise FileExistsError("Preserve existing preflight receipt")
    tf, keras = runtime("gpu")
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    with tf.device("/GPU:0"):
        head = build_teacher_head(keras, 42)
        optimizer = keras.optimizers.Adam(.001)
    update = teacher_update(tf, keras, head, optimizer)
    weights = np.array([[.5, .5, 0, 0], [0, 0, .5, .5]], np.float32)
    y = np.array([0, 1], np.int64)
    value = update([(rng.normal(size=(4, 1024)).astype(np.float32), weights, y, ["a", "b"])])
    if not np.isfinite(value.numpy()).all() or int(optimizer.iterations.numpy()) != 1:
        raise ValueError("New teacher-head optimizer preflight failed")
    checked = []
    for name, method in CANDIDATES.items():
        keras.backend.clear_session()
        with tf.device("/GPU:0"):
            model = build_model(keras, method, 42)
            optimizer = keras.optimizers.Adam(.001)
        update = make_update(tf, keras, model, optimizer, OPTIONS, True)
        x = rng.normal(size=(4, 98, 40, 1)).astype(np.float32)
        targets = np.array([[5, 0, -4], [5, 0, -4], [0, 5, -4], [0, 5, -4]], np.float32)
        for _ in range(2):
            value = update([(x, weights, y, targets)])
            if not np.isfinite(value.numpy()).all():
                raise ValueError("Non-finite synthetic KD update")
        devices = sorted({v.device for v in model.variables})
        if any("GPU:0" not in v for v in devices) or int(optimizer.iterations.numpy()) != 2:
            raise ValueError("Candidate not executing on selected GPU")
        checked.append({"candidate": name, "optimizer_steps": 2, "devices": devices, "complexity": complexity(model, keras)})
        if name == "temporal24":
            _, exported = export_int8(tf, model, x, REPORT / "preflight_integer", [{"split": "synthetic", "sample_id": f"synthetic-{i}"} for i in range(4)], pipeline_only=True)
    _, embed = backbone_embed(tf)
    signal = .1 * np.sin(np.arange(15600, dtype=np.float32) * (2 * np.pi * 1000 / 16000))
    vector = embed(signal).numpy()
    if vector.shape != (1024,) or not np.isfinite(vector).all():
        raise ValueError("Local frozen YAMNet backbone unavailable")
    write_json(REPORT / "preflight.json", {"passed": True, "real_audio_files": 0, "pipeline_only": True,
               "teacher_head_update_verified": True, "candidates": checked, "full_integer_export": exported,
               "backbone_pb_sha256": source.sha256(BACKBONE / "saved_model.pb"), "apps_or_packages_downloaded": False})
    print("EXPANSION_SYNTHETIC_GPU_PREFLIGHT_PASS", flush=True)


def prepare():
    frozen_inputs()
    if not source.read_json(REPORT / "preflight.json")["passed"]:
        raise ValueError("Synthetic GPU preflight required")
    tf, _ = runtime("gpu")
    tf.config.experimental.enable_op_determinism()
    _, embed = backbone_embed(tf)
    backbone_hash = source.sha256(BACKBONE / "saved_model.pb")
    store, receipts = ExpandedFeatureStore(), []
    rows = read_rows(0)
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        values, metadata = store.load(row)
        target = EMBEDDINGS / f"{row['sample_id']}.npy"
        if target.exists():
            receipt = source.read_json(target.with_suffix(".json"))
            if receipt["source_sha256"] != metadata["source_sha256"] or receipt["backbone_pb_sha256"] != backbone_hash or receipt["embedding_sha256"] != source.sha256(target) or receipt["window_count"] != len(values):
                raise ValueError("Previously aligned embedding cache changed")
        else:
            directory = store.cache / source.fingerprint([metadata["source_sha256"], store.signature])
            with wave.open(str(directory / "audio.wav")) as handle:
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            vectors = np.stack([embed(chunk[:15600].astype(np.float32) / 32768).numpy() for _, chunk in windows(pcm)])
            if vectors.shape != (len(values), 1024) or not np.isfinite(vectors).all():
                raise ValueError("New aligned embedding shape/values failed")
            np.save(target, vectors, allow_pickle=False)
            receipt = {"sample_id": row["sample_id"], "source_sha256": metadata["source_sha256"],
                       "decoded_pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                       "student_feature_sha256": metadata["feature_sha256"], "window_starts": metadata["starts"],
                       "embedding_sha256": source.sha256(target), "backbone_pb_sha256": backbone_hash,
                       "window_count": len(vectors), "event_annotation": False}
            write_json(target.with_suffix(".json"), receipt)
        receipts.append(receipt)
        if index % 100 == 0 or index == len(rows):
            write_json(REPORT / "feature_preparation.json", {"complete": index == len(rows), "files": index,
                       "total_files": len(rows), "windows": sum(r["window_count"] for r in receipts),
                       "fixed_test_files": 0, "elapsed_seconds": time.monotonic() - started})
            print("EXPANSION_FEATURES", index, len(rows), flush=True)
    write_json(REPORT / "feature_provenance.json", receipts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "prepare"))
    preflight() if parser.parse_args().mode == "preflight" else prepare()
