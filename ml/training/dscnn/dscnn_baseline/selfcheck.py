"""Synthetic CPU-only training, reload and conversion; never dataset accuracy."""
from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .common import ROOT, config, new_directory, runtime, source, write_json
from .export import export_int8, interpreter_for, predict_int8
from .frontend import LogMel, synthetic_pcm
from .model import build_model, complexity


def run(output):
    output = new_directory(output)
    started = time.monotonic()
    status = {"kind": "synthetic_software_check", "pipeline_only": True, "deployment_ready": False,
              "real_audio_used": 0, "gpu_execution": False, "fixed_test_evaluated": False,
              "synthetic_probability_max_abs_tolerance": 0.05,
              "reload_rtol": 1e-5, "reload_atol": 1e-6,
              "command": [sys.executable, *sys.argv], "exit_code": None}
    write_json(output / "status.json", status)
    try:
        tf, keras = runtime("cpu")
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("Synthetic software check must expose no GPU")
        cfg, _ = config()
        keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        model = build_model(keras)
        optimizer = keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"])
        frontend = LogMel()
        features = np.stack([frontend(pcm) for pcm in synthetic_pcm(24)])
        # Numeric IDs exercise all three output slots, not acoustic class labels.
        labels = np.arange(len(features), dtype=np.int32) % 3
        before = [v.numpy().copy() for v in model.trainable_variables]
        losses = []
        for _ in range(2):
            with tf.GradientTape() as tape:
                probabilities = model(features[:6], training=True)
                loss = keras.losses.sparse_categorical_crossentropy(labels[:6], probabilities)
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
                raise RuntimeError("Missing/non-finite synthetic gradients")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(float(loss.numpy()))
        if not any(np.any(old != v.numpy()) for old, v in zip(before, model.trainable_variables)):
            raise RuntimeError("Optimizer did not change trainable parameters")
        float_prediction = model(features, training=False).numpy()
        model_path = output / "pipeline_only.keras"
        model.save(str(model_path))
        restored = keras.models.load_model(str(model_path), compile=False)
        reloaded = restored(features, training=False).numpy()
        np.testing.assert_allclose(reloaded, float_prediction, rtol=1e-5, atol=1e-6)
        path, metadata = export_int8(tf, restored, features, output / "export",
                                    [{"split": "synthetic", "index": i} for i in range(len(features))], pipeline_only=True)
        integer_prediction = predict_int8(interpreter_for(tf, path), features)
        error = float(np.max(np.abs(integer_prediction - float_prediction)))
        if not np.isfinite(integer_prediction).all() or error > status["synthetic_probability_max_abs_tolerance"]:
            raise RuntimeError(f"Synthetic float/int8 probability mismatch: max abs={error}")
        np.testing.assert_allclose(integer_prediction.sum(axis=1), 1, rtol=0, atol=3 / 256)
        np.savez(output / "synthetic_outputs.npz", float32=float_prediction, reloaded=reloaded, int8_dequantized=integer_prediction)
        with (output / "model_summary.txt").open("w", encoding="utf-8") as handle:
            model.summary(print_fn=lambda line: handle.write(line + "\n"))
        packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        (output / "packages.txt").write_text(packages.stdout, encoding="utf-8")
        status.update(passed=True, exit_code=0, tensorflow=tf.__version__, keras=keras.__version__,
                      forward_device=probabilities.device, gradient_devices=sorted({g.device for g in gradients}),
                      optimizer_updates=int(optimizer.iterations.numpy()), losses=losses,
                      reload_max_abs=float(np.max(np.abs(reloaded - float_prediction))),
                      float_int8_probability_max_abs=error,
                      float_int8_argmax_agreement=float(np.mean(float_prediction.argmax(1) == integer_prediction.argmax(1))),
                      synthetic_argmax_changed_indices=np.flatnonzero(float_prediction.argmax(1) != integer_prediction.argmax(1)).tolist(),
                      synthetic_float_top_two_margins=(np.sort(float_prediction, axis=1)[:, -1] - np.sort(float_prediction, axis=1)[:, -2]).tolist(),
                      code_sha256={p.name: source.sha256(p) for p in Path(__file__).parent.glob("*.py")},
                      complexity=complexity(model, keras), tflite=metadata,
                      keras_sha256=source.sha256(model_path))
    except Exception as error:
        status.update(passed=False, exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)
    print(f"SYNTHETIC_SOFTWARE_CHECK_PASS: {output}", flush=True)
    return status
