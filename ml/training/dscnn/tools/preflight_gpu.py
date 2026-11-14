"""Exercise the actual DS-CNN GPU operators on synthetic input only."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dscnn_baseline.common import new_directory, runtime, source, write_json
from dscnn_baseline.frontend import LogMel
from dscnn_baseline.model import build_model, complexity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = new_directory(args.output)
    started = time.monotonic()
    status = {"kind": "synthetic_dscnn_gpu_preflight", "real_audio_used": False,
              "real_dataset_training_executions": 0, "passed": False,
              "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "exit_code": None}
    write_json(output / "status.json", status)
    try:
        tf, keras = runtime("gpu")
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        frontend = LogMel()
        rng = np.random.default_rng(42)
        t = np.arange(16000) / 16000
        waves = [np.rint(6000 * np.sin(2 * np.pi * f * t)).astype(np.int16)
                 for f in (600, 1500)]
        waves.append(rng.integers(-1500, 1500, 16000, dtype=np.int16))
        x = np.stack([frontend(w) for w in waves])
        y = np.arange(3, dtype=np.int64)
        with tf.device("/CPU:0"):
            cpu = build_model(keras, 42)
            initial = cpu.get_weights()
            expected = cpu(x, training=False).numpy()
        with tf.device("/GPU:0"):
            model = build_model(keras, 42)
            model.set_weights(initial)
            optimizer = keras.optimizers.Adam(0.001)
            actual_tensor = model(x, training=False)
            if "GPU:0" not in actual_tensor.device:
                raise RuntimeError(f"Inference was not placed on the selected GPU: {actual_tensor.device}")
            actual = actual_tensor.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

        @tf.function
        def update(values, labels):
            with tf.device("/GPU:0"), tf.GradientTape() as tape:
                loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(
                    labels, model(values, training=True)))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        losses = [float(update(x, y).numpy()) for _ in range(3)]
        if not np.isfinite(losses).all():
            raise RuntimeError("Synthetic GPU loss is not finite")
        changed = any(not np.array_equal(a, b) for a, b in zip(initial, model.get_weights()))
        if not changed or int(optimizer.iterations.numpy()) != 3:
            raise RuntimeError("GPU optimizer did not update the model")
        with tf.device("/GPU:0"):
            before_save = model(x, training=False).numpy()
            model.save(str(output / "synthetic_only.keras"))
            restored = keras.models.load_model(str(output / "synthetic_only.keras"), compile=False)
            after_save = restored(x, training=False).numpy()
        np.testing.assert_allclose(after_save, before_save, rtol=1e-5, atol=1e-6)
        status.update(passed=True, exit_code=0, optimizer_steps=3, weights_changed=changed,
                      losses=losses, inference_device=actual_tensor.device,
                      variable_devices=sorted({v.device for v in model.variables}),
                      cpu_gpu_probability_max_abs=float(np.max(np.abs(actual - expected))),
                      save_reload_probability_max_abs=float(np.max(np.abs(after_save - before_save))),
                      tensorflow=tf.__version__, complexity=complexity(model, keras),
                      checkpoint_sha256=source.sha256(output / "synthetic_only.keras"))
        print("DSCNN_GPU_PREFLIGHT_PASS", flush=True)
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
