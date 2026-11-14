"""Synthetic GPU checks for distillation math and integer export."""
from __future__ import annotations

import numpy as np

from dscnn_baseline.common import runtime, new_directory, write_json
from dscnn_baseline.weak_labels import bag_arrays
from dscnn_baseline.export import export_int8, interpreter_for, predict_int8
from .model import build_model, make_update
from .protocol import REPORT, OPTIONS, CANDIDATES, code_identity


def run():
    output = new_directory(REPORT / "student_preflight")
    tf, keras = runtime("gpu")
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    with tf.device("/GPU:0"):
        inputs = keras.Input((98, 40, 1))
        pooled = keras.layers.GlobalAveragePooling2D()(inputs)
        logits = keras.layers.Dense(3, kernel_initializer="zeros", bias_initializer="zeros", name="class_logits")(pooled)
        net = keras.Model(inputs, keras.layers.Softmax()(logits))
        optimizer = keras.optimizers.Adam(.001)
    arrays = [rng.normal(size=(1, 98, 40, 1)).astype(np.float32), rng.normal(size=(3, 98, 40, 1)).astype(np.float32)]
    x, weights = bag_arrays(arrays)
    teacher = np.array([[4, 0, 0], [0, 3, 0], [1, 2, 0], [0, 0, 2]], np.float32)
    scaled = teacher / OPTIONS["temperature"]
    q = np.exp(scaled - scaled.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    per_window = np.sum(q * (np.log(q) - np.log(1 / 3)), axis=1)
    expected_soft = float(np.mean(weights @ per_window) * OPTIONS["temperature"] ** 2)
    actual = make_update(tf, keras, net, optimizer, OPTIONS, False)([(x, weights, np.array([0, 1], np.int64), teacher)]).numpy()
    alpha = OPTIONS["hard_label_weight"]
    np.testing.assert_allclose(actual, [alpha * np.log(3) + (1 - alpha) * expected_soft, np.log(3), expected_soft], atol=1e-6)
    records = []
    for name, method in CANDIDATES.items():
        keras.backend.clear_session(); keras.utils.set_random_seed(42)
        with tf.device("/GPU:0"):
            net = build_model(keras, method, 42)
            optimizer = keras.optimizers.Adam(.001)
        update = make_update(tf, keras, net, optimizer, OPTIONS, method["specaugment"])
        for step in range(2):
            loss = update([(x, weights, np.array([0, 1], np.int64), teacher)]).numpy()
            if not np.isfinite(loss).all():
                raise ValueError("Synthetic distillation loss is not finite")
        devices = sorted({v.device for v in net.variables})
        if not all("GPU:0" in device for device in devices) or int(optimizer.iterations.numpy()) != 2:
            raise ValueError("Student preflight escaped GPU or skipped optimizer")
        path = output / f"{name}.keras"; net.save(str(path))
        restored = keras.models.load_model(str(path), compile=False)
        np.testing.assert_allclose(net(x, training=False).numpy(), restored(x, training=False).numpy(), atol=1e-6)
        records.append({"candidate": name, "losses": loss.tolist(), "optimizer_steps": 2, "variable_devices": devices})
    calibration = rng.uniform(-12, 8, (16, 98, 40, 1)).astype(np.float32)
    model_path, info = export_int8(tf, net, calibration, output / "export", [{"split": "synthetic"} for _ in calibration], pipeline_only=True)
    integer = predict_int8(interpreter_for(tf, model_path), calibration)
    if not np.isfinite(integer).all():
        raise ValueError("Synthetic int8 inference invalid")
    write_json(output / "status.json", {"passed": True, "exit_code": 0, "pipeline_only": True,
               "real_audio_files": 0, "file_balanced_KL_formula_verified": True, "code_identity": code_identity(),
               "candidates": records, "largest_candidate_int8_contract": info})
    print("DISTILLATION_STUDENT_GPU_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    run()
