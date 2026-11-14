"""Check the real file-label bag loss/gradient accumulation with synthetic bags."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import new_directory, runtime, source, write_json
from dscnn_baseline.frontend import LogMel, synthetic_pcm
from dscnn_baseline.model import build_model
from dscnn_baseline.weak_labels import bag_arrays, file_label_update


def main():
    output = new_directory(ROOT / "reports/real_training_20261114/weak_label_gpu_preflight")
    status = {"kind": "synthetic_file_label_bag_gpu_preflight", "real_audio_used": False,
              "passed": False, "window_labels_assigned": False, "exit_code": None,
              "weak_labels_code_sha256": source.sha256(ROOT / "dscnn_baseline/weak_labels.py")}
    write_json(output / "status.json", status)
    try:
        tf, keras = runtime("gpu")
        tf.config.experimental.enable_op_determinism()
        frontend = LogMel()
        values = [frontend(pcm) for pcm in synthetic_pcm()]
        bags = [np.stack([values[(i+j) % len(values)] for j in range(n)])
                for i, n in enumerate((1, 9, 3, 2, 11, 5))]
        microbatches = []
        for begin in (0, 3):
            x, weights = bag_arrays(bags[begin:begin+3])
            microbatches.append((x, weights, np.array([0, 1, 2], np.int64), ["synthetic"] * 3))
        with tf.device("/GPU:0"):
            model = build_model(keras, 42)
            optimizer = keras.optimizers.Adam(.001)
        before = model.get_weights()
        update = file_label_update(tf, keras, model, optimizer, "/GPU:0")
        losses = [float(update(microbatches).numpy()) for _ in range(3)]
        if not np.isfinite(losses).all() or int(optimizer.iterations.numpy()) != 3:
            raise ValueError("File-label GPU optimizer check failed")
        if not any(not np.array_equal(a,b) for a,b in zip(before,model.get_weights())):
            raise ValueError("File-label optimizer did not change weights")
        model.save(str(output / "synthetic_bag_only.keras"))
        restored = keras.models.load_model(str(output / "synthetic_bag_only.keras"),compile=False)
        x = np.stack(values)
        np.testing.assert_allclose(model(x,training=False).numpy(),restored(x,training=False).numpy(),rtol=1e-5,atol=1e-6)
        status.update(passed=True,exit_code=0,losses=losses,optimizer_steps=3,
                      microbatches_per_update=2,variable_devices=sorted({v.device for v in model.variables}),
                      save_reload_verified=True)
        print("FILE_LABEL_BAG_GPU_PREFLIGHT_PASS",flush=True)
    except Exception as error:
        status.update(exit_code=1,first_error=f"{type(error).__name__}: {error}")
        (output/"first_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        write_json(output/"status.json",status)


if __name__ == "__main__":
    main()
