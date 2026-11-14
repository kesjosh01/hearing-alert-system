"""Verify GPU-origin synthetic checkpoint -> full-int8 host inference, no real audio."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import new_directory, runtime, write_json
from dscnn_baseline.export import export_int8, interpreter_for, predict_int8
from dscnn_baseline.frontend import LogMel, synthetic_pcm


def main():
    base = ROOT / "reports/real_training_20261114"
    output = new_directory(base / "gpu_export_preflight")
    status = {"kind": "synthetic_gpu_origin_int8_preflight", "real_audio_used": False,
              "passed": False, "deployment_ready": False, "exit_code": None}
    write_json(output / "status.json", status)
    try:
        tf, keras = runtime("gpu")
        with tf.device("/GPU:0"):
            model = keras.models.load_model(str(base / "gpu_preflight/synthetic_only.keras"), compile=False)
        frontend = LogMel()
        features = np.stack([frontend(pcm) for pcm in synthetic_pcm()])
        provenance = [{"split": "synthetic", "sample_id": f"synthetic-{i}"} for i in range(len(features))]
        path, metadata = export_int8(tf, model, features, output / "export", provenance, pipeline_only=True)
        floating = model(features, training=False).numpy()
        integer = predict_int8(interpreter_for(tf, path), features)
        error = float(np.max(np.abs(floating - integer)))
        if error > .05:
            raise ValueError(f"Synthetic quantization path tolerance exceeded: {error}")
        status.update(passed=True, exit_code=0, probability_max_abs=error,
                      argmax_agreement=float(np.mean(floating.argmax(1) == integer.argmax(1))),
                      int8_metadata=metadata)
        print("SYNTHETIC_GPU_INT8_PREFLIGHT_PASS", flush=True)
    except Exception as error:
        status.update(exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
