"""Full-integer LiteRT export with explicit train-only calibration provenance."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import config, source, write_json
from .frontend import LogMel, quantize

TFLM_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX"}


def export_int8(tf, model, calibration, directory, provenance, pipeline_only=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    calibration = np.asarray(calibration, dtype=np.float32)
    if calibration.ndim != 4 or calibration.shape[1:] != LogMel().shape or not len(calibration) or not np.isfinite(calibration).all():
        raise ValueError("Representative calibration must contain finite [N,98,40,1] log-Mel windows")
    if not provenance or len(provenance) != len(calibration):
        raise ValueError("One provenance record per calibration window is required")
    expected_role = "synthetic" if pipeline_only else "train"
    if any(r.get("split") != expected_role for r in provenance):
        raise ValueError("Calibration may use training windows only; validation/test is forbidden")
    np.save(directory / "calibration_logmel.npy", calibration, allow_pickle=False)
    write_json(directory / "calibration_provenance.json", provenance)

    class Serving(tf.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        @tf.function(input_signature=[tf.TensorSpec([1, *LogMel().shape], tf.float32, name="log_mel")])
        def serve(self, log_mel):
            return {"probabilities": self.net(log_mel, training=False)}

    serving = Serving(model)
    saved_model = directory / "saved_model"
    tf.saved_model.save(serving, str(saved_model), signatures={"serving_default": serving.serve})
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([item[None]] for item in calibration)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converted = converter.convert()
    name = "pipeline_only_int8" if pipeline_only else "dscnn_int8"
    path = directory / f"{name}.tflite"
    path.write_bytes(converted)
    interpreter = interpreter_for(tf, path)
    info = inspect(interpreter)
    info.update(model_sha256=source.sha256(path), model_bytes=len(converted),
                labels=config()[0]["labels"], frontend=config()[0]["frontend"],
                frontend_sha256=source.fingerprint(config()[0]["frontend"]),
                pipeline_only=pipeline_only, deployment_ready=False,
                calibration_windows=len(calibration), calibration_role=expected_role,
                calibration_sha256=source.sha256(directory / "calibration_logmel.npy"))
    write_json(directory / "model_metadata.json", info)
    header = ["// Generated model bytes. See model_metadata.json before use.",
              f"// pipeline_only={str(pipeline_only).lower()}, deployment_ready=false",
              "#pragma once", "#include <cstddef>", "#include <cstdint>",
              "namespace hearing {", f"static constexpr bool kModelPipelineOnly = {str(pipeline_only).lower()};",
              f'static constexpr char kModelFrontendSha256[] = "{info["frontend_sha256"]}";',
              "alignas(16) static const uint8_t kModelData[] = {"]
    header += ["  " + ", ".join(f"0x{v:02x}" for v in converted[i:i + 16]) + "," for i in range(0, len(converted), 16)]
    header += ["};", "static constexpr size_t kModelBytes = sizeof(kModelData);", "}", ""]
    (directory / f"{name}.h").write_text("\n".join(header), encoding="utf-8")
    return path, info


def interpreter_for(tf, path):
    # Reference host kernels validate the flatbuffer without delegating to GPU.
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1,
                                     experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    return interpreter


def inspect(interpreter):
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one input and one output")
    if inputs[0]["dtype"] != np.int8 or outputs[0]["dtype"] != np.int8 or list(inputs[0]["shape"]) != [1, 98, 40, 1] or list(outputs[0]["shape"]) != [1, 3]:
        raise ValueError("Unexpected int8 input/output contract")
    operations = [item["op_name"] for item in interpreter._get_ops_details()]
    if set(operations) - TFLM_OPS:
        raise ValueError(f"Unexpected microcontroller operator(s): {set(operations) - TFLM_OPS}")
    tensors = interpreter.get_tensor_details()
    if any(t["dtype"] not in (np.int8, np.int32) for t in tensors):
        raise ValueError("Non-integer tensor found in full-int8 model")

    def detail(tensor):
        scale, zero = tensor["quantization"]
        if scale <= 0:
            raise ValueError("Missing quantization parameters")
        return {"shape": tensor["shape"].tolist(), "dtype": "int8", "scale": float(scale), "zero_point": int(zero)}

    return {"input": detail(inputs[0]), "output": detail(outputs[0]), "operators": operations,
            "operator_set": sorted(set(operations)), "all_tensor_types": sorted({np.dtype(t["dtype"]).name for t in tensors}),
            "board_tflm_allocate_tensors_verified": False}


def predict_int8(interpreter, features):
    inputs, outputs = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero = inputs["quantization"]
    oscale, ozero = outputs["quantization"]
    predictions = []
    for feature in features:
        interpreter.set_tensor(inputs["index"], quantize(feature[None], scale, zero))
        interpreter.invoke()
        predictions.append((interpreter.get_tensor(outputs["index"])[0].astype(np.float32) - ozero) * oscale)
    return np.asarray(predictions)
