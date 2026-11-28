"""Export the pinned domestic checkpoint for the S3 and measure conversion drift.

Use TensorFlow and tf-keras 2.21, LiteRT, scipy and numpy. The output is an experiment
artifact. The separated comparison recordings measure conversion, not recall.
"""
import argparse
import csv
import hashlib
import json
import sys
import tf_keras as keras
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample_poly
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "ml/training/dscnn"))
from dscnn_baseline.frontend import LogMel, quantize, windows

CHECKPOINT_SHA = "2f58eed947db6fb6d9e59c624f13635ec0a1346c421475500961f9e270c5ad2a"
TEMPERATURE = 4.0  # the policy simulated for this checkpoint in per_file_results.csv
CUTS = np.array([190, 162])
OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX"}


def checkpoint_model(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Unexpected domestic checkpoint SHA-256")
    return keras.models.load_model(path, compile=False)


def features(directory, names):
    frontend = LogMel()
    groups = []
    for name in names:
        rate, pcm = wavfile.read(directory / name)
        if rate != 44100 or pcm.ndim != 1 or pcm.dtype != np.int16:
            raise ValueError(f"Unexpected ESC-50 PCM format in {name}")
        signal = resample_poly(pcm.astype(np.float64), 160, 441)
        pcm16 = np.clip(np.rint(signal), -32768, 32767).astype(np.int16)
        groups.append(np.stack([frontend(chunk) for _, chunk in windows(pcm16)]))
    return groups


def decisions(scores):
    # Reset history between files, as the S3 does following discontinuity.
    winning = np.argmax(scores, axis=1)
    hit = (winning[:, None] == [0, 1]) & (scores[:, :2] >= CUTS)
    padded = np.pad(hit.astype(int), ((2, 0), (0, 0)))
    return padded[:-2] + padded[1:-1] + padded[2:] >= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="audio/ from the pinned ESC-50 revision")
    parser.add_argument("output", type=Path, help="new directory for model and comparison report")
    args = parser.parse_args()
    manifest = json.loads((HERE / "recordings.json").read_text())
    with (REPO / manifest["manifest"]).open(encoding="utf-8-sig", newline="") as f:
        training = {r["source_asset_id"]: r["split_group_id"] for r in csv.DictReader(f) if r["split"] == "train"}
    cal_names, check_names = manifest["calibration"], manifest["conversion_check"]
    if {training[n] for n in cal_names} & {training[n] for n in check_names}:
        raise ValueError("Calibration and comparison recordings share a group")
    args.output.mkdir(parents=True, exist_ok=False)
    cal = np.concatenate(features(args.audio, cal_names))
    checks = features(args.audio, check_names)
    model = checkpoint_model(HERE.parent / "classifier.keras")
    # Scale the final linear layer before softmax. This is stable even for
    # saturated T=1 probabilities; the original .keras file stays untouched.
    logits = keras.Model(model.inputs, model.get_layer("class_logits").output)
    reference = [tf.nn.softmax(logits(x, training=False) / TEMPERATURE).numpy() for x in checks]
    head = model.get_layer("class_logits")
    head.set_weights([w / TEMPERATURE for w in head.get_weights()])

    @tf.function(input_signature=[tf.TensorSpec([1, 98, 40, 1], tf.float32)])
    def predict(x):
        return model(x, training=False)

    frozen = convert_variables_to_constants_v2(predict.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = converter.inference_output_type = tf.int8
    converter.representative_dataset = lambda: ([x[None]] for x in cal)
    binary = converter.convert()
    interpreter = Interpreter(model_content=binary, num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    ops = {op["op_name"] for op in interpreter._get_ops_details()}
    if (ops != OPS or inp["shape"].tolist() != [1, 98, 40, 1] or out["shape"].tolist() != [1, 3]
            or inp["dtype"] != np.int8 or out["dtype"] != np.int8
            or out["quantization"] != (1 / 256, -128)):
        raise ValueError("Converted model does not satisfy the firmware contract")

    matches = policy_matches = total = 0
    errors = []
    for x, expected in zip(checks, reference):
        observed = []
        for window in x:
            interpreter.set_tensor(inp["index"], quantize(window[None], *inp["quantization"]))
            interpreter.invoke()
            observed.append(interpreter.get_tensor(out["index"])[0].astype(np.int16) + 128)
        observed = np.stack(observed)
        errors.extend(np.abs(expected - observed / 256).ravel().tolist())
        matches += np.count_nonzero(expected.argmax(axis=1) == observed.argmax(axis=1))
        ref_scores = np.clip(np.floor(expected * 256 + 0.5), 0, 255)
        policy_matches += np.count_nonzero(np.all(decisions(ref_scores) == decisions(observed), axis=1))
        total += len(x)
    report = dict(source_checkpoint_sha256=CHECKPOINT_SHA, tflite_sha256=hashlib.sha256(binary).hexdigest(),
                  tflite_bytes=len(binary), tensorflow=tf.__version__, keras=keras.__version__,
                  temperature=TEMPERATURE, integer_cuts=CUTS.tolist(), ops=sorted(ops),
                  input_quantization=list(inp["quantization"]), output_quantization=list(out["quantization"]),
                  calibration_recordings=len(cal_names), calibration_windows=len(cal),
                  conversion_check_recordings=len(check_names), conversion_check_windows=total,
                  same_argmax_windows=int(matches), same_2_of_3_decision_windows=int(policy_matches),
                  mean_absolute_probability_error=float(np.mean(errors)), max_absolute_probability_error=max(errors),
                  korean_recordings_tested=False, board_tested=False)
    (args.output / "model_int8.tflite").write_bytes(binary)
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
