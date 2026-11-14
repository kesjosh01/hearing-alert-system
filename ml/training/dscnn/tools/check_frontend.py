"""Compile the portable frontend on the host and compare against NumPy."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import REPO, new_directory, runtime, source, write_json
from dscnn_baseline.frontend import LogMel, quantize, synthetic_pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, help="Also compare C++ vs Python int8 input through host inference")
    args = parser.parse_args()
    output = new_directory(args.output)
    started = time.monotonic()
    metadata = source.read_json(args.model_metadata)
    frontend = LogMel()
    if metadata["frontend_sha256"] != source.fingerprint(frontend.options):
        raise ValueError("Model/frontend contract mismatch")
    scale, zero = metadata["input"]["scale"], metadata["input"]["zero_point"]
    status = {"float_atol": 0.001, "float_rtol": 0.0, "int8_max_lsb": 1,
              "host_only": True, "board_execution": False, "exit_code": None}
    write_json(output / "status.json", status)
    directory = REPO / "firmware/transmitter_s3/dscnn"
    command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
               "-I", str(directory / "include"), str(directory / "log_mel_frontend.cc"),
               str(ROOT / "tools/frontend_host.cc"), "-o", str(output / "frontend_host")]
    records = []
    try:
        compiled = subprocess.run(command, check=True, capture_output=True, text=True)
        (output / "compile.log").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
        signals = list(synthetic_pcm(24))
        signals.extend([np.full(16000, -32768, dtype=np.int16), np.full(16000, 32767, dtype=np.int16)])
        impulse = np.zeros(16000, dtype=np.int16)
        impulse[1234] = 32767
        signals.append(impulse)
        interpreter = None
        if args.tflite:
            if source.sha256(args.tflite) != metadata["model_sha256"]:
                raise ValueError("Model hash mismatch")
            tf, _ = runtime("cpu")
            from dscnn_baseline.export import interpreter_for
            interpreter = interpreter_for(tf, args.tflite)
        for index, signal in enumerate(signals):
            pcm_path = output / f"signal_{index}.pcm"
            float_path = output / f"signal_{index}.f32"
            int_path = output / f"signal_{index}.i8"
            signal.astype("<i2").tofile(pcm_path)
            subprocess.run([str(output / "frontend_host"), str(pcm_path), str(float_path)], check=True)
            subprocess.run([str(output / "frontend_host"), str(pcm_path), str(int_path), str(scale), str(zero)], check=True)
            reference = frontend(signal)
            actual = np.fromfile(float_path, dtype="<f4").reshape(reference.shape)
            integer = np.fromfile(int_path, dtype=np.int8).reshape(reference.shape)
            float_error = float(np.max(np.abs(reference - actual)))
            quant_error = int(np.max(np.abs(quantize(reference, scale, zero).astype(np.int16) - integer.astype(np.int16))))
            record = {"signal": index, "float_max_abs": float_error, "int8_max_lsb": quant_error}
            records.append(record)
            np.testing.assert_allclose(actual, reference, atol=status["float_atol"], rtol=0)
            if quant_error > status["int8_max_lsb"]:
                raise AssertionError(f"C++ quantized frontend differs by {quant_error} LSB")
            if interpreter is not None:
                input_index = interpreter.get_input_details()[0]["index"]
                output_index = interpreter.get_output_details()[0]["index"]
                interpreter.set_tensor(input_index, quantize(reference[None], scale, zero))
                interpreter.invoke()
                expected_output = interpreter.get_tensor(output_index).copy()
                interpreter.set_tensor(input_index, integer[None])
                interpreter.invoke()
                actual_output = interpreter.get_tensor(output_index).copy()
                np.testing.assert_array_equal(actual_output, expected_output)
                record["int8_model_output_identical"] = True
        status.update(passed=True, exit_code=0)
    except Exception as error:
        status.update(passed=False, exit_code=1, first_error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(signals=records, elapsed_seconds=time.monotonic() - started,
                      compile_command=command, quantization={"scale": scale, "zero_point": zero},
                      frontend_workspace_bytes=5124)
        write_json(output / "status.json", status)
    print(json.dumps({"passed": status["passed"], "signals": len(records),
                      "float_max_abs": max(r["float_max_abs"] for r in records),
                      "int8_max_lsb": max(r["int8_max_lsb"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
