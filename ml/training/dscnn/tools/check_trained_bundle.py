"""Exercise the shipped offline CLI on all five new development doorbells."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import YAMNET, config, new_directory, source, write_json
from dscnn_baseline.data import FeatureStore


def check(bundle, output):
    bundle = Path(bundle).resolve()
    output = new_directory(output)
    status = {"kind": "trained_bundle_offline_cli_check", "exit_code": None,
              "passed": False, "host_only": True, "new_independent_test": False,
              "note": "These five files were used in development/refit. This checks the saved-audio CLI, not generalization.",
              "records": [], "commands": []}
    write_json(output / "status.json", status)
    try:
        manifest = source.read_json(bundle / "release_manifest.json")
        if manifest["pipeline_only"] is not False:
            raise ValueError("A trained bundle is required")
        for name, expected in manifest["files"].items():
            if source.sha256(bundle / name) != expected:
                raise ValueError("Bundle hash mismatch")
        extension = source.read_json(ROOT / config()[0]["development_extension_path"])
        assignments = sorted(extension["assignments"], key=lambda a: a["row"]["sample_id"])
        expected_ids = {f"kr-user-db-{i:02}" for i in range(6, 11)}
        if {a["row"]["sample_id"] for a in assignments} != expected_ids:
            raise ValueError("The declared five-file CLI check inventory changed")
        store = FeatureStore()
        for assignment in assignments:
            row = dict(assignment["row"], split="train")
            features, metadata = store.load(row)
            audio = YAMNET / row["relative_path"]
            predictions = {}
            for backend in ("float", "int8"):
                name = f"{row['sample_id']}_{backend}"
                result_path = output / f"{name}.json"
                command = [sys.executable, "-B", str(ROOT / "tools/predict_audio.py"),
                           "--bundle", str(bundle), "--audio", str(audio),
                           "--backend", backend, "--output", str(result_path)]
                status["commands"].append(command)
                environment = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
                with (output / f"{name}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                prediction = source.read_json(result_path)
                if prediction["audio_sha256"] != metadata["source_sha256"] or prediction["window_count"] != len(features):
                    raise ValueError("Offline CLI source/window identity differs from training frontend")
                predictions[backend] = prediction
            floating, integer = predictions["float"], predictions["int8"]
            labels = config()[0]["labels"]
            status["records"].append({"sample_id": row["sample_id"], "window_count": len(features),
                                      "float_label": floating["predicted_label"],
                                      "int8_label": integer["predicted_label"],
                                      "declared_decision_agrees": floating["predicted_label"] == integer["predicted_label"],
                                      "probability_max_abs": max(abs(floating["probabilities"][label] - integer["probabilities"][label]) for label in labels)})
            write_json(output / "status.json", status)
        status.update(passed=True, exit_code=0, checked_files=len(assignments), cli_executions=2 * len(assignments),
                      bundle_files_unchanged=all(source.sha256(bundle / name) == digest for name, digest in manifest["files"].items()))
        if not status["bundle_files_unchanged"]:
            raise ValueError("Bundle changed during CLI check")
    except Exception as error:
        status.update(passed=False, exit_code=1, first_error=f"{type(error).__name__}: {error}")
        (output / "first_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_json(output / "status.json", status)
    print("TRAINED_BUNDLE_OFFLINE_CLI_CHECK_PASS", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.bundle, args.output)
