"""Fail-closed GPU-only companion to the unchanged sound_baseline CLI.

No installs, extraction, test evaluation, refit, resume, or automatic retry.
Use `validate` for the minimal GPU check; `run` also executes the full CV.
Source the dedicated activation script first.
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sound_baseline.core import (LABELS, GroupSampler, fingerprint, load_splits,
                                 read_csv, read_json, settings, sha256, write_csv, write_json)
from sound_baseline.features import FeatureStore, load_classifier, tensorflow_runtime
from sound_baseline import training

GPU_UUID = "GPU-10f7f4b0-eee7-9d43-26de-7bbefdbfc7ce"
NVIDIA = {
    "nvidia-cublas-cu12": "12.8.4.1", "nvidia-cuda-cupti-cu12": "12.8.90",
    "nvidia-cuda-runtime-cu12": "12.8.90", "nvidia-cuda-nvcc-cu12": "12.8.93",
    "nvidia-cuda-nvrtc-cu12": "12.8.93", "nvidia-cudnn-cu12": "9.7.1.26",
    "nvidia-cufft-cu12": "11.3.3.83", "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-cusolver-cu12": "11.7.3.90", "nvidia-cusparse-cu12": "12.5.8.93",
    "nvidia-nccl-cu12": "2.26.2", "nvidia-nvjitlink-cu12": "12.8.93",
}
CORE = {"tensorflow": "2.18.1", "tf-keras": "2.18.0", "tensorflow-hub": "0.16.1",
        "numpy": "1.26.4", "setuptools": "80.9.0"}
NAMES = {w: f"ubuntu_gpu_cv_w{w}_s42" for w in (1, 2, 4)}
PROFILES = {
    "pinned218": {"venv": ".venv-gpu", "changes": {}},
    "official221": {"venv": ".venv-gpu-official221", "changes": {
        "tensorflow": "2.21.0", "tf-keras": "2.21.0", "h5py": "3.14.0",
        "protobuf": "6.33.6", "nvidia-cudnn-cu12": "9.8.0.87", "nvidia-nccl-cu12": "2.27.7"}},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def command_text(command):
    return shlex.join([str(x) for x in command])


def training_args(weight, name, validation=False):
    return ["train", "--fold", "0" if validation else "all", "--seed", "42",
            "--domestic-weight", str(weight), "--run-name", name,
            "--epochs", "1" if validation else "30", "--patience", "8",
            "--batch-size", "30", "--steps-per-epoch", "2" if validation else "30",
            "--learning-rate", "0.001", "--wandb-mode", "disabled"]


def environment_packages(python):
    result = subprocess.run([str(python), "-m", "pip", "list", "--format=json"],
                            capture_output=True, text=True, check=True)
    return {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(result.stdout)}


def protected_roots():
    return [ROOT / "data", ROOT / "config", ROOT / "sound_baseline", ROOT / "tests",
            ROOT / "requirements-yamnet.txt", ROOT / ".venv/pyvenv.cfg", ROOT / ".venv-gpu/pyvenv.cfg",
            ROOT / ".venv-gpu-official221/pyvenv.cfg",
            *sorted((ROOT / "runs/yamnet").iterdir())]


def snapshot(roots):
    files = {}
    for root in roots:
        root = Path(root)
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(ROOT))] = sha256(path)
    return files


def cache_audit():
    baseline, protocol = settings()
    folds, _ = load_splits(protocol)
    store = FeatureStore(baseline)
    development = [r for r in folds[0] if r["split"] != "test"]
    test = [r for r in folds[0] if r["split"] == "test"]
    present = [r["sample_id"] for r in test if store.paths(r["sample_id"])["embedding"].exists()]
    if present:
        raise RuntimeError(f"Fixed-test embeddings must remain zero: {present}")
    patches = sum(len(store.load(r)) for r in development)
    return {"folds": len(folds), "records_per_fold": len(folds[0]),
            "development_embeddings": len(development), "development_patches": patches,
            "fixed_test_records": len(test), "fixed_test_embeddings": len(present),
            "feature_signature": store.signature, "test_evaluated": False}


def tensor_hashes(h5):
    import h5py
    output = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            values = obj[()]
            output[name.replace("\\", "/")] = {
                "dtype": str(values.dtype), "shape": list(values.shape),
                "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest()}
    h5.visititems(visit)
    return output


def normalize_checkpoint(destination, audit_path):
    """Normalize only HDF5/ZIP member paths; preserve every tensor byte."""
    import h5py
    source = ROOT / "runs/yamnet/first_fold0_w4/classifier.keras"
    before = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Normalized checkpoint already exists: {destination}")
    h5_bytes = io.BytesIO()
    with zipfile.ZipFile(source) as original:
        with h5py.File(io.BytesIO(original.read("model.weights.h5")), "r") as src, h5py.File(h5_bytes, "w") as dst:
            original_hashes = tensor_hashes(src)
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            def copy(name, obj):
                normalized = name.replace("\\", "/")
                if isinstance(obj, h5py.Group):
                    new = dst.require_group(normalized)
                else:
                    parent, _, leaf = normalized.rpartition("/")
                    new = dst.require_group(parent).create_dataset(leaf, data=obj[()], dtype=obj.dtype)
                for key, value in obj.attrs.items():
                    new.attrs[key] = value
            src.visititems(copy)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as normalized_zip:
            for member in original.infolist():
                content = h5_bytes.getvalue() if member.filename == "model.weights.h5" else original.read(member.filename)
                normalized_zip.writestr(member.filename.replace("\\", "/"), content)
    with zipfile.ZipFile(destination) as z, h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as h5:
        normalized_hashes = tensor_hashes(h5)
    if original_hashes != normalized_hashes or before != sha256(source):
        raise RuntimeError("Checkpoint source/tensor hashes changed")
    write_json(audit_path, {"source": str(source), "source_sha256_before": before,
               "source_sha256_after": sha256(source), "normalized": str(destination),
               "normalized_sha256": sha256(destination), "tensor_hashes_match": True,
               "tensors": normalized_hashes, "change": "backslash HDF5 paths to slash only"})


class GPUEnforcer:
    """Place Dense, gradient creation and Adam updates on GPU, fail on fallback.

    The original training/CLI source and its sampling, seeds, model, and guards
    are unmodified. Host-side I/O, summaries, metrics and sampling stay on CPU.
    """
    def __init__(self, output):
        self.output = Path(output)
        self.evidence = {"dense": {}, "gradient": {}, "matmul": {}, "optimizers": []}
        self.require_gpu = True
        self.tf, self.keras = tensorflow_runtime()
        tf = self.tf
        physical = tf.config.list_physical_devices("GPU")
        if len(physical) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise RuntimeError(f"Exactly the selected GPU is required: {physical}")
        for device in physical:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_soft_device_placement(False)
        details = tf.config.experimental.get_device_details(physical[0])
        write_json(self.output / "tensorflow_gpu.json", {
            "selected_uuid": GPU_UUID, "physical_devices": [str(x) for x in physical],
            "details": details, "build": tf.sysconfig.get_build_info(),
            "soft_device_placement": tf.config.get_soft_device_placement(),
            "memory_growth": tf.config.experimental.get_memory_growth(physical[0]),
            "tf32_default": tf.config.experimental.tensor_float_32_execution_enabled()})
        self.install()

    def require(self, tensor, category, name):
        if tensor is None:
            raise RuntimeError(f"Missing gradient: {name}")
        device = tensor.device
        if "GPU:0" not in device.upper():
            raise RuntimeError(f"CPU fallback prohibited: {category} {name} {device}")
        self.evidence[category][name] = device

    def install(self):
        tf, keras = self.tf, self.keras
        original_build, original_call = keras.layers.Dense.build, keras.layers.Dense.call
        original_gradient = tf.GradientTape.gradient
        original_apply, original_init = keras.optimizers.Adam.apply_gradients, keras.optimizers.Adam.__init__

        @functools.wraps(original_build)
        def build(layer, *args, **kwargs):
            if not self.require_gpu:
                return original_build(layer, *args, **kwargs)
            with tf.device("/GPU:0"):
                result = original_build(layer, *args, **kwargs)
            for var in layer.trainable_variables:
                self.require(var, "dense", var.name)
            return result

        @functools.wraps(original_call)
        def call(layer, inputs):
            # Keras inspects the actual argspec. **kwargs would advertise
            # support for training/mask, which Dense.call does not accept.
            if not self.require_gpu:
                return original_call(layer, inputs)
            with tf.device("/GPU:0"):
                result = original_call(layer, inputs)
            self.require(result, "dense", layer.name + "/output")
            return result

        @functools.wraps(original_gradient)
        def gradient(tape, *args, **kwargs):
            with tf.device("/GPU:0"):
                result = original_gradient(tape, *args, **kwargs)
            for i, tensor in enumerate(tf.nest.flatten(result)):
                self.require(tensor, "gradient", getattr(tensor, "name", f"eager_{i}"))
            if not tf.executing_eagerly():
                for op in tf.compat.v1.get_default_graph().get_operations():
                    if op.type in ("MatMul", "BatchMatMul", "BatchMatMulV2"):
                        if "GPU:0" not in op.device.upper():
                            raise RuntimeError(f"MatMul lacks explicit GPU placement: {op.name} {op.device}")
                        self.evidence["matmul"][op.name] = {"device": op.device, "type": op.type}
            self.save()
            return result

        @functools.wraps(original_init)
        def initialize(optimizer, *args, **kwargs):
            with tf.device("/GPU:0"):
                original_init(optimizer, *args, **kwargs)
            self.evidence["optimizers"].append({"sequence": len(self.evidence["optimizers"]),
                                                 "python_id": id(optimizer), "class": type(optimizer).__name__})

        @functools.wraps(original_apply)
        def apply(optimizer, grads_and_vars, *args, **kwargs):
            pairs = list(grads_and_vars)
            for grad, var in pairs:
                self.require(grad, "gradient", var.name + "/gradient")
                self.require(var, "dense", var.name)
            with tf.device("/GPU:0"):
                result = original_apply(optimizer, pairs, *args, **kwargs)
            for var in optimizer.variables():
                self.require(var, "dense", "optimizer/" + var.name)
            self.save()
            return result

        keras.layers.Dense.build, keras.layers.Dense.call = build, call
        tf.GradientTape.gradient = gradient
        keras.optimizers.Adam.apply_gradients, keras.optimizers.Adam.__init__ = apply, initialize

    def save(self):
        write_json(self.output / "device_evidence.json", self.evidence)

    def libraries(self):
        lines = Path("/proc/self/maps").read_text().splitlines()
        paths = sorted({line.split()[-1] for line in lines if any(lib in line for lib in
                       ("libcuda", "libcublas", "libcudnn", "libcufft", "libcusolver", "libcusparse", "libnvrtc"))})
        prefix = str(Path(sys.prefix).resolve()) + "/"
        wrong = [path for path in paths if "/libcuda.so" not in path and not path.startswith(prefix)]
        write_json(self.output / "loaded_cuda_libraries.json", {"paths": paths, "unexpected": wrong})
        if wrong:
            raise RuntimeError(f"CUDA libraries outside selected virtual environment: {wrong}")


def checkpoint_parity(guard, x, sample_ids):
    # The numerical parity check alone runs this checkpoint on CPU.
    tf = guard.tf
    output = guard.output
    checkpoint = Path(os.environ["CAPSTONE_TF_CACHE"]) / "normalized_windows_classifier.keras"
    guard.require_gpu = False
    tf32_default = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    with tf.device("/CPU:0"):
        cpu_model = load_classifier(checkpoint)
        cpu_output = cpu_model(x, training=False).numpy()
    guard.require_gpu = True
    with tf.device("/GPU:0"):
        gpu_model = load_classifier(checkpoint)
        gpu_tensor = gpu_model(x, training=False)
        guard.require(gpu_tensor, "dense", "parity/output")
        gpu_output = gpu_tensor.numpy()
    np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-4, atol=1e-5)
    np.save(Path(output) / "parity_input.npy", x, allow_pickle=False)
    np.save(Path(output) / "parity_cpu_output.npy", cpu_output, allow_pickle=False)
    np.save(Path(output) / "parity_gpu_output.npy", gpu_output, allow_pickle=False)
    write_json(Path(output) / "cpu_gpu_parity.json", {"passed": True, "rtol": 1e-4, "atol": 1e-5,
               "max_abs_difference": float(np.max(np.abs(gpu_output - cpu_output))),
               "sample_ids": sample_ids, "tf32_disabled_for_parity_only": True,
               "cpu_use": "existing checkpoint numerical parity only; no CPU training"})
    tf.config.experimental.enable_tensor_float_32_execution(tf32_default)
    del cpu_model, gpu_model


def validation_worker(output, validation_name):
    def phase(name):
        write_json(Path(output) / "phase.json", {"phase": name, "started_at": now()})
        print(f"VALIDATION_PHASE={name}", flush=True)

    phase("load_development_cache")
    baseline, protocol = settings()
    folds, _ = load_splits(protocol)
    store = FeatureStore(baseline)
    features = training.load_features(store, [r for r in folds[0] if r["split"] != "test"])
    phase("initialize_gpu")
    guard = GPUEnforcer(output)
    tf, keras = guard.tf, guard.keras
    tf.config.experimental.enable_op_determinism()
    sampler = GroupSampler(folds[0], features, protocol, 4, 42)
    x, y, sample_ids = sampler.draw(30)
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    phase("gpu_forward_backward_optimizer")
    model = keras.Sequential([
        keras.layers.Input(shape=(baseline["embedding_dimension"],)),
        keras.layers.Dense(baseline["hidden_units"], activation="relu", kernel_initializer=keras.initializers.GlorotUniform(seed=42)),
        keras.layers.Dropout(baseline["dropout"], seed=43),
        keras.layers.Dense(len(LABELS), kernel_initializer=keras.initializers.GlorotUniform(seed=44)),
    ], name="yamnet_embedding_classifier")
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    before = [v.numpy().copy() for v in model.trainable_variables]

    @tf.function
    def step(a, b):
        with tf.GradientTape() as tape:
            loss = loss_fn(b, model(a, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    tf.debugging.set_log_device_placement(True)
    with tf.device("/GPU:0"):
        loss, gradients = step(x, y)
    value = float(loss.numpy())
    for i, grad in enumerate(gradients):
        guard.require(grad, "gradient", f"executed_probe_gradient_{i}")
        if not np.isfinite(grad.numpy()).all():
            raise RuntimeError("Non-finite GPU probe gradient")
    tf.debugging.set_log_device_placement(False)
    changed = [not np.array_equal(a, v.numpy()) for a, v in zip(before, model.trainable_variables)]
    if not np.isfinite(value) or not all(changed) or int(optimizer.iterations.numpy()) != 1:
        raise RuntimeError("GPU forward/backward/Adam update was not verified")
    write_json(Path(output) / "gpu_update.json", {"passed": True, "loss": value,
               "all_trainable_tensors_changed": changed, "optimizer_iterations": int(optimizer.iterations.numpy()),
               "architecture": "1024 -> Dense(128,relu) -> Dropout(0.2) -> Dense(3)"})
    guard.save()
    guard.libraries()
    # Real fold 0, 1 epoch x 2 steps, through the original training CLI.
    phase("fold0_1epoch_2steps_and_reload")
    run_native(guard, 4, validation_name, validation=True)
    directory = ROOT / baseline["runs_dir"] / validation_name
    restored = load_classifier(directory / "classifier.keras")
    predictions = training.predict_clips(restored, [r for r in folds[0] if r["split"] == "validation"], features)
    saved = read_csv(directory / "validation_clips.csv")
    if [r["sample_id"] for r in predictions] != [r["sample_id"] for r in saved]:
        raise RuntimeError("Reloaded validation sample inventory differs")
    columns = [f"p_{label}" for label in LABELS]
    current_values = np.array([[r[c] for c in columns] for r in predictions])
    saved_values = np.array([[float(r[c]) for c in columns] for r in saved])
    np.testing.assert_allclose(current_values, saved_values, rtol=1e-4, atol=1e-5)
    result = read_json(directory / "result.json")
    if result["epochs_completed"] != 1 or sum(int(r["draws"]) for r in read_csv(directory / "sample_exposure.csv")) != 60:
        raise RuntimeError("Validation must be exactly 1 epoch, 2 steps, batch 30")
    write_json(Path(output) / "validation_pass.json", {"passed": True, "run": str(directory),
               "epochs": 1, "steps": 2, "validation_clips": len(predictions), "excluded_from_cv": True,
               "reload_max_abs_difference": float(np.max(np.abs(current_values - saved_values))),
               "checkpoint_sha256": sha256(directory / "classifier.keras")})
    phase("existing_checkpoint_cpu_gpu_parity")
    checkpoint_parity(guard, x, sample_ids)
    guard.save()
    guard.libraries()
    phase("complete")
    print("GPU_VALIDATION_PASS", flush=True)


def run_native(guard, weight, name, validation=False):
    from sound_baseline.__main__ import parser
    baseline, protocol = settings()
    folds, _ = load_splits(protocol)
    original_fit = training.fit_head

    def tracked_fit(rows, features, baseline, protocol, options, directory, fold=0, kind="development"):
        started = time.monotonic()
        optimizer_start = len(guard.evidence["optimizers"])
        result = original_fit(rows, features, baseline, protocol, options, directory, fold, kind)
        optimizers = guard.evidence["optimizers"][optimizer_start:]
        if len(optimizers) != 1:
            raise RuntimeError(f"Each fold must construct one new optimizer: {optimizers}")
        write_json(directory / "gpu_execution.json", {"gpu_uuid": GPU_UUID, "fold_seed": options["seed"] + fold,
                   "elapsed_seconds": time.monotonic() - started, "optimizer": optimizers[0],
                   "explicit_dense_gradient_adam_gpu": True, "soft_device_placement": False,
                   "original_cli_args": training_args(weight, name, validation)})
        return result

    training.fit_head = tracked_fit
    try:
        args = parser().parse_args(training_args(weight, name, validation))
        return training.train_command(args, baseline, protocol, folds, FeatureStore(baseline))
    finally:
        training.fit_head = original_fit


def cv_worker(output, weight):
    guard = GPUEnforcer(output)
    run_native(guard, weight, NAMES[weight])
    guard.save()
    guard.libraries()
    print(f"GPU_CV_WEIGHT_{weight}_PASS", flush=True)


def run_logged(command, log, events, timeout=None):
    started = time.monotonic()
    event = {"command": [str(x) for x in command], "command_text": command_text(command),
             "started_at": now(), "log": str(log), "timeout_seconds": timeout}
    print(f"START {event['command_text']} -> {log}", flush=True)
    with Path(log).open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        event["pid"] = process.pid
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            event["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            code = process.returncode
        event.update(exit_code=code, elapsed_seconds=time.monotonic()-started, finished_at=now())
    events.append(event)
    if code or event.get("timed_out"):
        raise RuntimeError(f"{'Timeout' if event.get('timed_out') else 'Execution failed'}: exit={code}; first-error log={log}")
    return event


def verify_prior_validation(directory, profile, current_files, packages):
    """Bind a completed GPU validation to unchanged code, data and packages."""
    directory = directory.resolve()
    directory.relative_to(ROOT / "reports")
    status = read_json(directory / "status.json")
    if not (status.get("scope") == "gpu_validation_only"
            and status.get("environment_profile") == profile
            and status.get("experiment_completed") and status.get("gpu_validation_passed")
            and status.get("integrity_passed") and status.get("exit_code") == 0):
        raise RuntimeError("Prior validation is not a completed, preserved pass for this profile")
    integrity = read_json(directory / "integrity.json")
    if not (integrity["protected_hashes_unchanged"] and integrity["package_lists_unchanged"]
            and integrity["fixed_test_embeddings"] == 0):
        raise RuntimeError("Prior validation integrity did not pass")
    prior_files = read_json(directory / "protected_after.json")["files"]
    changed = {name: [digest, current_files.get(name)] for name, digest in prior_files.items()
               if digest != current_files.get(name)}
    if changed:
        raise RuntimeError(f"Inputs changed since GPU validation: {changed}")
    if packages != read_json(directory / "packages_gpu.json"):
        raise RuntimeError("Selected packages changed since GPU validation")
    previous_helper = directory / "executed_helpers/gpu_cv.py"
    def nodes(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    old_nodes, new_nodes = nodes(previous_helper), nodes(Path(__file__))
    for name in ("GPUEnforcer", "run_native", "training_args", "cv_worker"):
        if old_nodes[name] != new_nodes[name]:
            raise RuntimeError(f"GPU execution code changed since validation: {name}")
    environment = read_json(directory / "environment.json")
    if environment["helper_sha256"]["activate_gpu.sh"] != sha256(ROOT / "tools/activate_gpu.sh"):
        raise RuntimeError("GPU activation script changed since validation")
    gpu = read_json(directory / "gpu_validation/tensorflow_gpu.json")
    if gpu["selected_uuid"] != GPU_UUID:
        raise RuntimeError("Prior validation used a different GPU")
    evidence = ["status.json", "integrity.json", "packages_gpu.json", "environment.json",
                "checkpoint_normalization.json", "executed_helpers/gpu_cv.py",
                *["gpu_validation/" + name for name in ("tensorflow_gpu.json", "device_evidence.json",
                    "loaded_cuda_libraries.json", "validation_pass.json", "gpu_update.json", "cpu_gpu_parity.json")]]
    for name in ("validation_pass.json", "gpu_update.json", "cpu_gpu_parity.json"):
        if not read_json(directory / "gpu_validation" / name)["passed"]:
            raise RuntimeError(f"Prior GPU validation did not pass: {name}")
    reload = read_json(directory / "gpu_validation/validation_pass.json")
    if sha256(Path(reload["run"]) / "classifier.keras") != reload["checkpoint_sha256"]:
        raise RuntimeError("Prior GPU save/reload checkpoint changed")
    return {"passed": True, "directory": str(directory), "gpu_uuid": GPU_UUID,
            "environment_profile": profile, "protected_inputs_checked": len(prior_files),
            "unchanged_gpu_execution_nodes": ["GPUEnforcer", "run_native", "training_args", "cv_worker"],
            "evidence_sha256": {name: sha256(directory / name) for name in evidence},
            "note": "Reuse the verified minimal run; it remains excluded from regular CV. No CPU model execution in this CV invocation."}


def run_comparison(output, validation_only=False, profile="pinned218", validated_report=None):
    started = time.monotonic()
    deadline = started + 3600
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "gpu_training_validation" if validation_only else "ubuntu_gpu_cv_s42"
    if profile != "pinned218":
        prefix += "_" + profile
    output = (output or ROOT / "reports" / f"{prefix}_{stamp}").resolve()
    output.relative_to(ROOT / "reports")
    output.mkdir(parents=True, exist_ok=False)
    events, roots, before = [], [], None
    status = {"started_at": now(), "experiment_completed": False, "gpu_validation_passed": False,
              "completed_weights": [], "test_evaluated": False, "refit_executed": False,
              "installation_executed": False, "stage": "preflight", "report_directory": str(output),
              "scope": "gpu_validation_only" if validation_only else ("regular_gpu_cv" if validated_report else "gpu_validation_and_cv"),
              "environment_profile": profile}
    write_json(output / "status.json", status)
    print(f"REPORT_DIRECTORY={output}", flush=True)
    error = None
    def validation_time_left():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Initial GPU validation exceeded 3600 seconds; CV prohibited")
        return remaining
    try:
        if validation_only and validated_report:
            raise ValueError("A minimal validation cannot reuse another validation")
        selected_profile = PROFILES[profile]
        if Path(sys.prefix).resolve() != ROOT / selected_profile["venv"] or platform.python_version_tuple()[:2] != ("3", "12"):
            raise RuntimeError(f"Must run under Python 3.12 {selected_profile['venv']}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID or os.environ.get("WANDB_MODE") != "disabled":
            raise RuntimeError("Source tools/activate_gpu.sh before running")
        names = [ROOT / "runs/yamnet" / name for name in NAMES.values()]
        conflicts = [str(p) for p in names if p.exists()]
        if conflicts:
            raise FileExistsError(f"Run names already exist; refusing overwrite: {conflicts}")
        validation_name = f"ubuntu_gpu_validation_1e2s_s42_{stamp}"
        if not validated_report:
            status["validation_run_name"] = validation_name
        roots = protected_roots()
        before = snapshot(roots)
        write_json(output / "protected_before.json", {"roots": [str(p) for p in roots], "files": before,
                   "fingerprint": fingerprint(before)})
        base_packages = environment_packages(ROOT / ".venv/bin/python")
        gpu_packages = environment_packages(sys.executable)
        reference_gpu_packages = environment_packages(ROOT / ".venv-gpu/bin/python")
        write_json(output / "packages_original.json", base_packages)
        write_json(output / "packages_gpu.json", gpu_packages)
        write_json(output / "packages_reference_gpu.json", reference_gpu_packages)
        differences = {key: [base_packages.get(key), gpu_packages.get(key)]
                       for key in base_packages.keys() | gpu_packages.keys()
                       if not key.startswith("nvidia-") and base_packages.get(key) != gpu_packages.get(key)}
        required = CORE | NVIDIA | selected_profile["changes"]
        expected = base_packages | required
        bad_versions = {key: [gpu_packages.get(key), value] for key, value in expected.items()
                        if gpu_packages.get(key) != value}
        unexpected = sorted(gpu_packages.keys() - expected.keys())
        write_json(output / "package_comparison.json", {"non_cuda_differences": differences,
                   "fixed_version_mismatches": bad_versions, "unexpected_packages": unexpected,
                   "environment_profile": profile, "approved_profile_changes": selected_profile["changes"],
                   "packages_installed_by_this_run": []})
        if bad_versions or unexpected:
            raise RuntimeError(f"Package profile mismatch; do not auto-install: {bad_versions}, {unexpected}")
        write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "variables": {key: os.environ.get(key) for key in (
                       "VIRTUAL_ENV", "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                       "CAPSTONE_TF_CACHE", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "XLA_FLAGS",
                       "CUBLAS_WORKSPACE_CONFIG", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_USE_LEGACY_KERAS",
                       "WANDB_MODE", "WANDB_DISABLED", "PYTHONNOUSERSITE")},
                   "helper_sha256": {p.name: sha256(p) for p in (Path(__file__), ROOT/"tools/activate_gpu.sh", ROOT/"tools/gpu_cv_report.py")}})
        import shutil
        helper_directory = output / "executed_helpers"
        helper_directory.mkdir()
        for path in (Path(__file__), ROOT/"tools/activate_gpu.sh", ROOT/"tools/gpu_cv_report.py"):
            shutil.copy2(path, helper_directory / path.name)
        run_logged(["nvidia-smi", "-i", GPU_UUID, "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used", "--format=csv"],
                   output / "gpu_info.csv", events, validation_time_left())
        run_logged(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory", "--format=csv"],
                   output / "gpu_processes_before.csv", events, validation_time_left())
        if len((output / "gpu_processes_before.csv").read_text().strip().splitlines()) > 1:
            raise RuntimeError("An existing GPU compute process was found; do not share or terminate it")
        run_logged([sys.executable, "-m", "pip", "freeze", "--all"], output / "packages_gpu_freeze.txt", events, validation_time_left())
        run_logged([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze", "--all"], output / "packages_original_freeze.txt", events, validation_time_left())
        run_logged([sys.executable, "-m", "pip", "check"], output / "pip_check.log", events, validation_time_left())
        run_logged([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], output / "unit_tests.log", events, validation_time_left())
        if "Ran 7 tests" not in (output / "unit_tests.log").read_text():
            raise RuntimeError("Expected the existing seven unit tests")
        audit = cache_audit()
        write_json(output / "cache_before.json", audit)
        write_json(output / "planned_commands.json", {"validation": [] if validated_report else [sys.executable, "-m", "sound_baseline", *training_args(4, validation_name, True)],
                   "cv": [] if validation_only else [[sys.executable, "-m", "sound_baseline", *training_args(w, NAMES[w])] for w in NAMES],
                   "note": "Commands are invoked by GPU-enforcing companion in the same worker process; no test/refit/extract."})
        status["stage"] = "gpu_validation"
        write_json(output / "status.json", status)
        if validated_report:
            reused = verify_prior_validation(validated_report, profile, before, gpu_packages)
            write_json(output / "reused_gpu_validation.json", reused)
            status["validated_report"] = reused["directory"]
            status["minimal_validation_reexecuted"] = False
        else:
            normalized = Path(os.environ["CAPSTONE_TF_CACHE"]) / "normalized_windows_classifier.keras"
            normalize_checkpoint(normalized, output / "checkpoint_normalization.json")
            validation_output = output / "gpu_validation"
            validation_output.mkdir()
            run_logged([sys.executable, str(Path(__file__)), "validation-worker", "--output", str(validation_output),
                        "--validation-name", validation_name], output / "gpu_validation.log", events, validation_time_left())
            if not all(read_json(validation_output / filename)["passed"] for filename in
                       ("validation_pass.json", "gpu_update.json", "cpu_gpu_parity.json")):
                raise RuntimeError("Missing GPU validation pass evidence")
        status.update(gpu_validation_passed=True, initial_validation_seconds=time.monotonic()-started)
        write_json(output / "status.json", status)
        for weight in ([] if validation_only else NAMES):
            status["stage"] = f"cv_weight_{weight}"
            write_json(output / "status.json", status)
            child_output = output / f"weight_{weight}"
            child_output.mkdir()
            run_logged([sys.executable, str(Path(__file__)), "cv-worker", "--output", str(child_output), "--weight", str(weight)],
                       output / f"cv_weight_{weight}.log", events)
            status["completed_weights"].append(weight)
            write_json(output / "status.json", status)
        status.update(experiment_completed=True, stage="complete")
    except Exception as exc:
        error = exc
        status.update(stage="stopped", first_error=str(exc), first_error_type=type(exc).__name__)
        (output / "first_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"STOPPED: {exc}", flush=True)
    finally:
        write_json(output / "execution_events.json", events)
        if before is not None:
            try:
                after = snapshot(roots)
                write_json(output / "protected_after.json", {"files": after, "fingerprint": fingerprint(after)})
                differences = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
                package_after = {"original": environment_packages(ROOT / ".venv/bin/python"), "gpu": environment_packages(sys.executable),
                                 "reference_gpu": environment_packages(ROOT / ".venv-gpu/bin/python")}
                write_json(output / "packages_after.json", package_after)
                package_unchanged = package_after == {"original": read_json(output/"packages_original.json"), "gpu": read_json(output/"packages_gpu.json"),
                                                      "reference_gpu": read_json(output/"packages_reference_gpu.json")}
                final_audit = cache_audit()
                write_json(output / "cache_after.json", final_audit)
                integrity = {"protected_file_count": len(before), "protected_hashes_unchanged": not differences,
                             "differences": differences, "package_lists_unchanged": package_unchanged,
                             "fixed_test_embeddings": final_audit["fixed_test_embeddings"], "test_evaluated": False,
                             "before_fingerprint": fingerprint(before), "after_fingerprint": fingerprint(after)}
                write_json(output / "integrity.json", integrity)
                if differences or not package_unchanged:
                    raise RuntimeError("Protected file/package integrity changed; inspect integrity.json")
                status["integrity_passed"] = True
            except Exception as exc:
                status.update(integrity_passed=False, integrity_error=str(exc), experiment_completed=False)
                (output / "integrity_error.log").write_text(traceback.format_exc(), encoding="utf-8")
                error = error or exc
        status.update(finished_at=now(), elapsed_seconds=time.monotonic()-started, exit_code=1 if error else 0)
        write_json(output / "status.json", status)
        from gpu_cv_report import make_report
        make_report(output)
        print(f"REPORT_DIRECTORY={output}\nEXIT_CODE={status['exit_code']}", flush=True)
    if error:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "run", "validation-worker", "cv-worker", "summarize"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-name")
    parser.add_argument("--weight", type=int, choices=(1, 2, 4))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="pinned218")
    parser.add_argument("--validated-report", type=Path,
                        help="Reuse a preserved GPU validation after checking identical inputs and execution code")
    args = parser.parse_args()
    if args.stage == "validation-worker":
        validation_worker(args.output, args.validation_name)
    elif args.stage == "cv-worker":
        cv_worker(args.output, args.weight)
    elif args.stage == "summarize":
        from gpu_cv_report import make_report
        make_report(args.output)
    else:
        run_comparison(args.output, validation_only=args.stage == "validate", profile=args.profile,
                       validated_report=args.validated_report)


if __name__ == "__main__":
    main()
