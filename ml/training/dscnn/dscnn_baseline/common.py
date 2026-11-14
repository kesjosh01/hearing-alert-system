"""Shared immutable YAMNet split access and explicit runtime setup."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
YAMNET = ROOT.parent / "yamnet"
WANDB_MODES = ("disabled", "offline", "online")
sys.path.insert(0, str(YAMNET))
# Import only the NumPy/metadata layer. Do not invoke YAMNet feature extraction.
from sound_baseline import core as source  # noqa: E402


def config():
    value = source.read_json(ROOT / "config/dscnn_v1.json")
    _, protocol = source.settings()
    if value["labels"] != source.LABELS or value["protocol_version"] != protocol["version"]:
        raise ValueError("DS-CNN label map or frozen source protocol changed")
    return value, protocol


def new_directory(path):
    path = Path(path).resolve()
    # Keep generated data away from existing YAMNet data and model artefacts.
    if path == YAMNET or YAMNET in path.parents:
        raise ValueError("DS-CNN outputs must not be inside the YAMNet directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def configure_wandb(mode="disabled"):
    """Keep checks local unless this process explicitly selects W&B recording."""
    if mode not in WANDB_MODES:
        raise ValueError(f"Invalid W&B mode: {mode}")
    os.environ["WANDB_MODE"] = mode
    if mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)


def runtime(device="cpu", wandb_mode="disabled"):
    """Select CPU explicitly for software checks; GPU training is fail-closed."""
    if "tensorflow" in sys.modules:
        raise RuntimeError("Select the device before importing TensorFlow")
    cache = ROOT / ".cache/runtime"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["KERAS_HOME"] = str(cache / "keras")
    os.environ["XDG_CACHE_HOME"] = str(cache / "xdg")
    configure_wandb(wandb_mode)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device != "gpu" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        raise RuntimeError("GPU training requires one explicitly selected GPU UUID in CUDA_VISIBLE_DEVICES")
    import tensorflow as tf
    import tf_keras as keras
    if device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) != 1:
            raise RuntimeError(f"Exactly one selected GPU is required, found {len(gpus)}")
        tf.config.set_soft_device_placement(False)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
    return tf, keras


def write_json(path, value):
    source.write_json(path, value)


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
