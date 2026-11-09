"""FFmpeg -> 16 kHz PCM WAV -> frozen YAMNet native-patch embeddings."""
from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

import numpy as np

from .core import ROOT, fingerprint, read_json, sha256, write_json


def runtime_cache():
    override = os.environ.get("CAPSTONE_TF_CACHE")
    if override:
        directory = Path(override).resolve()
    elif str(ROOT).isascii():
        directory = ROOT / ".cache/tensorflow"
    elif os.name == "nt":
        directory = Path(os.environ["LOCALAPPDATA"]) / "CapstoneSoundCache"
    else:
        directory = Path.home() / ".cache/capstone-sound"
    if os.name == "nt" and not str(directory).isascii():
        raise RuntimeError("TensorFlow on Windows needs an ASCII cache path. Set CAPSTONE_TF_CACHE to a writable English-only path.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def tensorflow_runtime():
    # YAMNet/TF Hub use legacy Keras. Set this before the first TF import.
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cache = runtime_cache()
    os.environ["KERAS_HOME"] = str(cache / "keras")
    os.environ["TFHUB_CACHE_DIR"] = str(cache / "tfhub")
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    # Some managed Windows environments have an inaccessible default temp dir.
    import tempfile
    tempfile.tempdir = str(cache / "tmp")
    try:
        import tensorflow as tf
        import tf_keras as keras
    except ImportError as error:
        raise RuntimeError("TensorFlow import failed. Activate the project virtual environment and install "
                           f"requirements-yamnet.txt. Import error: {error}") from error
    return tf, keras


def save_classifier(model, destination):
    # TF native file APIs on Windows can reject Korean paths. Copy via Python.
    temporary = runtime_cache() / "checkpoints" / f"{uuid.uuid4().hex}.keras"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(temporary))
        shutil.copy2(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_classifier(source):
    _, keras = tensorflow_runtime()
    temporary = runtime_cache() / "classifiers" / f"{sha256(source)}.keras"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if not temporary.is_file():
        shutil.copy2(source, temporary)
    return keras.models.load_model(str(temporary), compile=False)


def ffmpeg_binary():
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg not found in PATH. Install FFmpeg; see UBUNTU.md or TRAINING_GUIDE.md.")
    return executable


def decode_audio(source, destination, sample_rate=16000):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.wav")
    result = subprocess.run([ffmpeg_binary(), "-nostdin", "-v", "error", "-y", "-i", str(source),
                             "-map", "0:a:0", "-vn", "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(temporary)],
                            capture_output=True, timeout=180)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed for {source.name}: {result.stderr.decode('utf-8', errors='replace')}")
    read_wave(temporary, sample_rate)  # Reject empty/malformed output before committing.
    temporary.replace(destination)


def read_wave(path, sample_rate=16000):
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (sample_rate, 1, 2):
            raise ValueError(f"Expected {sample_rate} Hz mono PCM16: {path}")
        waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Empty/non-finite waveform: {path}")
    return waveform


class FeatureStore:
    def __init__(self, baseline):
        self.config = baseline
        self.directory = ROOT / baseline["processed_dir"]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audio_signature = fingerprint({"revision": 1, "sample_rate": baseline["sample_rate"], "channels": 1, "encoding": "PCM16"})
        self.signature = fingerprint({key: baseline[key] for key in (
            "version", "model_handle", "sample_rate", "minimum_samples", "embedding_dimension", "embedding_unit", "amplitude_policy")})
        self._yamnet = None
        self._memory = {}
        self._model_identity = None

    def paths(self, sample_id):
        if Path(sample_id).name != sample_id or any(ch in sample_id for ch in ("/", "\\", ":")):
            raise ValueError("Unsafe sample ID")
        return {"wav": self.directory / "audio" / f"{sample_id}.wav",
                "audio_meta": self.directory / "audio" / f"{sample_id}.json",
                "embedding": self.directory / "embeddings" / f"{sample_id}.npy",
                "embedding_meta": self.directory / "embeddings" / f"{sample_id}.json"}

    def preprocess(self, row):
        paths = self.paths(row["sample_id"])
        source_hash = sha256(ROOT / row["relative_path"])
        meta = read_json(paths["audio_meta"]) if paths["audio_meta"].exists() else {}
        valid = paths["wav"].exists() and meta.get("source_sha256") == source_hash and meta.get("audio_signature") == self.audio_signature
        if not valid or meta.get("wav_sha256") != sha256(paths["wav"]):
            decode_audio(ROOT / row["relative_path"], paths["wav"], self.config["sample_rate"])
            waveform = read_wave(paths["wav"], self.config["sample_rate"])
            meta = {"sample_id": row["sample_id"], "source_sha256": source_hash, "wav_sha256": sha256(paths["wav"]),
                    "audio_signature": self.audio_signature, "sample_rate": self.config["sample_rate"], "channels": 1,
                    "samples": len(waveform), "duration_s": len(waveform) / self.config["sample_rate"],
                    "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    "near_clipping_fraction": float(np.mean(np.abs(waveform) >= 0.999)),
                    "near_silence_fraction": float(np.mean(np.abs(waveform) < 0.0001))}
            write_json(paths["audio_meta"], meta)
        return meta

    def yamnet(self):
        if self._yamnet is None:
            tf, _ = tensorflow_runtime()
            try:
                import tensorflow_hub as hub
                resolved = Path(hub.resolve(self.config["model_handle"]))
                self._yamnet = hub.load(str(resolved))
                self._model_identity = {"handle": self.config["model_handle"], "saved_model_sha256": sha256(resolved / "saved_model.pb")}
                write_json(self.directory / "yamnet_identity.json", self._model_identity)
                if int(self._yamnet(tf.zeros([self.config["minimum_samples"]], dtype=tf.float32))[1].shape[-1]) != self.config["embedding_dimension"]:
                    raise ValueError("Unexpected YAMNet embedding shape")
            except Exception as error:
                raise RuntimeError("Cannot load YAMNet. Check internet/model cache and pinned TF/tf-keras versions. " + str(error)) from error
        return self._yamnet

    def embed_waveform(self, waveform):
        waveform = np.asarray(waveform, dtype=np.float32)
        if len(waveform) < self.config["minimum_samples"]:
            waveform = np.pad(waveform, (0, self.config["minimum_samples"] - len(waveform)))
        _, embeddings, _ = self.yamnet()(waveform)
        result = np.asarray(embeddings.numpy(), dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != self.config["embedding_dimension"] or not len(result) or not np.isfinite(result).all():
            raise ValueError("Invalid embedding array")
        return result

    def extract(self, row):
        meta = self.preprocess(row)
        paths = self.paths(row["sample_id"])
        expected = {"source_sha256": meta["source_sha256"], "wav_sha256": meta["wav_sha256"], "feature_signature": self.signature}
        cached = read_json(paths["embedding_meta"]) if paths["embedding_meta"].exists() else {}
        valid = paths["embedding"].exists() and all(cached.get(key) == value for key, value in expected.items())
        if not valid or cached.get("embedding_sha256") != sha256(paths["embedding"]):
            embeddings = self.embed_waveform(read_wave(paths["wav"], self.config["sample_rate"]))
            paths["embedding"].parent.mkdir(parents=True, exist_ok=True)
            temporary = paths["embedding"].with_suffix(".partial.npy")
            np.save(temporary, embeddings, allow_pickle=False)
            temporary.replace(paths["embedding"])
            cached = {**expected, "sample_id": row["sample_id"], "embedding_sha256": sha256(paths["embedding"]),
                      "patch_count": len(embeddings), "dimension": embeddings.shape[1], "yamnet": self._model_identity,
                      "file_label_policy": self.config["file_label_policy"]}
            write_json(paths["embedding_meta"], cached)
        return {"sample_id": row["sample_id"], "duration_s": meta["duration_s"], "patch_count": cached["patch_count"],
                "source_sha256": meta["source_sha256"], "feature_signature": self.signature}

    def load(self, row):
        sample_id = row["sample_id"]
        if sample_id not in self._memory:
            paths = self.paths(sample_id)
            if not paths["embedding"].is_file() or not paths["embedding_meta"].is_file():
                raise ValueError(f"Missing embeddings for {sample_id}. Run: python -m sound_baseline prepare")
            meta = read_json(paths["embedding_meta"])
            if meta.get("feature_signature") != self.signature or meta.get("source_sha256") != sha256(ROOT / row["relative_path"]):
                raise ValueError(f"Stale embedding cache: {sample_id}. Rerun prepare.")
            if meta.get("embedding_sha256") != sha256(paths["embedding"]):
                raise ValueError(f"Corrupt embedding cache: {sample_id}. Rerun prepare.")
            values = np.load(paths["embedding"], allow_pickle=False)
            if values.ndim != 2 or values.shape[1] != self.config["embedding_dimension"] or not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid embeddings: {sample_id}")
            self._memory[sample_id] = values
        return self._memory[sample_id]


def package_versions():
    packages = ("numpy", "tensorflow", "tensorflow-hub", "tf-keras", "setuptools", "wandb")
    return {package: _installed_version(package) for package in packages}


def _installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
