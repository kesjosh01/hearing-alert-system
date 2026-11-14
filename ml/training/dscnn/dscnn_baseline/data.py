"""Read the original splits; require reviewed target intervals for training."""
from __future__ import annotations

import csv
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, YAMNET, config, source, write_json
from .frontend import LogMel, window_starts, windows

ANNOTATION_FIELDS = ("sample_id", "start_s", "end_s", "label", "source_sha256")


def development_splits():
    cfg, protocol = config()
    folds, _ = source.load_splits(protocol)
    # Metadata validation checks frozen membership, but no test audio is decoded.
    extension = load_extension(cfg, protocol, folds)
    return [source.validate_rows([r for r in fold if r["split"] != "test"] + [
        dict(item["row"], split="validation" if item["cv_fold"] == i else "train",
             cv_fold=str(item["cv_fold"]), split_version=extension["version"])
        for item in extension["assignments"]]) for i, fold in enumerate(folds)]


def load_extension(cfg, protocol, folds):
    """Validate a local, append-only development extension; preserve old holdout."""
    relative = cfg.get("development_extension_path")
    if not relative:
        return {"version": protocol["version"], "assignments": []}
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise ValueError("Development extension must stay inside DS-CNN")
    extension = source.read_json(path)
    if extension["parent_protocol"] != protocol["version"]:
        raise ValueError("Extension parent protocol changed")
    if not extension.get("source_independence_confirmed_by_user"):
        raise ValueError("New source independence requires user confirmation")
    for i in range(len(folds)):
        if extension["parent_split_sha256"][str(i)] != source.sha256(source.split_path(protocol, i)):
            raise ValueError("Frozen parent split changed")
    original = {r["sample_id"] for r in folds[0]}
    groups = {r["split_group_id"] for r in folds[0]}
    sources = {(r["source_dataset"], r["source_recording_id"]) for r in folds[0]}
    ids, new_groups, new_sources = set(), set(), set()
    for item in extension["assignments"]:
        row = item["row"]
        sample = row["sample_id"]
        group = row["split_group_id"]
        recording = (row["source_dataset"], row["source_recording_id"])
        if sample in original or sample in ids or group in groups or group in new_groups or recording in sources or recording in new_sources:
            raise ValueError(f"Extension source overlaps a frozen or repeated source: {sample}")
        if type(item["cv_fold"]) is not int or item["cv_fold"] not in range(4):
            raise ValueError("Invalid extension fold")
        if row["target_domain"] != "ko_target" or row["sampler_pool"] != "domestic_target":
            raise ValueError("Extension only accepts declared domestic targets")
        if row["label"] not in ("fire_alarm", "doorbell"):
            raise ValueError("Extension target class must be fire_alarm or doorbell")
        relative_audio = Path(row["relative_path"])
        if relative_audio.is_absolute() or ".." in relative_audio.parts:
            raise ValueError("Out-of-project extension audio")
        if source.sha256(YAMNET / relative_audio) != item["source_sha256"]:
            raise ValueError(f"Extension audio hash changed: {sample}")
        ids.add(sample); new_groups.add(group); new_sources.add(recording)
    return extension


def development_refit_rows():
    cfg, protocol = config()
    folds, refit = source.load_splits(protocol)
    extension = load_extension(cfg, protocol, folds)
    return source.validate_rows([r for r in refit if r["split"] != "test"] + [
        dict(item["row"], split="train", cv_fold=str(item["cv_fold"]),
             split_version=extension["version"]) for item in extension["assignments"]])


def development_identity():
    cfg, protocol = config()
    identity = {"parent_protocol": protocol["version"],
                "effective_fold_fingerprints": [source.fingerprint(rows) for rows in development_splits()]}
    if cfg.get("development_extension_path"):
        path = ROOT / cfg["development_extension_path"]
        identity.update(extension_version=source.read_json(path)["version"],
                        extension_sha256=source.sha256(path))
    return identity


def audit():
    cfg, protocol = config()
    folds, _ = source.load_splits(protocol)
    development = development_splits()
    return {
        "kind": "metadata_only_no_audio_decoding",
        "labels": cfg["labels"], "protocol_version": protocol["version"],
        "folds": [{"fold": i, "rows_by_role": dict(Counter(r["split"] for r in rows)),
                   "source_groups_by_role": {role: len({r["split_group_id"] for r in rows if r["split"] == role})
                                             for role in ("train", "validation", "test")},
                   "split_sha256": source.sha256(source.split_path(protocol, i))}
                  for i, rows in enumerate(folds)],
        "test_audio_decoded": 0, "test_features_created": 0, "test_evaluated": False,
        "positive_development_clips_needing_annotations": sum(r["label"] != "other" for r in development[0]),
        "effective_development_folds": [{"fold": i, "rows_by_role": dict(Counter(r["split"] for r in rows))}
                                        for i, rows in enumerate(development)],
        "development_identity": development_identity(),
    }


def require_development(row):
    if row["split"] not in ("train", "validation"):
        raise ValueError(f"Fixed test is unavailable to the DS-CNN development pipeline: {row['sample_id']}")


def read_annotations(path, rows):
    allowed = {r["sample_id"]: r for r in rows}
    if any(r["split"] == "test" for r in rows):
        raise ValueError("Annotation inventory must exclude fixed test")
    annotations = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ANNOTATION_FIELDS:
            raise ValueError(f"Annotation columns must be {ANNOTATION_FIELDS}")
        for item in reader:
            sample = item["sample_id"]
            if sample not in allowed:
                raise ValueError(f"Unknown or held-out annotation: {sample}")
            row = allowed[sample]
            if row["label"] == "other" or item["label"] != row["label"]:
                raise ValueError(f"Annotation must match an original target class: {sample}")
            start, end = float(item["start_s"]), float(item["end_s"])
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError(f"Invalid event interval: {sample}")
            if len(item["source_sha256"]) != 64 or item["source_sha256"] != source.sha256(YAMNET / row["relative_path"]):
                raise ValueError(f"Annotation source hash changed: {sample}")
            annotations[sample].append((start, end))
    missing = sorted(r["sample_id"] for r in rows if r["label"] != "other" and r["sample_id"] not in annotations)
    if missing:
        raise ValueError(f"Reviewed event intervals required for {len(missing)} development target clips; first: {missing[:5]}")
    # Merge overlapping/adjacent annotations before computing coverage.
    for sample, intervals in annotations.items():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        annotations[sample] = merged
    return dict(annotations)


def positive_window_indices(starts, intervals, coverage=0.8, sample_rate=16000):
    """Keep windows covering >= coverage * min(event duration, 1 second)."""
    if not 0 < coverage <= 1:
        raise ValueError("Event coverage must be in (0, 1]")
    return [i for i, start in enumerate(starts)
            if any(max(0, min(start / sample_rate + 1, end) - max(start / sample_rate, begin))
                   >= coverage * min(1, end - begin) - 1e-9 for begin, end in intervals)]


class FeatureStore:
    def __init__(self, cache=None):
        self.cache = Path(cache or ROOT / ".cache/features").resolve()
        if self.cache == YAMNET or YAMNET in self.cache.parents:
            raise ValueError("Use a separate DS-CNN cache")
        self.frontend = LogMel()
        self.signature = source.fingerprint({"frontend": self.frontend.options,
                                             "frontend_code": source.sha256(Path(__file__).with_name("frontend.py"))})

    def load(self, row):
        require_development(row)
        return self._load_audio(row)

    def _load_audio(self, row):
        audio = YAMNET / row["relative_path"]
        digest = source.sha256(audio)
        identity = source.fingerprint([digest, self.signature])
        directory = self.cache / identity
        array_path, meta_path = directory / "logmel.npy", directory / "metadata.json"
        if not meta_path.is_file():
            if directory.exists():
                raise ValueError(f"Incomplete DS-CNN cache; preserve/review before retrying: {directory}")
            directory.mkdir(parents=True, exist_ok=False)
            # Decode only to a new cache. Original WAVs and YAMNet caches are read-only.
            from sound_baseline.features import decode_audio
            decode_audio(audio, directory / "audio.wav")
            with wave.open(str(directory / "audio.wav"), "rb") as handle:
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            features = np.stack([self.frontend(chunk) for _, chunk in windows(pcm)])
            np.save(array_path, features, allow_pickle=False)
            write_json(meta_path, {"source_sha256": digest, "frontend_signature": self.signature,
                                  "samples": len(pcm), "starts": window_starts(len(pcm)),
                                  "feature_sha256": source.sha256(array_path)})
        meta = source.read_json(meta_path)
        if meta["source_sha256"] != digest or meta["frontend_signature"] != self.signature or source.sha256(array_path) != meta["feature_sha256"]:
            raise ValueError(f"DS-CNN cache integrity error: {row['sample_id']}")
        features = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if features.shape != (len(meta["starts"]), *self.frontend.shape) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError(f"Invalid log-Mel tensor: {row['sample_id']}")
        return features, meta


def load_fold(rows, annotations, coverage):
    store = FeatureStore()
    features, provenance = {}, []
    for row in rows:
        require_development(row)
        full, meta = store.load(row)
        intervals = annotations.get(row["sample_id"], [])
        if any(end > meta["samples"] / 16000 + 1 / 16000 for _, end in intervals):
            raise ValueError(f"Annotation exceeds actual recording: {row['sample_id']}")
        chosen = list(range(len(full)))
        if row["split"] == "train" and row["label"] != "other":
            chosen = positive_window_indices(meta["starts"], intervals, coverage)
            if not chosen:
                raise ValueError(f"No annotated training windows: {row['sample_id']}")
        # Do not hold the full feature set in RAM. The sampler reads one window.
        features[row["sample_id"]] = WindowView(full, chosen)
        provenance.append({"sample_id": row["sample_id"], "source_sha256": meta["source_sha256"],
                           "feature_sha256": meta["feature_sha256"], "window_indices": chosen})
    return features, provenance


def load_file_label_fold(rows):
    store = FeatureStore()
    features, provenance = {}, []
    for row in rows:
        require_development(row)
        full, meta = store.load(row)
        features[row["sample_id"]] = full
        provenance.append({"sample_id": row["sample_id"], "source_sha256": meta["source_sha256"],
                           "feature_sha256": meta["feature_sha256"], "window_count": len(full),
                           "supervision": "file_label_bag", "window_labels_assigned": False})
    return features, provenance


class WindowView:
    def __init__(self, array, indices):
        self.array, self.indices = array, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.array[self.indices[index]]
        return self.array[self.indices[index]]
