"""Append-only labels, grouped inventories and feature access for the expansion."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import wave

import numpy as np

from dscnn_baseline.common import ROOT, YAMNET, config, source, write_json
from dscnn_baseline.data import FeatureStore, development_splits, development_refit_rows
from dscnn_baseline.frontend import windows, window_starts
from .protocol import DATA, REPORT, VERSION

FIELDS = ("sample_id", "relative_path", "label", "label_id", "target_domain", "source_dataset",
          "source_recording_id", "split_group_id", "split", "split_version", "cv_fold", "sampler_pool",
          "audio_root", "original_label", "label_basis", "source_sha256")


def audio_path(row):
    base = YAMNET if row.get("audio_root", "yamnet") == "yamnet" else ROOT
    relative = Path(row["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Audio must stay within its declared project root")
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file():
        raise ValueError(f"Missing/out-of-root audio: {row['sample_id']}")
    return target


def validate(rows):
    ids, groups, recordings = set(), defaultdict(set), defaultdict(set)
    labels = defaultdict(set)
    for row in rows:
        if row["sample_id"] in ids or row["split"] not in ("train", "validation"):
            raise ValueError("Duplicate ID or fixed test in development")
        ids.add(row["sample_id"])
        if source.LABELS.get(row["label"]) != int(row["label_id"]):
            raise ValueError("Invalid class mapping")
        audio_path(row)
        groups[row["split_group_id"]].add(row["split"])
        labels[row["split_group_id"]].add(row["label"])
        recordings[(row["source_dataset"], str(row["source_recording_id"]))].add(row["split"])
    if any(len(v) != 1 for table in (groups, labels, recordings) for v in table.values()):
        raise ValueError("Source/recording group crosses class or train/validation")
    return rows


def read_rows(fold=None):
    path = DATA / ("refit.csv" if fold is None else f"fold_{fold}.csv")
    return validate(source.read_csv(path))


def identity():
    return {name: source.sha256(DATA / name) for name in ("accepted_public.json", "label_overrides.csv", "dataset.json", "refit.csv", *[f"fold_{i}.csv" for i in range(4)])}


def build():
    if (DATA / "dataset.json").exists():
        raise FileExistsError("Versioned dataset already exists")
    folds = development_splits()
    refit = development_refit_rows()
    original_folds, _ = source.load_splits(config()[1])
    original = {r["sample_id"]: r for r in original_folds[0]}
    original.update({r["sample_id"]: r for r in folds[0]})
    old_targets = [r for r in original.values() if r["label"] != "other"]
    old_hashes = {source.sha256(YAMNET / r["relative_path"]): r["sample_id"] for r in original.values()}
    old_pcm = {}
    # Old test is never decoded. Only development target audio is decoded for
    # exact PCM duplicate detection against differently wrapped public files.
    for r in old_targets:
        if r["split"] == "test":
            continue
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(YAMNET / r["relative_path"]), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, check=True)
        import hashlib
        old_pcm[hashlib.sha256(result.stdout).hexdigest()] = r["sample_id"]
    records = source.read_json(DATA / "gongu_downloads.json") + source.read_json(DATA / "foreign_downloads.json")
    accepted, excluded, seen_hash, seen_pcm = [], [], dict(old_hashes), dict(old_pcm)
    old_recordings = {(r["source_dataset"], str(int(r["source_recording_id"])) if str(r["source_recording_id"]).isdigit() else r["source_recording_id"]) for r in original.values()}
    for record in records:
        key = record["source_dataset"], str(int(record["source_recording_id"]))
        duplicate = seen_hash.get(record["source_sha256"]) or seen_pcm.get(record["decoded_pcm_sha256"])
        if key in old_recordings or duplicate:
            excluded.append({"sample_id": record["sample_id"], "duplicate_of": duplicate or str(key), "reason": "existing source ID, exact bytes or decoded PCM"})
            continue
        if source.sha256(ROOT / record["relative_path"]) != record["source_sha256"] or record["license_id"] not in ("CC-BY-4.0", "CC0-1.0"):
            raise ValueError("New source changed or license not accepted")
        accepted.append(record)
        seen_hash[record["source_sha256"]] = record["sample_id"]
        seen_pcm[record["decoded_pcm_sha256"]] = record["sample_id"]
    if not accepted or not any(r["label"] == "fire_alarm" for r in accepted) or not any(r["origin"] == "KR_public_catalog" for r in accepted):
        raise ValueError("Required domestic collection and overseas fire-alarm supplementation incomplete")
    meta = {r["sample_id"]: r for r in source.read_csv(YAMNET / "data/metadata/esc50_manifest.csv")}
    overrides = []

    def revise(rows, record_overrides=False):
        revised = []
        for old in rows:
            row = dict(old, audio_root="yamnet", original_label=old["label"], label_basis="previously_approved_file_label", source_sha256="")
            if row["source_dataset"] == "ESC-50" and meta[row["sample_id"]]["confuser_tags"] == "siren":
                row.update(label="fire_alarm", label_id=0, sampler_pool="auxiliary_target", label_basis="user_explicit_siren_positive_policy")
                if record_overrides:
                    overrides.append({"sample_id": row["sample_id"], "split_group_id": row["split_group_id"], "split": row["split"], "original_label": "other", "new_label": "fire_alarm", "basis": "explicit_user_request", "domestic_weight": 4, "overseas_weight": 1})
            if row["sample_id"] == "bsb-1153":
                row["split_group_id"] = "bsb-smoke-detector-family"
            if row["sample_id"] == "bsb-2880":
                row["split_group_id"] = "bsb-nadja-house-doorbell-family"
            row["split_version"] = VERSION
            revised.append({k: row.get(k, "") for k in FIELDS})
        for record in accepted:
            row = dict(record, split="train", cv_fold="", split_version=VERSION, original_label=record["label"], label_basis=record["label_basis"])
            revised.append({k: row.get(k, "") for k in FIELDS})
        return validate(revised)

    revised_folds = [revise(rows, i == 0) for i, rows in enumerate(folds)]
    revised_refit = revise(refit)
    if Counter(r["split"] for r in overrides) != {"train": 24, "validation": 8}:
        raise ValueError("Expected exactly 24 training and8 validation sirens; do not silently change scope")
    test_ids = {r["sample_id"] for r in original.values() if r["split"] == "test"}
    test_groups = {r["split_group_id"] for r in original.values() if r["split"] == "test"}
    if any(test_ids.intersection(r["sample_id"] for r in rows) or test_groups.intersection(r["split_group_id"] for r in rows) for rows in [*revised_folds, revised_refit]):
        raise ValueError("Frozen historical holdout leaked into development")
    source.write_csv(DATA / "label_overrides.csv", overrides)
    source.write_json(DATA / "accepted_public.json", accepted)
    source.write_json(DATA / "excluded_duplicates.json", excluded)
    for i, rows in enumerate(revised_folds):
        source.write_csv(DATA / f"fold_{i}.csv", rows)
    source.write_csv(DATA / "refit.csv", revised_refit)
    counts = Counter(r["label"] for r in revised_refit if r["split"] == "train")
    groups = {label: len({r["split_group_id"] for r in revised_refit if r["split"] == "train" and r["label"] == label}) for label in source.LABELS}
    if not .8 <= counts["fire_alarm"] / counts["doorbell"] <= 1.25:
        raise ValueError("Collected positive-class balance is outside the declared range; review before training")
    result = {"version": VERSION, "accepted_new_public_files": len(accepted), "excluded_duplicate_files": len(excluded),
              "new_public_counts": dict(Counter(f"{r['origin']}:{r['label']}" for r in accepted)),
              "training_clips_by_class": dict(counts), "training_groups_by_class": groups,
              "fire_to_doorbell_training_clip_ratio": counts["fire_alarm"] / counts["doorbell"],
              "training_sirens_reclassified": 24, "validation_sirens_reclassified": 8,
              "new_public_role": "train_only", "new_public_country_basis": "Publication catalog, not verified physical device country",
              "new_public_human_listening_verified": False, "event_interval_annotations_used": False,
              "frozen_test_files_in_development": 0, "test_audio_decoded": 0, "test_membership_unchanged": True}
    write_json(DATA / "dataset.json", result)
    write_json(REPORT / "dataset_preparation.json", dict(result, dataset_identity=identity()))
    print("EXPANDED_DATASET_READY", result, flush=True)


class ExpandedFeatureStore(FeatureStore):
    def load(self, row):
        if row["split"] not in ("train", "validation"):
            raise ValueError("Fixed test unavailable to expanded feature pipeline")
        if row.get("audio_root", "yamnet") == "yamnet":
            return super().load(row)
        audio = audio_path(row)
        digest = source.sha256(audio)
        if digest != row["source_sha256"]:
            raise ValueError("Downloaded source hash differs from frozen inventory")
        directory = self.cache / source.fingerprint([digest, self.signature])
        array_path, meta_path = directory / "logmel.npy", directory / "metadata.json"
        if not meta_path.is_file():
            directory.mkdir(parents=True, exist_ok=False)
            from sound_baseline.features import decode_audio
            decode_audio(audio, directory / "audio.wav")
            with wave.open(str(directory / "audio.wav")) as handle:
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            values = np.stack([self.frontend(chunk) for _, chunk in windows(pcm)])
            np.save(array_path, values, allow_pickle=False)
            write_json(meta_path, {"source_sha256": digest, "frontend_signature": self.signature, "samples": len(pcm),
                                  "starts": window_starts(len(pcm)), "feature_sha256": source.sha256(array_path)})
        metadata = source.read_json(meta_path)
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if metadata["source_sha256"] != digest or metadata["frontend_signature"] != self.signature or source.sha256(array_path) != metadata["feature_sha256"] or values.shape != (len(metadata["starts"]), 98, 40, 1) or not np.isfinite(values).all():
            raise ValueError("Expanded features failed integrity/alignment checks")
        return values, metadata


def load_features(rows):
    store, features, provenance = ExpandedFeatureStore(), {}, []
    for row in rows:
        values, metadata = store.load(row)
        features[row["sample_id"]] = values
        provenance.append({"sample_id": row["sample_id"], "source_sha256": metadata["source_sha256"],
                           "feature_sha256": metadata["feature_sha256"], "window_count": len(values)})
    return features, provenance


if __name__ == "__main__":
    build()
