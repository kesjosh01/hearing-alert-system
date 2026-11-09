"""Manifest validation, group sampling and metrics, without TensorFlow imports."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"fire_alarm": 0, "doorbell": 1, "other": 2}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write a CSV without rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def settings(config_path=None):
    baseline = read_json(config_path or ROOT / "config/yamnet_baseline_v1.json")
    protocol = read_json(ROOT / baseline["protocol_path"])
    if protocol["labels"] != LABELS:
        raise ValueError("Expected fire_alarm=0, doorbell=1, other=2")
    return baseline, protocol


def split_path(protocol, fold=0, refit=False):
    name = "refit.csv" if refit else f"fold_{fold}.csv"
    return ROOT / "data/splits" / protocol["version"] / name


def validate_rows(rows, checked_paths=None):
    checked_paths = set() if checked_paths is None else checked_paths
    ids, group_roles, source_roles = set(), defaultdict(set), defaultdict(set)
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        ids.add(sample_id)
        if row["label"] not in LABELS or int(row["label_id"]) != LABELS[row["label"]]:
            raise ValueError(f"Invalid label mapping: {sample_id}")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {sample_id}")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Out-of-project audio: {sample_id}")
        # Avoid resolving the same 2,046 OneDrive paths five times per command.
        if row["relative_path"] not in checked_paths:
            if not (ROOT / relative).is_file():
                raise ValueError(f"Missing audio: {sample_id}")
            checked_paths.add(row["relative_path"])
        group_roles[row["split_group_id"]].add(row["split"])
        source_roles[(row["source_dataset"], row["source_recording_id"])].add(row["split"])
    if any(len(roles) != 1 for roles in [*group_roles.values(), *source_roles.values()]):
        raise ValueError("Original recording group crosses train/validation/test")
    return rows


def load_splits(protocol):
    checked_paths = set()
    folds = [validate_rows(read_csv(split_path(protocol, i)), checked_paths) for i in range(protocol["split"]["development_folds"])]
    refit = validate_rows(read_csv(split_path(protocol, refit=True)), checked_paths)
    lock = read_json(split_path(protocol).parent / "holdout_lock.json")
    reference_ids = {row["sample_id"] for row in folds[0]}
    test_ids = {row["sample_id"] for row in folds[0] if row["split"] == "test"}
    validated = defaultdict(int)
    for fold_number, rows in enumerate(folds):
        if {row["sample_id"] for row in rows} != reference_ids:
            raise ValueError("Fold inventories differ")
        if {row["sample_id"] for row in rows if row["split"] == "test"} != test_ids:
            raise ValueError("Test membership differs across folds")
        for row in rows:
            if row["split_version"] != protocol["version"]:
                raise ValueError("Protocol version mismatch")
            assignment = lock["assignments"].get(row["sample_id"])
            if assignment is not None:
                expected = "test" if assignment["test"] else "validation" if assignment["cv_fold"] == fold_number else "train"
                if row["split"] != expected:
                    raise ValueError("Domestic split does not match frozen assignment")
                if row["split"] == "validation":
                    validated[row["sample_id"]] += 1
    development_ids = {key for key, assignment in lock["assignments"].items() if not assignment["test"]}
    if set(validated) != development_ids or any(count != 1 for count in validated.values()):
        raise ValueError("Domestic development OOF coverage is not exactly once")
    if {row["sample_id"] for row in refit} != reference_ids or {row["sample_id"] for row in refit if row["split"] == "test"} != test_ids:
        raise ValueError("Refit inventory/test differs")
    original = {row["sample_id"]: row for row in folds[0]}
    for row in refit:
        expected = "train" if row["sample_id"] in development_ids else original[row["sample_id"]]["split"]
        if row["split"] != expected:
            raise ValueError("Refit includes an unexpected source")
    return folds, refit


class GroupSampler:
    """Class/pool probability -> original group -> uniformly chosen native patch."""

    def __init__(self, rows, features, protocol, domestic_weight=4, seed=42):
        if domestic_weight <= 0:
            raise ValueError("Domestic weight must be positive")
        self.rng = np.random.default_rng(seed)
        self.rows = {row["sample_id"]: row for row in rows if row["split"] == "train"}
        self.groups = defaultdict(list)
        for row in self.rows.values():
            self.groups[row["split_group_id"]].append(row)
        self.ids = sorted(self.groups)
        self.probabilities = np.zeros(len(self.ids), dtype=np.float64)
        self.references = {}
        descriptors = []
        for group_id in self.ids:
            members = self.groups[group_id]
            labels = {row["label"] for row in members}
            if len(labels) != 1:
                raise ValueError("Group spans multiple classes")
            label = members[0]["label"]
            pool = members[0]["sampler_pool"]
            if any(row["sampler_pool"] == "overseas_confuser" for row in members):
                pool = "overseas_confuser"
            refs = [(row["sample_id"], i) for row in members for i in range(len(features[row["sample_id"]]))]
            if not refs:
                raise ValueError(f"No usable patches in {group_id}")
            self.references[group_id] = refs
            descriptors.append((label, pool))
        sampling = protocol["sampling"]
        for label, class_probability in sampling["class_probabilities"].items():
            indices = [i for i, (value, _) in enumerate(descriptors) if value == label]
            if not indices:
                raise ValueError(f"No training groups for class {label}")
            if label != "other":
                weights = np.array([domestic_weight if descriptors[i][1] == "domestic_target" else sampling["target_class_auxiliary_group_weight"] for i in indices], dtype=np.float64)
                self.probabilities[indices] = class_probability * weights / weights.sum()
            else:
                for pool, fraction in sampling["other_pool_probabilities"].items():
                    selected = [i for i in indices if descriptors[i][1] == pool]
                    if not selected:
                        raise ValueError(f"Empty other training pool: {pool}")
                    self.probabilities[selected] = class_probability * fraction / len(selected)
        if not np.isclose(self.probabilities.sum(), 1):
            raise ValueError("Sampler probabilities do not sum to 1")
        self.features = features

    def draw(self, batch_size):
        indices = self.rng.choice(len(self.ids), size=batch_size, p=self.probabilities)
        x, y, source_ids = [], [], []
        for index in indices:
            refs = self.references[self.ids[index]]
            sample_id, patch = refs[self.rng.integers(len(refs))]
            x.append(self.features[sample_id][patch])
            y.append(int(self.rows[sample_id]["label_id"]))
            source_ids.append(sample_id)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), source_ids

    def table(self):
        return [{"split_group_id": group, "group_probability": float(probability), "patch_count": len(self.references[group])}
                for group, probability in zip(self.ids, self.probabilities)]


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def classification(y, probabilities):
    if not len(y):
        return {"count": 0, "accuracy": None, "macro_f1": None, "loss": None, "classes": {}, "confusion_matrix": [[0] * 3 for _ in range(3)]}
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    classes = {}
    for label, index in LABELS.items():
        tp = int(matrix[index, index])
        support, positives = int(matrix[index].sum()), int(matrix[:, index].sum())
        precision = tp / positives if positives else 0.0
        recall = tp / support if support else 0.0
        classes[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          "support": support, "correct": tp, "predicted": positives}
    return {"count": len(y), "accuracy": float(np.mean(y == predicted)), "macro_f1": float(np.mean([item["f1"] for item in classes.values()])),
            "loss": float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-9, 1)).mean()),
            "classes": classes, "confusion_matrix": matrix.tolist()}


def prediction_row(row, probabilities, patches):
    result = {key: row[key] for key in ("sample_id", "split_group_id", "label", "label_id", "target_domain", "source_dataset", "split")}
    result.update({f"p_{label}": float(probabilities[index]) for label, index in LABELS.items()})
    result.update(predicted_label=list(LABELS)[int(np.argmax(probabilities))], patch_count=int(patches))
    return result


def aggregate_groups(predictions):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["split_group_id"]].append(row)
    output = []
    for group_id, members in grouped.items():
        if len({row["label"] for row in members}) != 1:
            raise ValueError("Evaluation source group spans multiple labels")
        values = np.array([[float(row[f"p_{label}"]) for label in LABELS] for row in members])
        item = prediction_row(members[0], values.mean(axis=0), sum(int(row["patch_count"]) for row in members))
        item["sample_id"] = ";".join(row["sample_id"] for row in members)
        output.append(item)
    return output


def report(predictions):
    def scores(rows):
        return classification([int(row["label_id"]) for row in rows], [[float(row[f"p_{label}"]) for label in LABELS] for row in rows])
    groups = aggregate_groups(predictions)
    domestic = [row for row in groups if row["target_domain"] == "ko_target"]
    overseas_other = [row for row in predictions if row["target_domain"] != "ko_target" and row["label"] == "other"]
    doorlocks = [row for row in domestic if row["label"] == "other"]
    domestic_report = scores(domestic)
    target_classes = [domestic_report["classes"].get(label) for label in ("fire_alarm", "doorbell")]
    target_recall = float(np.mean([item["recall"] for item in target_classes])) if all(item and item["support"] for item in target_classes) else None
    return {"clips": scores(predictions), "groups": scores(groups), "domestic_groups": domestic_report,
            "domestic_target_macro_recall": target_recall,
            "overseas_other_clip_count": len(overseas_other),
            "overseas_other_clip_false_positives": sum(row["predicted_label"] != "other" for row in overseas_other),
            "overseas_other_clip_fpr": sum(row["predicted_label"] != "other" for row in overseas_other) / len(overseas_other) if overseas_other else None,
            "domestic_doorlock_group_count": len(doorlocks),
            "domestic_doorlock_false_positive_groups": sum(row["predicted_label"] != "other" for row in doorlocks)}


def checkpoint_rank(metrics, guards):
    fpr = metrics["overseas_other_clip_fpr"]
    recall = metrics["domestic_target_macro_recall"]
    if fpr is None or recall is None or not metrics["domestic_doorlock_group_count"]:
        raise ValueError("Development validation must contain both domestic targets, domestic other, and overseas other")
    excess = max(0.0, fpr - guards["max_overseas_other_clip_fpr"])
    extra_doorlocks = max(0, metrics["domestic_doorlock_false_positive_groups"] - guards["max_domestic_doorlock_false_positive_groups"])
    feasible = excess <= 1e-12 and extra_doorlocks == 0
    # Once feasible, primary recall decides; before that, reduce guard violations.
    return (int(feasible), -excess - extra_doorlocks, recall, metrics["domestic_groups"]["macro_f1"], -metrics["groups"]["loss"])
