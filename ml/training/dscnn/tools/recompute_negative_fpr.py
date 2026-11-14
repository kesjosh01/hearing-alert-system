"""Apply the user's negative-evaluation scope to saved predictions, without ML.

Training inventories, labels, model selection, thresholds and original metrics
are immutable inputs. This writes an additive evaluation revision only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/improvement_20261114"
BASELINE = ROOT / "reports/real_training_20261114/study_calibrated_v2_resumed"
MANIFEST = ROOT.parent / "yamnet/data/metadata/esc50_manifest.csv"
LABELS = ("fire_alarm", "doorbell", "other")
# clock_alarm, church_bells and door_wood_knock are distinct source categories.
EXCLUDED_CATEGORIES = frozenset({"doorbell", "fire_alarm", "siren"})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty audit table: {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decide(row, thresholds):
    if thresholds is None:
        if row["predicted_label"] not in LABELS:
            raise ValueError("Unknown saved prediction")
        return row["predicted_label"]
    probabilities = [float(row[f"p_{label}"]) for label in LABELS]
    if not all(math.isfinite(p) and 0 <= p <= 1.001 for p in probabilities):
        raise ValueError("Invalid saved probability")
    index = max(range(3), key=probabilities.__getitem__)
    label = LABELS[index]
    return "other" if probabilities[index] < thresholds.get(label, 0.0) else label


def run(output):
    tracked = {}

    def track(path):
        path = Path(path).resolve()
        tracked[str(path)] = digest(path)
        return path

    manifest = read_csv(track(MANIFEST))
    metadata = {row["sample_id"]: row for row in manifest}
    if len(metadata) != len(manifest):
        raise ValueError("Duplicate source metadata IDs")
    categories = {}
    for row in manifest:
        match = re.search(r"Original category=([^;]+)", row["notes"])
        if match is None or row["source_dataset"] != "ESC-50":
            raise ValueError("Unrecognized source category metadata")
        categories[row["sample_id"]] = match.group(1)
    results, exclusions, negative_inventories = [], [], {}

    def score(key, path, role, thresholds=None, expected_fpr=None):
        rows = read_csv(track(path))
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate prediction IDs: {key}")
        negatives = [r for r in rows if r["target_domain"] != "ko_target" and r["label"] == "other"]
        if not negatives:
            raise ValueError(f"No foreign negatives: {key}")
        for row in negatives:
            meta = metadata.get(row["sample_id"])
            if meta is None or row["source_dataset"] != "ESC-50" or meta["label"] != "other" or meta["split"] != role or row["split"] != role:
                raise ValueError(f"Unmapped/mismatched negative metadata: {key}, {row['sample_id']}")
        ids = frozenset(row["sample_id"] for row in negatives)
        if role in negative_inventories and negative_inventories[role] != ids:
            raise ValueError(f"Unequal comparison populations for {role}")
        negative_inventories[role] = ids
        excluded = [r for r in negatives if categories[r["sample_id"]] in EXCLUDED_CATEGORIES]
        eligible = [r for r in negatives if categories[r["sample_id"]] not in EXCLUDED_CATEGORIES]
        fp = sum(decide(r, thresholds) != "other" for r in eligible)
        excluded_fp = sum(decide(r, thresholds) != "other" for r in excluded)
        raw_fp = fp + excluded_fp
        raw_fpr = raw_fp / len(negatives)
        if expected_fpr is not None and not math.isclose(raw_fpr, expected_fpr, abs_tol=1e-12):
            raise ValueError(f"Saved frozen-policy metric mismatch: {key}")
        results.append({"evaluation": key, "split": role, "prediction_file": str(Path(path).resolve()),
                        "original_negative_clips": len(negatives), "original_false_positives": raw_fp,
                        "original_fpr": raw_fpr, "excluded_clips": len(excluded),
                        "excluded_original_false_positives": excluded_fp,
                        "eligible_negative_clips": len(eligible), "false_positives": fp,
                        "fpr": fp / len(eligible), "eligible_fpr_le_5_percent": fp / len(eligible) <= 0.05,
                        "eligible_fp_categories": dict(sorted(Counter(categories[r["sample_id"]] for r in eligible if decide(r, thresholds) != "other").items()))})
        for row in sorted(excluded, key=lambda r: r["sample_id"]):
            exclusions.append({"evaluation": key, "sample_id": row["sample_id"],
                               "split_group_id": row["split_group_id"], "original_category": categories[row["sample_id"]],
                               "original_label": row["label"], "frozen_predicted_label": decide(row, thresholds),
                               "reason": "User excludes foreign target-like categories from negative FPR; no subtype ground truth inferred"})

    for name, directory in (("baseline", BASELINE), ("improved", REPORT)):
        for precision in ("float", "int8"):
            metrics = read_json(track(directory / f"historical_holdout/{precision}_metrics.json"))
            score(f"{name}_{precision}_historical", directory / f"historical_holdout/{precision}_clips.csv", "test",
                  expected_fpr=metrics["overseas_other_clip_fpr"])
    for summary_path in sorted(BASELINE.glob("w*_cv_summary.json")):
        summary = read_json(track(summary_path))
        for fold in summary["folds"]:
            threshold = fold["common_threshold"]
            score(f"original_w{summary['domestic_weight']}_fold{fold['fold']}",
                  BASELINE / f"w{summary['domestic_weight']}_fold{fold['fold']}_common_threshold_clips.csv", "validation",
                  {"fire_alarm": threshold, "doorbell": threshold}, fold["overseas_other_fpr"])
    for summary_path in sorted(REPORT.glob("*_cv_summary.json")):
        summary = read_json(track(summary_path))
        for fold in summary["folds"]:
            score(f"{summary['candidate']}_fold{fold['fold']}", Path(fold["run"]) / "validation_clips.csv", "validation",
                  summary["decision_policy"]["thresholds"], fold["overseas_other_fpr"])

    decision = read_json(track(REPORT / "decision.json"))
    final = ROOT / "runs/dscnn_improvement_20261114_final"
    final_metrics = read_json(track(final / "validation_metrics.json"))
    score("improved_refit_validation", final / "validation_clips.csv", "validation",
          decision["decision_policy"]["thresholds"], final_metrics["overseas_other_clip_fpr"])

    selected = read_json(track(REPORT / f"{decision['candidate']}_cv_summary.json"))
    epoch_evidence = []
    for fold in selected["folds"]:
        directory = Path(fold["run"])
        history = read_csv(track(directory / "history.csv"))
        status = read_json(track(directory / "status.json"))
        run_metadata = read_json(track(directory / "run.json"))
        chosen = next(r for r in history if int(r["epoch"]) == status["selected_epoch"])
        last = history[-1]
        if len(history) != status["epochs_completed"] or fold["selected_epoch"] != status["selected_epoch"]:
            raise ValueError("Epoch evidence disagrees with frozen selection")
        epoch_evidence.append({"fold": fold["fold"], "selected_epoch": status["selected_epoch"],
                               "epochs_completed": len(history), "epoch_cap": run_metadata["options"]["epochs"],
                               "patience": run_metadata["options"]["patience"],
                               "best_epoch_hard_label_loss": float(chosen["hard_label_loss"]),
                               "last_epoch_hard_label_loss": float(last["hard_label_loss"]),
                               "best_epoch_local_threshold_recall": float(chosen["domestic_target_recall"]),
                               "last_epoch_local_threshold_recall": float(last["domestic_target_recall"]),
                               "caveat": "Each epoch has its own development-calibrated thresholds; this is observational, not a fixed-threshold causal test."})
    oof = read_csv(track(REPORT / f"{decision['candidate']}_oof_domestic.csv"))
    if len({r["split_group_id"] for r in oof}) != len(oof):
        raise ValueError("Unexpected OOF grouping; do not count clips as independent groups")
    coverage = {label: {"source_groups": sum(r["label"] == label for r in oof),
                        "correct": sum(r["label"] == label and r["predicted_label"] == label for r in oof)} for label in LABELS}
    used = read_csv(track(final / "used_train_validation.csv"))
    siren_roles = Counter(r["split"] for r in used if categories.get(r["sample_id"]) == "siren")
    # Protect all frozen sources/models and original study/evaluation receipts.
    for module in (ROOT / "dscnn_baseline", ROOT / "improvement"):
        for path in sorted(module.glob("*.py")):
            track(path)
    for bundle in (ROOT.parents[1] / "models/dscnn_filelabels_20261114", ROOT.parents[1] / "models/dscnn_improved_20261114"):
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                track(path)
    for directory in (BASELINE / "historical_holdout", REPORT / "historical_holdout"):
        for path in sorted(directory.glob("*")):
            if path.is_file():
                track(path)
    track(REPORT / "frozen_model.json")
    track(REPORT / "FINAL_VERIFICATION.json")
    track(Path(__file__))
    output = Path(output).resolve()
    if not output.is_relative_to(REPORT.resolve()):
        raise ValueError("Revision must be an additive subdirectory of the improvement report")
    output.mkdir(exist_ok=False)
    policy = {"version": "foreign-negative-evaluation-user-v2-20261114", "created_at": datetime.now(timezone.utc).isoformat(),
              "authorization": "User requests excluding overseas doorbell/fire alarm/siren audio from foreign-other FPR.",
              "metric": "FP on eligible foreign negative clips / all eligible foreign negative clips",
              "excluded_source_categories": sorted(EXCLUDED_CATEGORIES), "observed_excluded_categories": ["siren"],
              "eligibility_uses_predictions": False, "all_category_members_excluded": True,
              "source_inventory": str(MANIFEST), "source_category_field": "notes: Original category=...",
              "scope": "Saved historical predictions and frozen development validation policies; no reselection",
              "domestic_targets_and_doorlock_metrics_unchanged": True, "raw_labels_and_original_metrics_unchanged": True,
              "siren_subtype_not_inferred": True, "alarm_clock_church_bell_door_knock_remain_negatives": True,
              "new_training_executions": 0, "new_model_inference_executions": 0, "threshold_recalibrations": 0}
    write_json(output / "policy.json", policy)
    write_json(output / "results.json", results)
    write_csv(output / "metrics.csv", [{**r, "eligible_fp_categories": json.dumps(r["eligible_fp_categories"], sort_keys=True)} for r in results])
    write_csv(output / "excluded_predictions.csv", exclusions)
    category_counts = Counter(categories.values())
    write_csv(output / "source_category_inventory.csv", [
        {"category": category, "total_clips": count, "excluded_from_negative_fpr": category in EXCLUDED_CATEGORIES,
         **{role: sum(categories[r["sample_id"]] == category and r["split"] == role for r in manifest)
            for role in ("train", "validation", "test")}} for category, count in sorted(category_counts.items())])
    write_json(output / "data_and_epoch_evidence.json", {"selected_candidate": decision["candidate"],
               "refit_epochs": decision["refit_epochs"], "domestic_OOF_groups": coverage,
               "selected_fold_epochs": epoch_evidence, "refit_siren_roles_still_labeled_other": dict(siren_roles),
               "raw_label_policy_review_required_before_future_training": True,
               "oof_errors": [{k: r[k] for k in ("sample_id", "label", "predicted_label", "source_dataset")} for r in oof if r["label"] != r["predicted_label"]]})
    if any(digest(path) != expected for path, expected in tracked.items()):
        raise ValueError("Input mutated during metrics-only correction")
    output_hashes = {str(path): digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "verification.json", {"kind": "saved_prediction_evaluation_revision", "passed": True,
               "tracked_input_sha256": tracked, "output_sha256": output_hashes, "evaluations": len(results),
               "input_files_unchanged": True, "prediction_dependent_exclusions": False,
               "all_original_fprs_reproduced_before_exclusion": True,
               "model_reselection": False, "training_executions": 0, "model_inference_executions": 0})
    print("NEGATIVE_FPR_REVISION_COMPLETE", output, "evaluations", len(results))
    for result in results[:4]:
        print(result["evaluation"], f"{result['false_positives']}/{result['eligible_negative_clips']}", f"{100 * result['fpr']:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "evaluation_scope_v2")
    run(parser.parse_args().output)
