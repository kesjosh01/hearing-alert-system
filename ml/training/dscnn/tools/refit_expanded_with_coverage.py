"""Pre-test refit seed amendment based only on training-file coverage.

The original successful refit is preserved. No model scores are used to choose
the seed. Architecture, epochs, group probabilities and thresholds stay fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, source, write_json
from dscnn_baseline.weak_labels import FileBagSampler
from expansion.data import read_rows
from expansion.protocol import DATA, OPTIONS, REPORT
import expansion.train as training


def draw_counts(rows, epochs, seed):
    # Bag member RNG depends on group/member counts, never on feature values or
    # window counts. Synthetic one-window arrays reproduce the exact schedule.
    fake = {r["sample_id"]: np.zeros((1, 1), np.float32) for r in rows}
    sampler = FileBagSampler(rows, fake, config()[1], OPTIONS["domestic_weight"], seed)
    counts = Counter()
    for _ in range(epochs * OPTIONS["steps_per_epoch"]):
        for _, _, _, ids in sampler.draw_bags(OPTIONS["batch_size"], OPTIONS["microbatch_files"]):
            counts.update(ids)
    return {r["sample_id"]: counts[r["sample_id"]] for r in rows if r["split"] == "train"}


def prepare():
    target = REPORT / "refit_coverage_amendment.json"
    if target.exists() or (REPORT / "historical_holdout").exists():
        raise FileExistsError("Coverage amendment must precede any historical model comparison")
    base = source.read_json(REPORT / "frozen_model.json")
    directory = Path(base["refit"])
    rows = read_rows()
    actual = {r["sample_id"]: int(r["draws"]) for r in source.read_csv(directory / "sample_exposure.csv")}
    epochs = base["decision"]["refit_epochs"]
    if draw_counts(rows, epochs, OPTIONS["seed"]) != actual:
        raise ValueError("Metadata schedule does not reproduce initial refit; diagnose before changing anything")
    required = {r["sample_id"] for r in rows if r["split"] == "train" and r["label"] != "other"}
    missing = sorted(sid for sid in required if actual[sid] == 0)
    if not missing:
        raise ValueError("No target coverage gap; a refit amendment is unnecessary")
    for seed in range(OPTIONS["seed"] + 1, OPTIONS["seed"] + 65):
        counts = draw_counts(rows, epochs, seed)
        if all(counts[sid] > 0 for sid in required):
            break
    else:
        raise ValueError("No covered schedule found; do not alter epoch budget automatically")
    amended = {"kind": "pre_test_training_coverage_seed_amendment", "reason": "one added public target had zero direct draws in the initial five-epoch refit",
               "original_refit": str(directory), "original_frozen_sha256": source.sha256(REPORT / "frozen_model.json"),
               "original_seed": OPTIONS["seed"], "missing_target_ids": missing, "selected_refit_seed": seed,
               "seed_selection_rule": "First integer greater than original seed whose metadata-only exact sampling schedule covers every training target; no model performance used",
               "same_candidate": base["decision"]["candidate"], "same_epochs": epochs, "same_optimizer_steps": epochs * 30,
               "same_thresholds": base["decision"]["decision_policy"]["thresholds"], "same_domestic_group_weight": 4,
               "same_class_probabilities": [1/3, 1/3, 1/3], "new_test_evaluations_before_amendment": 0,
               "training_target_count": len(required), "wrapper_sha256": source.sha256(Path(__file__)),
               "output": str(ROOT / f"runs/dscnn_expanded_20261114_{base['decision']['candidate']}_coverage_final")}
    source.write_csv(REPORT / "refit_coverage_schedule.csv", [{"sample_id": sid, "expected_draws": n} for sid, n in sorted(counts.items())])
    write_json(target, amended)
    print("COVERAGE_REFIT_SEED_FROZEN", seed, "missing before", missing, "required targets", len(required), flush=True)


def run():
    amendment_path = REPORT / "refit_coverage_amendment.json"
    amendment = source.read_json(amendment_path)
    if amendment["wrapper_sha256"] != source.sha256(Path(__file__)) or (REPORT / "historical_holdout").exists():
        raise ValueError("Changed/non-pre-test coverage amendment")
    base = source.read_json(REPORT / "frozen_model.json")
    output = Path(amendment["output"])
    original_directory = training.run_directory
    # Explicit configuration layering within this process, recorded below.
    # Frozen CV code/options files are unchanged, and no safety gate is bypassed.
    training.OPTIONS = dict(OPTIONS, seed=amendment["selected_refit_seed"])
    training.run_directory = lambda role, fold=None, candidate=None: output if role == "student" and fold is None else original_directory(role, fold, candidate)
    training.run(SimpleNamespace(role="student", candidate=base["decision"]["candidate"], fold=0,
                                refit_epochs=base["decision"]["refit_epochs"], decision=REPORT / "decision.json"))
    actual = {r["sample_id"]: int(r["draws"]) for r in source.read_csv(output / "sample_exposure.csv")}
    expected = {r["sample_id"]: int(r["expected_draws"]) for r in source.read_csv(REPORT / "refit_coverage_schedule.csv")}
    if actual != expected:
        raise ValueError("Actual amended refit schedule differs from frozen metadata prediction")
    run_metadata = source.read_json(output / "run.json")
    run_metadata["refit_coverage_amendment"] = {"path": str(amendment_path), "sha256": source.sha256(amendment_path),
                                              "execution_wrapper_sha256": source.sha256(Path(__file__)),
                                              "changes": "final random seed and output directory only"}
    write_json(output / "run.json", run_metadata)
    amended_frozen = dict(base, refit=str(output), model_sha256=source.sha256(output / "classifier.keras"),
                         int8_sha256=source.sha256(output / "export/dscnn_int8.tflite"),
                         refit_coverage_amendment=run_metadata["refit_coverage_amendment"])
    write_json(REPORT / "frozen_model_coverage.json", amended_frozen)
    write_json(REPORT / "coverage_refit_status.json", {"exit_code": 0, "all100_training_targets_sampled": True,
               "metadata_predicted_and_actual_draw_counts_identical": True, "original_refit_preserved": True,
               "selected_seed": amendment["selected_refit_seed"], "test_used_to_select_seed": False,
               "refit": str(output), "additional_training_runs": 1})
    print("COVERAGE_REFIT_COMPLETE", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    prepare() if parser.parse_args().prepare else run()
