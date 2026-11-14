from __future__ import annotations

from pathlib import Path

from dscnn_baseline.common import ROOT, YAMNET, config, source
from dscnn_baseline.data import development_identity, development_splits, development_refit_rows

REPORT = ROOT / "reports/improvement_20261114"
TEACHER_CACHE = ROOT / ".cache/distillation_aligned_v1"
BACKBONE = YAMNET / ".cache/final_refit_official221_s42_v1/tensorflow/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
CANDIDATES = {
    "kd24": {"channels": 24, "keep_frequency": False, "specaugment": False},
    "kd24_freq": {"channels": 24, "keep_frequency": True, "specaugment": False},
    "kd24_freq_aug": {"channels": 24, "keep_frequency": True, "specaugment": True},
    "kd48_freq_aug": {"channels": 48, "keep_frequency": True, "specaugment": True},
}
GRID = [0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
OPTIONS = {"seed": 42, "epochs": 30, "patience": 8, "batch_size": 30,
           "steps_per_epoch": 30, "microbatch_files": 6, "learning_rate": 0.001,
           "domestic_weight": 1, "temperature": 2.0, "hard_label_weight": 0.5,
           "max_frequency_mask": 4, "max_time_mask": 10}


def code_identity():
    return {str(p.relative_to(ROOT)): source.sha256(p) for directory in (ROOT / "dscnn_baseline", ROOT / "improvement")
            for p in sorted(directory.glob("*.py"))}


def teacher_directory(fold=None):
    base = YAMNET / "runs/yamnet"
    return base / "ubuntu_gpu_refit_w1_s42" if fold is None else base / f"ubuntu_gpu_cv_w1_s42/fold_{fold}"


def verify_teacher(fold=None):
    """A CV teacher's TRAIN groups must be a subset of this student's TRAIN."""
    directory = teacher_directory(fold)
    run = source.read_json(directory / "run.json")
    result = source.read_json(directory / "result.json")
    expected_kind = "refit" if fold is None else "development"
    if run["kind"] != expected_kind or result["kind"] != expected_kind or (fold is not None and run["fold"] != fold):
        raise ValueError("Wrong teacher role/fold")
    if run["labels"] != config()[0]["labels"]:
        raise ValueError("Teacher class mapping differs")
    teacher_rows = source.read_csv(directory / "used_train_validation.csv")
    student_rows = development_refit_rows() if fold is None else development_splits()[fold]
    groups = lambda rows, role: {r["split_group_id"] for r in rows if r["split"] == role}
    teacher_train = groups(teacher_rows, "train")
    if not teacher_train <= groups(student_rows, "train") or teacher_train & groups(student_rows, "validation"):
        raise ValueError("Teacher training leaks this student's validation groups")
    if any(r["split"] == "test" for r in teacher_rows):
        raise ValueError("Teacher fitting inventory contains fixed test")
    return {"directory": str(directory), "model_sha256": source.sha256(directory / "classifier.keras"),
            "training_inventory_sha256": source.sha256(directory / "used_train_validation.csv"),
            "fold": fold, "training_groups": len(teacher_train), "validation_group_leakage": False}


def plan():
    return {"kind": "dscnn_development_improvement_plan", "options": OPTIONS,
            "candidates": CANDIDATES, "threshold_grid_per_target": GRID,
            "development_identity": development_identity(), "code_identity": code_identity(),
            "teachers": [verify_teacher(i) for i in range(4)], "refit_teacher": verify_teacher(),
            "backbone_pb_sha256": source.sha256(BACKBONE / "saved_model.pb"),
            "student_frontend": config()[0]["frontend"], "event_intervals_reviewed": False,
            "teacher_alignment": "For every student's 16000-sample window, YAMNet sees the same first 15600 PCM16 samples (one native patch). No peak normalization.",
            "loss": "0.5 * file-label CE(mean window softmax) + 0.5 * T^2 * mean_file(mean_window(KL(teacher_T || student_T))); T=2",
            "teacher_scope": "Frozen pretrained backbone + matching fold's trained three-class head; teacher outputs are soft supervision, never human event annotations",
            "baseline_control": "Recalibrate preserved 1/2/4-weight baseline CV predictions with the same per-target threshold grid; no baseline retraining",
            "selection": "All four unchanged guards must pass. Maximize mean domestic target recall, then pooled domestic macro-F1 and minimum target precision, then prefer smaller student and lower foreign FPR. Report every fold and regressions. Keep a baseline if no candidate improves development results.",
            "refit": "Freeze candidate, weight, per-target thresholds and rounded median best epoch before refit; no validation epoch reselection",
            "historical_test": "Already observed. Not used for improvement selection. No repeated test calls during search; any final comparison is retrospective and cannot support a new independent-test claim.",
            "references": ["https://arxiv.org/abs/1503.02531", "https://keras.io/examples/vision/knowledge_distillation/",
                           "https://www.tensorflow.org/tutorials/audio/transfer_learning_audio", "https://arxiv.org/abs/1904.08779"],
            "installation_executed": False, "board_execution": False}
