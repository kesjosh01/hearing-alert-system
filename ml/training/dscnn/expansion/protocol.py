from __future__ import annotations

from dscnn_baseline.common import ROOT, source

VERSION = "public-target-siren-expansion-20261114-v1"
DATA = ROOT / "data/expansion_20261114"
REPORT = ROOT / "reports/expansion_20261114"
RUNS = ROOT / "runs"
EMBEDDINGS = ROOT / ".cache/distillation_aligned_v1"
BACKBONE = ROOT.parent / "yamnet/.cache/final_refit_official221_s42_v1/tensorflow/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
OPTIONS = {"seed": 42, "epochs": 30, "patience": 8, "batch_size": 30,
           "steps_per_epoch": 30, "microbatch_files": 6, "learning_rate": .001,
           "domestic_weight": 4.0, "foreign_target_weight": 1.0,
           "temperature": 2.0, "hard_label_weight": .5,
           "max_frequency_mask": 4, "max_time_mask": 10}
CANDIDATES = {"freq24": {"channels": 24, "depthwise_time_kernel": 3},
              "temporal24": {"channels": 24, "depthwise_time_kernel": 9}}


def code_identity():
    return {str(p.relative_to(ROOT)): source.sha256(p)
            for directory in (ROOT / "dscnn_baseline", ROOT / "improvement", ROOT / "expansion")
            for p in sorted(directory.glob("*.py"))}


def methods():
    return {"version": VERSION, "options": OPTIONS, "candidates": CANDIDATES,
            "supervision": "File labels: user-confirmed existing recordings, explicit catalog labels for newly collected public files, and user-directed broad siren=fire_alarm policy. No reviewed event timestamps.",
            "source_scope": "New public files train-only. Original domestic CV assignments and frozen historical holdout membership preserved.",
            "domestic_definition": "Previously confirmed Korean recordings plus Korean-publication sound catalog; physical device country of catalog recordings is unverified and recorded separately.",
            "siren_relabeling": "ESC-50 train24/validation8 -> fire_alarm in this version only; historical test metadata and audio never enter development.",
            "ratio": "Fire/doorbell/other training sampling probability 1/3 each, source-group sampling, domestic positive group weight4 vs overseas1; no simultaneous class/domain loss weights.",
            "collection_balance": "Korean licensed targets first. Fill missing confirmed fire-alarm sources with overseas smoke/fire/evacuation alarms. Limit related intercom variations, avoid ambiguous clock alarms/telephone rings; keep target clip totals within 0.8..1.25 ratio after deduplication where public supply permits.",
            "grouping": "Conservative device/creator series; new sources remain train-only and are not presented as independent evaluation samples.",
            "teacher": "Frozen official YAMNet backbone; retrain each fold's three-class head with corrected labels and matching training inventory. Never distill old negative siren logits as the only supervision.",
            "student": "CE of mean window softmax + aligned temperature2 teacher KL, coefficients0.5/0.5; same light SpecAugment for both candidates.",
            "architecture_hypothesis": "Increase local temporal receptive field before averaging from15 to51 log-Mel frames using9x3 depthwise kernels. Input/output and integer-compatible operators unchanged. Larger receptive field is a testable hypothesis, not a guaranteed improvement.",
            "selection": "Require <=5% foreign true-negative FPR and0 domestic doorlock FP in every fold. Select max mean domestic target recall, pooled macroF1, then smaller parameter count and lower FPR. Same thresholds grid as previous improvement, selected on development only. Keep all regressions visible.",
            "refit": "Rounded median of four selected epochs, fixed before final refit and int8 export. No test-driven reselection or calibration.",
            "historical_test": "Previously observed retrospective comparison only, after model/threshold hashes fixed. Exclude foreign sirens from negative FPR; do not invent their subtype ground truth.",
            "no_app_or_package_download_or_install": True, "no_microphone_or_board_execution": True,
            "references": ["https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D",
                           "https://www.isca-archive.org/interspeech_2020/xu20d_interspeech.pdf",
                           "https://arxiv.org/abs/1503.02531"]}
