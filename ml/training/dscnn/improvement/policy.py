"""Per-class target rejection, calibrated using development predictions only."""
from __future__ import annotations

import itertools
import numpy as np

from dscnn_baseline.common import source
from dscnn_baseline import decisions
from .protocol import GRID


def decide(probabilities, thresholds):
    p = np.asarray(probabilities, dtype=np.float64)
    threshold = np.array([thresholds["fire_alarm"], thresholds["doorbell"], 0.0])
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold > 1)):
        raise ValueError("Invalid probability/threshold array")
    y = p.argmax(1)
    return np.where(p[np.arange(len(p)), y] < threshold[y], 2, y)


def apply(rows, thresholds):
    if not rows:
        return []
    p = [[float(r[f"p_{label}"]) for label in source.LABELS] for r in rows]
    y = decide(p, thresholds)
    return [dict(r, predicted_label=list(source.LABELS)[int(v)]) for r, v in zip(rows, y)]


def report(rows, thresholds):
    clips = apply(rows, thresholds)
    groups = apply(source.aggregate_groups(rows), thresholds)
    domestic = [r for r in groups if r["target_domain"] == "ko_target"]
    overseas = [r for r in clips if r["target_domain"] != "ko_target" and r["label"] == "other"]
    locks = [r for r in domestic if r["label"] == "other"]
    dm = decisions.scores(domestic)
    return {"clips": decisions.scores(clips), "groups": decisions.scores(groups), "domestic_groups": dm,
            "domestic_target_macro_recall": float(np.mean([dm["classes"][s]["recall"] for s in ("fire_alarm", "doorbell")])) if domestic else None,
            "overseas_other_clip_count": len(overseas),
            "overseas_other_clip_false_positives": sum(r["predicted_label"] != "other" for r in overseas),
            "overseas_other_clip_fpr": sum(r["predicted_label"] != "other" for r in overseas) / len(overseas) if overseas else None,
            "domestic_doorlock_group_count": len(locks),
            "domestic_doorlock_false_positive_groups": sum(r["predicted_label"] != "other" for r in locks),
            "decision_policy": {"kind": "per_target_argmax_confidence_rejection", "thresholds": thresholds}}


def matrix_scores(y, p, thresholds):
    predicted = decide(p, thresholds)
    cm = np.bincount(3 * np.asarray(y, dtype=int) + predicted, minlength=9).reshape(3, 3)
    support, total = cm.sum(1), cm.sum(0)
    diagonal = cm.diagonal()
    precision = np.divide(diagonal, total, out=np.zeros(3), where=total > 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {"macro_f1": float(f1.mean()), "loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()),
            "classes": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "support": int(support[i])}
                        for label, i in source.LABELS.items()}}, predicted


def prepare(rows):
    grouped = source.aggregate_groups(rows)
    domestic = [r for r in grouped if r["target_domain"] == "ko_target"]
    overseas = [r for r in rows if r["target_domain"] != "ko_target" and r["label"] == "other"]
    make = lambda records: (np.array([int(r["label_id"]) for r in records]),
                            np.array([[float(r[f"p_{label}"]) for label in source.LABELS] for r in records]))
    return {"groups": make(grouped), "domestic": make(domestic), "overseas": make(overseas)}


def fast_report(table, thresholds):
    dm, predicted = matrix_scores(*table["domestic"], thresholds)
    gm, _ = matrix_scores(*table["groups"], thresholds)
    foreign = decide(table["overseas"][1], thresholds)
    y = table["domestic"][0]
    return {"groups": gm, "domestic_groups": dm,
            "domestic_target_macro_recall": (dm["classes"]["fire_alarm"]["recall"] + dm["classes"]["doorbell"]["recall"]) / 2,
            "overseas_other_clip_fpr": float(np.mean(foreign != 2)),
            "domestic_doorlock_group_count": int(np.sum(y == 2)),
            "domestic_doorlock_false_positive_groups": int(np.sum((y == 2) & (predicted != 2)))}


def calibrate(folds, guards):
    if len(folds) not in (1, 4) or any(r["split"] != "validation" for rows in folds for r in rows):
        raise ValueError("Calibration requires development validation rows only")
    oof = [r for rows in folds for r in rows if r["target_domain"] == "ko_target"]
    if len({r["sample_id"] for r in oof}) != len(oof):
        raise ValueError("Domestic OOF duplicates")
    tables = [prepare(rows) for rows in folds]
    pooled = prepare(oof)
    best = None
    for fire, bell in itertools.product(GRID, repeat=2):
        thresholds = {"fire_alarm": fire, "doorbell": bell}
        metrics = [fast_report(t, thresholds) for t in tables]
        ranks = [source.checkpoint_rank(m, guards) for m in metrics]
        domestic, _ = matrix_scores(*pooled["domestic"], thresholds)
        rank = (int(all(r[0] for r in ranks)), sum(r[1] for r in ranks),
                float(np.mean([m["domestic_target_macro_recall"] for m in metrics])), domestic["macro_f1"],
                -float(np.mean([m["overseas_other_clip_fpr"] for m in metrics])), -fire - bell, -fire)
        if best is None or rank > best[0]:
            best = rank, thresholds
    rank, thresholds = best
    return thresholds, [report(rows, thresholds) for rows in folds], report(oof, thresholds), rank
