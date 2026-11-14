"""Development-only target rejection threshold; keep raw probabilities unchanged."""
from __future__ import annotations

import math
import numpy as np

from .common import source


def apply(rows, threshold=0.0):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be finite and within [0,1]")
    result=[]
    labels=list(source.LABELS)
    for row in rows:
        values=np.asarray([float(row[f"p_{label}"]) for label in labels])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite decision probabilities")
        index=int(values.argmax())
        if index != source.LABELS["other"] and values[index] < threshold:
            index=source.LABELS["other"]
        result.append(dict(row,predicted_label=labels[index],decision_threshold=threshold))
    return result


def aggregate(rows,threshold=0.0):
    # Apply rejection after equal-file probability averaging at the source level.
    return apply(source.aggregate_groups(rows),threshold)


def scores(rows):
    if not rows:
        return source.classification([],[])
    y=np.asarray([int(row["label_id"]) for row in rows])
    predicted=np.asarray([source.LABELS[row["predicted_label"]] for row in rows])
    probabilities=np.asarray([[float(row[f"p_{label}"]) for label in source.LABELS] for row in rows])
    metrics=source.classification(y,np.eye(3)[predicted])
    # Classification decisions use rejection; loss still measures raw model probabilities.
    metrics["loss"]=float(-np.log(np.clip(probabilities[np.arange(len(y)),y],1e-9,1)).mean())
    return metrics


def report(rows,threshold=0.0):
    clips=apply(rows,threshold)
    groups=aggregate(rows,threshold)
    domestic=[r for r in groups if r["target_domain"]=="ko_target"]
    overseas=[r for r in clips if r["target_domain"]!="ko_target" and r["label"]=="other"]
    locks=[r for r in domestic if r["label"]=="other"]
    domestic_metrics=scores(domestic)
    targets=[domestic_metrics["classes"].get(label) for label in ("fire_alarm","doorbell")]
    recall=float(np.mean([r["recall"] for r in targets])) if all(r and r["support"] for r in targets) else None
    return {"clips":scores(clips),"groups":scores(groups),"domestic_groups":domestic_metrics,
            "domestic_target_macro_recall":recall,
            "overseas_other_clip_count":len(overseas),
            "overseas_other_clip_false_positives":sum(r["predicted_label"]!="other" for r in overseas),
            "overseas_other_clip_fpr":sum(r["predicted_label"]!="other" for r in overseas)/len(overseas) if overseas else None,
            "domestic_doorlock_group_count":len(locks),
            "domestic_doorlock_false_positive_groups":sum(r["predicted_label"]!="other" for r in locks),
            "decision_policy":{"kind":"argmax_with_target_confidence_rejection","threshold":threshold},
            "loss_note":"Cross entropy uses raw probabilities; confusion matrices and FPR use the declared rejection policy."}


def calibrate(rows,guards,grid):
    if any(r["split"]!="validation" for r in rows):
        raise ValueError("Threshold selection may use development validation only")
    choices=[]
    for threshold in grid:
        metrics=report(rows,threshold)
        rank=source.checkpoint_rank(metrics,guards)+(-metrics["overseas_other_clip_fpr"],-threshold)
        choices.append((rank,threshold,metrics))
    if not choices:raise ValueError("Threshold grid is empty")
    rank,threshold,metrics=max(choices,key=lambda item:item[0])
    return threshold,metrics,rank


def calibrate_joint(folds,guards,grid):
    if len(folds)!=4 or any(r["split"]!="validation" for rows in folds for r in rows):
        raise ValueError("Joint threshold selection requires exactly four development validation predictions")
    oof=[r for rows in folds for r in rows if r["target_domain"]=="ko_target"]
    if len({r["sample_id"] for r in oof})!=len(oof):
        raise ValueError("Domestic OOF samples repeat")
    choices=[]
    for threshold in grid:
        fold_metrics=[report(rows,threshold) for rows in folds]
        ranks=[source.checkpoint_rank(m,guards) for m in fold_metrics]
        pooled=report(oof,threshold)
        recall=float(np.mean([m["domestic_target_macro_recall"] for m in fold_metrics]))
        fpr=float(np.mean([m["overseas_other_clip_fpr"] for m in fold_metrics]))
        rank=(int(all(r[0] for r in ranks)),sum(r[1] for r in ranks),recall,
              pooled["domestic_groups"]["macro_f1"],-fpr,-threshold)
        choices.append((rank,threshold,fold_metrics,pooled))
    if not choices:raise ValueError("Threshold grid is empty")
    _,threshold,metrics,pooled=max(choices,key=lambda item:item[0])
    return threshold,metrics,pooled
