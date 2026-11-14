"""Causal event proxies with class-specific cross-trigger negatives.

File labels cannot establish true event onset or per-event product reliability.
The strict zero-observed-error operating point is a development proxy, never
evidence for one false notification per device-month in actual homes.
"""
from collections import defaultdict
import numpy as np
from comparison.metrics import LABELS,pool


def scores(prob,policy):
    p=np.asarray(prob);arg=p.argmax(1);out=np.zeros((len(p),2),np.float32)
    for k in (0,1):
        need,width=policy[k];history=[]
        for i in range(len(p)):
            history.append(float(p[i,k]) if arg[i]==k else 0.);history=history[-width:]
            out[i,k]=sorted(history,reverse=True)[need-1] if len(history)>=need else 0.
    return out


def calibrate(rows,window_prob,policy=((1,1),(1,1)),fraction=.5):
    maxima=np.stack([scores(window_prob[r['sample_id']],policy).max(0) for r in rows]);y=np.array([LABELS.index(r['label']) for r in rows])
    thresholds=[]
    for k in (0,1):
        # Wrong-class target alerts count as false alerts as well as other.
        negatives=maxima[y!=k,k];assert len(negatives)
        thresholds.append(float(np.nextafter(float(negatives.max()),np.inf)) if negatives.max()>0 else 1e-8)
    flags=(maxima>=thresholds)&(maxima>0);recall=[];classes={}
    for k,label in enumerate(LABELS[:2]):
        grouped=defaultdict(list)
        for i,r in enumerate(rows):
            if y[i]==k:grouped[r['split_group_id']].append(float(flags[i,k]))
        rate=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else 0.;recall.append(rate)
        classes[label]=dict(files=int((y==k).sum()),detected_files=int(flags[y==k,k].sum()),source_groups=len(grouped),source_mean_file_recall=rate,
          wrong_class_or_negative_alert_files=int(flags[y!=k,k].sum()))
    losses=[];margins=[]
    for i,r in enumerate(rows):
        p=pool(window_prob[r['sample_id']],fraction);losses.append(-np.log(max(float(p[y[i]]),1e-8)))
        if y[i] in (0,1):margins.append(float(maxima[i,y[i]]-thresholds[y[i]]))
    metric=dict(classes=classes,fire_recall=recall[0],doorbell_recall=recall[1],target_macro_source_file_recall=float(np.mean(recall)),
         any_other_false_alert_files=int(flags[y==2].any(1).sum()),other_files=int((y==2).sum()),
         total_cross_class_or_negative_alert_files=sum(classes[k]['wrong_class_or_negative_alert_files'] for k in LABELS[:2]),
         thresholds=dict(zip(LABELS[:2],thresholds)),policy=[list(x) for x in policy],loss=float(np.mean(losses)),
         positive_margin_mean=float(np.mean(margins)) if margins else -1.,
         event_onset_or_device_month_reliability_proven=False,measure='Weak-label file detection proxy; not true per-event recall')
    return thresholds,metric


def rank(m):
    return (int(m['total_cross_class_or_negative_alert_files']==0),min(m['fire_recall']/.999,m['doorbell_recall']/.95),
            m['target_macro_source_file_recall'],m['positive_margin_mean'],-m['loss'])
