"""Known-signal acoustic-condition fitting; not independent validation."""
from collections import defaultdict
import numpy as np
from .metrics import calibrate as original_calibrate,rank as original_rank,scores


def calibrate(rows,window_prob,policy=((1,1),(1,1)),fraction=.5):
    thresholds,metric=original_calibrate(rows,window_prob,policy,fraction)
    conditions={}
    for condition in ['original','quiet','room_snr20','room_snr10']:
        by_label={}
        for k,label in enumerate(['fire_alarm','doorbell']):
            rr=[r for r in rows if r['label']==label and r.get('probe_condition','original')==condition]
            assert rr
            hits=[bool((scores(window_prob[r['sample_id']],policy)[:,k]>=thresholds[k]).any()) for r in rr]
            by_label[label]=dict(files=len(rr),detected_files=sum(hits),file_recall=float(np.mean(hits)))
        conditions[condition]=by_label
    metric.update(training_seen_condition_results=conditions,
        minimum_condition_fire_recall=min(c['fire_alarm']['file_recall'] for c in conditions.values()),
        minimum_condition_doorbell_recall=min(c['doorbell']['file_recall'] for c in conditions.values()),
        distinct_original_target_recordings=43,transformed_probe_files=172,
        independent_target_validation_recordings=0,
        measure='TRAINING-SEEN original KR files and training waveform variants; source means average four conditions')
    return thresholds,metric


def rank(metric):
    fire=metric['minimum_condition_fire_recall'];bell=metric['minimum_condition_doorbell_recall']
    return (int(metric['total_cross_class_or_negative_alert_files']==0),
        min(fire/.999,bell/.95),(fire+bell)/2,*original_rank(metric)[1:])
