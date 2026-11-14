"""Causal saved-audio notification replay. No microphone/board or onset claims."""
from collections import defaultdict
import numpy as np
from .metrics import GRID,LABELS
from dscnn_baseline.frontend import window_starts

BASE_POLICY=dict(vote_rule='single',release_ratio=1.,cooldown_s=0.)


class CausalDecisionState:
    """Bounded state usable for streaming callers; caller supplies audio time."""
    def __init__(self,thresholds,policy):
        self.thresholds=tuple(thresholds);self.policy=dict(policy)
        if self.policy['vote_rule'] not in ('single','two_of_three'):raise ValueError('Unknown vote rule')
        if not 0<self.policy['release_ratio']<=1 or self.policy['cooldown_s']<0:raise ValueError('Invalid hysteresis/cooldown')
        if len(self.thresholds)!=2 or not all(np.isfinite(t) and 0<=t<=1.000001 for t in self.thresholds):raise ValueError('Invalid thresholds')
        self.need,self.width=(1,1) if policy['vote_rule']=='single' else (2,3)
        self.reset()

    def reset(self):
        self.history=[[],[]];self.active=[False,False];self.last=[-np.inf,-np.inf];self.last_window_end=-np.inf

    def step(self,probability,window_end_s):
        p=np.asarray(probability);t=float(window_end_s)
        if p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(t):raise ValueError('Invalid prediction/time')
        if t<=self.last_window_end:raise ValueError('Window timestamps must increase; never replace stale timestamps')
        if np.isfinite(self.last_window_end) and t-self.last_window_end>.75:
            self.history=[[],[]];self.active=[False,False]
        self.last_window_end=t;arg=int(p.argmax());emitted=[]
        for k in (0,1):
            self.history[k].append(bool(arg==k and p[k]>=self.thresholds[k]));self.history[k]=self.history[k][-self.width:]
            now=sum(self.history[k])>=self.need
            if self.active[k] and self.policy['release_ratio']<1:now=bool(p[k]>=max(.05,self.thresholds[k]*self.policy['release_ratio']))
            if now and not self.active[k] and t-self.last[k]>=self.policy['cooldown_s']:
                emitted.append(dict(label=LABELS[k],window_end_s=t,probability=float(p[k])));self.last[k]=t
            self.active[k]=now
        return emitted


def frame_times(duration_s,window_count):
    # The actual frontend appends an end-aligned final window, which may be
    # less than .5 seconds after the preceding one. Do not fabricate its time.
    starts=window_starts(int(round(duration_s*16000)))
    if len(starts)!=window_count:raise ValueError('Replay/frontend window count differs')
    return [(s+16000)/16000 for s in starts]


def events(probabilities,thresholds,policy,end_times=None):
    outputs=[[],[]];state=CausalDecisionState(thresholds,policy)
    if end_times is not None and len(end_times)!=len(probabilities):raise ValueError('Frame/time count differs')
    for i,p in enumerate(probabilities):
        for e in state.step(p,end_times[i] if end_times is not None else 1.+i*.5):outputs[LABELS.index(e['label'])].append(e['window_end_s'])
    return outputs


def evaluate(rows,windows,durations,thresholds,policy):
    found=[];counts=[];first=[]
    for r in rows:
        p=windows[r['sample_id']];e=events(p,thresholds,policy,frame_times(durations[r['sample_id']],len(p)));found.append([bool(t) for t in e]);counts.append([len(t) for t in e]);first.append([t[0] if t else None for t in e])
    found=np.asarray(found);counts=np.asarray(counts);classes={};recalls=[];domestic=[]
    for k,label in enumerate(LABELS[:2]):
        g=defaultdict(list);dg=defaultdict(list)
        for i,r in enumerate(rows):
            if r['label']==label:
                g[r['split_group_id']].append(float(found[i,k]))
                if r['sampler_pool']=='domestic_target':dg[r['split_group_id']].append(float(found[i,k]))
        recall=float(np.mean([np.mean(v) for v in g.values()])) if g else 0.
        recalls.append(recall);domestic.append(float(np.mean([np.mean(v) for v in dg.values()])) if dg else 0.)
        support=sum(r['label']==label for r in rows)
        times=[first[i][k] for i,r in enumerate(rows) if r['label']==label and first[i][k] is not None]
        classes[label]=dict(clips=support,correctly_alerted_files=int(sum(found[i,k] for i,r in enumerate(rows) if r['label']==label)),source_groups=len(g),source_macro_file_recall=recall,
                            median_seconds_from_file_start=float(np.median(times)) if times else None,
                            latency_basis='Actual frontend window end (including end-aligned final window) from file start; short files right-zero-pad to1sec; acoustic onset not annotated')
    neg=np.asarray([r['label']=='other' and r['sampler_pool']!='domestic_doorlock' for r in rows]);locks=np.asarray([r['sampler_pool']=='domestic_doorlock' for r in rows])
    n=int(neg.sum());fp=int((found.any(1)&neg).sum());lockfp=int((found.any(1)&locks).sum());seconds=sum(durations[r['sample_id']] for i,r in enumerate(rows) if neg[i]);nc=int(counts[neg].sum())
    cross=int(sum(found[i,1-LABELS.index(r['label'])] for i,r in enumerate(rows) if r['label'] in LABELS[:2]))
    return dict(target_macro_group_recall=float(np.mean(recalls)),domestic_target_macro_group_recall=float(np.mean(domestic)),
                classes=classes,foreign_negative_clips=n,foreign_false_positive_files=fp,foreign_fpr=fp/n if n else None,
                domestic_doorlock_fp=lockfp,guards_passed=bool(n and fp/n<=.05 and lockfp==0),
                negative_audio_seconds=seconds,negative_notifications=nc,negative_notifications_per_hour=nc*3600/seconds if seconds else None,
                positive_files_with_wrong_class_alert=cross,policy=policy,thresholds=dict(zip(LABELS[:2],map(float,thresholds))),
                event_onset_ground_truth_available=False,per_event_recall_not_measured=True,recording_boundary_resets=True)


def rank(m):
    return (int(m['guards_passed']),m['target_macro_group_recall'],m['domestic_target_macro_group_recall'],
            -(m['foreign_fpr'] or 0.),-(m['negative_notifications_per_hour'] or 0.))


def calibrate(rows,windows,durations,policy):
    found={};num={};target={};domestic={}
    negatives=np.asarray([r['label']=='other' and r['sampler_pool']!='domestic_doorlock' for r in rows]);locks=np.asarray([r['sampler_pool']=='domestic_doorlock' for r in rows]);n=int(negatives.sum())
    for k in (0,1):
        for t in GRID:
            thr=[1.000001,1.000001];thr[k]=t
            ev=[events(windows[r['sample_id']],thr,policy,frame_times(durations[r['sample_id']],len(windows[r['sample_id']])))[k] for r in rows]
            flags=np.asarray([bool(e) for e in ev]);found[k,t]=flags;num[k,t]=np.asarray([len(e) for e in ev])
            group=defaultdict(list);dg=defaultdict(list)
            for i,r in enumerate(rows):
                if r['label']==LABELS[k]:
                    group[r['split_group_id']].append(flags[i])
                    if r['sampler_pool']=='domestic_target':dg[r['split_group_id']].append(flags[i])
            target[k,t]=float(np.mean([np.mean(v) for v in group.values()])) if group else 0.
            domestic[k,t]=float(np.mean([np.mean(v) for v in dg.values()])) if dg else 0.
    best=None;chosen=None
    for a in GRID:
        for b in GRID:
            flags=found[0,a]|found[1,b];fp=int((flags&negatives).sum());lockfp=int((flags&locks).sum());fpr=fp/n if n else 1.
            notifications=int((num[0,a][negatives]+num[1,b][negatives]).sum())
            key=(int(n>0 and fpr<=.05 and lockfp==0),(target[0,a]+target[1,b])/2,(domestic[0,a]+domestic[1,b])/2,-fpr,-notifications,-(a+b))
            if best is None or key>best:best,chosen=key,(a,b)
    return chosen,evaluate(rows,windows,durations,chosen,policy)


def added_delay(rows,windows,thresholds,policy,durations=None):
    deltas=[];lost=0;gained=0
    for r in rows:
        if r['label'] not in LABELS[:2]:continue
        k=LABELS.index(r['label']);p=windows[r['sample_id']];times=frame_times(durations[r['sample_id']],len(p)) if durations else None
        base=events(p,thresholds,BASE_POLICY,times)[k];new=events(p,thresholds,policy,times)[k]
        if base and new:deltas.append(new[0]-base[0])
        elif base and not new:lost+=1
        elif new and not base:gained+=1
    return dict(matched_positive_files=len(deltas),median_additional_seconds=float(np.median(deltas)) if deltas else None,
                max_additional_seconds=float(max(deltas)) if deltas else None,lost_positive_files=lost,gained_positive_files=gained,
                baseline='Single-window notifications at exactly the same thresholds; no true onset assumptions')
