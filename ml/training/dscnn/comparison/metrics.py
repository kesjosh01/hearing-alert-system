"""Pure prediction/decision metrics; calibration never changes model weights."""
from collections import defaultdict
import numpy as np

LABELS=['fire_alarm','doorbell','other']
GRID=[0.,.3,.4,.5,.55,.6,.65,.7,.75,.8,.85,.9,.925,.95,.975,.99,.995,1.000001]


def pool(probabilities,fraction=1.):
    p=np.asarray(probabilities)
    if fraction<1:
        k=max(1,int(np.ceil(len(p)*fraction)))
        p=p[np.argsort(-(1-p[:,2]),kind='stable')[:k]]
    return p.mean(0)


def decide(p,thresholds):
    p=np.asarray(p);pred=p.argmax(1)
    for k in (0,1):pred[(pred==k)&(p[:,k]<thresholds[k])]=2
    return pred


def arrays(rows):
    p=np.asarray([[float(r['p_'+k]) for k in LABELS] for r in rows]);y=np.asarray([LABELS.index(r['label']) for r in rows])
    return p,y


def grouped(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['split_group_id']].append(r)
    out=[]
    for gid,rr in sorted(groups.items()):
        assert len({r['label'] for r in rr})==1
        p,_=arrays(rr);out.append(dict(rr[0],**{'p_'+k:float(v) for k,v in zip(LABELS,p.mean(0))}))
    return out


def assess(rows,thresholds):
    p,y=arrays(rows);pred=decide(p,thresholds);g=grouped(rows);gp,gy=arrays(g);gpred=decide(gp,thresholds)
    classes={}
    for k,label in enumerate(LABELS):
        s=int((y==k).sum());c=int(((y==k)&(pred==k)).sum())
        gs=int((gy==k).sum());gc=int(((gy==k)&(gpred==k)).sum())
        classes[label]=dict(clips=s,correct=c,recall=c/s if s else None,groups=gs,group_correct=gc,group_recall=gc/gs if gs else None)
    negative=np.asarray([r['label']=='other' and r.get('sampler_pool')!='domestic_doorlock' for r in rows])
    n=int(negative.sum());fp=int(((pred!=2)&negative).sum())
    locks=[i for i,r in enumerate(g) if r.get('sampler_pool')=='domestic_doorlock'];lockfp=int(sum(gpred[i]!=2 for i in locks))
    recalls=[classes[k]['group_recall'] for k in LABELS[:2]]
    macro=float(np.mean(recalls)) if all(x is not None for x in recalls) else None
    domestic=[r for r in g if r.get('sampler_pool')=='domestic_target']
    dr=[]
    if domestic:
        dp,dy=arrays(domestic);dpred=decide(dp,thresholds)
        for k in (0,1):
            if (dy==k).any():dr.append(float((dpred[dy==k]==k).mean()))
    loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-8)).mean())
    return dict(classes=classes,target_macro_group_recall=macro,domestic_target_macro_group_recall=float(np.mean(dr)) if len(dr)==2 else None,
                foreign_negative_clips=n,foreign_false_positives=fp,foreign_fpr=fp/n if n else None,
                domestic_doorlock_groups=len(locks),domestic_doorlock_fp=lockfp,
                guards_passed=bool(n and fp/n<=.05 and lockfp==0),loss=loss,thresholds=dict(zip(LABELS[:2],map(float,thresholds))),count=len(rows))


def rank(m):
    # Guard failures cannot defeat a feasible model solely by higher recall.
    return (int(m['guards_passed']),m['target_macro_group_recall'] or 0.,m['domestic_target_macro_group_recall'] or 0.,
            -(m['foreign_fpr'] or 0.),-m['loss'])


def calibrate(rows):
    p,y=arrays(rows);gr=grouped(rows);gp,gy=arrays(gr)
    pa=p.argmax(1);ga=gp.argmax(1)
    neg=np.asarray([r['label']=='other' and r.get('sampler_pool')!='domestic_doorlock' for r in rows]);n=int(neg.sum())
    locks=np.asarray([r.get('sampler_pool')=='domestic_doorlock' for r in gr])
    dom=np.asarray([r.get('sampler_pool')=='domestic_target' for r in gr])
    stats={}
    for k in (0,1):
        for t in GRID:
            active=(pa==k)&(p[:,k]>=t);gactive=(ga==k)&(gp[:,k]>=t)
            good=int((gactive&(gy==k)).sum());den=int((gy==k).sum())
            dd=int(((gy==k)&dom).sum());dg=int((gactive&(gy==k)&dom).sum())
            stats[k,t]=(int((active&neg).sum()),int((gactive&locks).sum()),good/den if den else 0.,dg/dd if dd else 0.)
    best=None;chosen=None
    for a in GRID:
        for b in GRID:
            af,al,ar,ad=stats[0,a];bf,bl,br,bd=stats[1,b]
            fp=af+bf;fpr=fp/n if n else 1.
            key=(int(n>0 and fpr<=.05 and al+bl==0),(ar+br)/2,(ad+bd)/2,-fpr,-(a+b))
            if best is None or key>best:best,chosen=key,(a,b)
    return chosen,assess(rows,chosen)


def apply(rows,thresholds):
    p,_=arrays(rows);pred=decide(p,thresholds)
    return [dict(r,predicted_label=LABELS[int(k)]) for r,k in zip(rows,pred)]
