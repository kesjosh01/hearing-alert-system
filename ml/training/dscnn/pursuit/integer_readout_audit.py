"""Canonical delivered-output scores, without probability renormalization."""
import argparse
import json
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from . import operating as op

ORIGINAL=op.score


def score(values,mode,temperature,policy):
    if mode!='delivered_probability':return ORIGINAL(values,mode,temperature,policy)
    p=np.asarray(values,dtype=np.float64);arg=p.argmax(1);a=np.zeros((len(p),2))
    for k in (0,1):
        need,width=policy[k];history=[]
        for i,value in enumerate(np.where(arg==k,p[:,k],0)):
            history.append(value);history=history[-width:];a[i,k]=sorted(history,reverse=True)[need-1] if len(history)>=need else 0
    return a


def main(name):
    out=REPORT/'int8_readout_trials'/name;target=out/'canonical_delivered_output_analysis.json'
    if target.exists():raise FileExistsError('Preserve completed audit')
    run=json.loads((ROOT/'runs'/name/'run.json').read_text());dataset=run['spec']['dataset'];rows=[r for r in read_csv(DATA/dataset/'development.csv') if r['role']=='selection'];op.score=score;results=[]
    for kind in ['probability','logits']:
        d=out/kind;meta=json.loads((d/'model_metadata.json').read_text());base=json.loads((d/'selection_analysis.json').read_text());p=d/'selection_outputs.npz';assert sha(p)==base['outputs_sha256']
        with np.load(p,allow_pickle=False) as h:values={k:h[k] for k in h.files}
        assert set(values)=={r['sample_id'] for r in rows}
        if kind=='logits':
            scale=meta['output']['scale'];zero=meta['output']['zero_point']
            values={k:np.rint(v/scale+zero).astype(np.int16) for k,v in values.items()};assert all(v.min()>=-128 and v.max()<=127 for v in values.values())
            mode='margin'
        else:mode='delivered_probability'
        rr=[]
        for fire in [(1,1),(2,3)]:
            for bell in [(1,1),(2,3)]:rr.append(op.measure(rows,values,mode=mode,policy=(fire,bell)))
        best=max(rr,key=op.key)
        if kind=='logits':best['integer_margin_thresholds']=[int(np.floor(x))+1 for x in best['thresholds']]
        results.append(dict(kind=kind,model_sha256=meta['model_sha256'],best=best,results=rr,exact_delivered_output_used=True,output_renormalization=False))
        print('CANONICAL_INT8_READOUT',kind,best,flush=True)
    write_json(target,dict(complete=True,results=results,previous_prototype_status_sha256=sha(out/'status.json'),
      previous_probability_analysis_superseded_reason='Prototype log-probability reuse normalized rounded outputs; this canonical audit uses delivered probabilities exactly.',
      integer_logit_margins_use_int16_differences=True,selection_only=True,model_or_input_data_changed=False,fresh_test_used=False,board_or_product_reliability_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();main(a.name)
