"""CPU-only selection analysis from frozen logits; no new audio/model runs."""
import argparse
import json
import numpy as np
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import DATA,REPORT
from . import operating as op

ORIGINAL_SCORE=op.score


def score(z,mode,temperature,policy):
    if mode!='independent_probability':return ORIGINAL_SCORE(z,mode,temperature,policy)
    p=op.softmax(z,temperature);out=np.zeros((len(p),2))
    for k in (0,1):
        need,width=policy[k];history=[]
        for i,value in enumerate(p[:,k]):
            history.append(value);history=history[-width:];out[i,k]=sorted(history,reverse=True)[need-1] if len(history)>=need else 0
    return out


def main(dataset,version=''):
    assert not version or __import__('pathlib').Path(version).name==version
    suffix='_'+version if version else ''
    out=REPORT/'operating'/dataset;target=out/('independent_readout_comparison'+suffix+'.json')
    if target.exists():raise FileExistsError('Preserve completed readout analysis')
    op.score=score;d=DATA/dataset;freeze=json.loads((d/'freeze.json').read_text());rows=[r for r in read_csv(d/'development.csv') if r['role']=='selection'];assert all(sha(d/k)==v for k,v in freeze['files'].items())
    ledger=[];records=[]
    for path in sorted(out.glob('pursuit_*.json')):
        base=json.loads(path.read_text());assert base['complete'] and not base['fresh_test_used'] and base['dataset_freeze_sha256']==sha(d/'freeze.json')
        p=path.with_name(path.stem+'_selection_logits.npz');assert sha(p)==base['logit_cache_sha256']
        with np.load(p,allow_pickle=False) as h:logits={k:h[k] for k in h.files}
        assert set(logits)=={r['sample_id'] for r in rows};results=list(base['results'])
        for temperature in [1.,2.,4.,8.]:
            for fire in [(1,1),(2,3)]:
                for bell in [(1,1),(2,3)]:results.append(op.measure(rows,logits,'independent_probability',temperature,(fire,bell)))
        best=max(results,key=op.key);record=dict(model=base['model'],model_sha256=base['model_sha256'],selection_only=True,best=best,all_results=results,previous_analysis_sha256=sha(path));records.append(record)
        ledger.append(dict(model=base['model'],mode=best['mode'],temperature=best['temperature'],policy=best['policy'],fire_recall=best['fire_recall'],doorbell_recall=best['doorbell_recall'],macro_recall=best['macro_recall']))
        print('INDEPENDENT_READOUT_SELECTION',ledger[-1],flush=True)
    write_json(target,dict(complete=True,models=records,dataset=dataset,fresh_test_or_calibration_used=False,true_event_or_product_reliability_verified=False))
    write_csv(out/('readout_summary_all'+suffix+'.csv'),ledger)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--version',default='');a=p.parse_args();main(a.dataset,a.version)
