"""Sequential approved experiments; first failure stops the queue intact."""
import json
import subprocess
import sys
from pathlib import Path
from comparison.dataset import sha,write_json
from .paths import ROOT,REPORT

STAGES=[
 ('r03_weak','pursuit.fit',['reports/pursuit_20261114/specs/r03_weak.json'],'runs/pursuit_20261114_r03_weak/status.json'),
 ('r03_strong','pursuit.strong_fit',['reports/pursuit_20261114/specs/r03_strong.json'],'runs/pursuit_20261114_r03_strong/status.json'),
 ('round02_embeddings','pursuit.embeddings',['round02'],'reports/pursuit_20261114/round02_yamnet_embeddings.json'),
 ('r02_yamnet_sparse','pursuit.head_v2',['pursuit_20261114_r02_yamnet_sparse','--dataset','round02','--fraction','.01'],'runs/pursuit_20261114_r02_yamnet_sparse/status.json'),
 ('r02_yamnet_warm','pursuit.head_v2',['pursuit_20261114_r02_yamnet_warm','--dataset','round02','--fraction','.01','--initialize','pursuit_20261114_r01_yamnet_sparse'],'runs/pursuit_20261114_r02_yamnet_warm/status.json'),
 ('domestic_seen_sparse','pursuit.domestic_fit',['reports/pursuit_20261114/specs/domestic_seen_sparse.json'],'runs/pursuit_20261114_domestic_seen_sparse/status.json'),
]


def main():
    target=REPORT/'next_study_plan.json'
    if not target.exists():
        write_json(target,dict(stages=STAGES,core_identity={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'pursuit').glob('*.py')},
            fresh_test_access=False,hardware_execution=False,overall_goal_complete=False))
    plan=json.loads(target.read_text());assert plan['stages']==[list(s[:2])+[s[2],s[3]] for s in STAGES]
    ledger=[]
    for tag,module,args,status_path in STAGES:
        assert all(sha(ROOT/k)==v for k,v in plan['core_identity'].items()),'Frozen experiment code changed; inspect before any continuation'
        status=ROOT/status_path
        if status.exists():
            result=json.loads(status.read_text());assert result['complete'] and result.get('exit_code',0)==0,'Inspect partial stage; no automatic retraining'
        else:
            log=REPORT/(tag+'_next.log')
            with log.open('x') as f:process=subprocess.run([sys.executable,'-B','-m','pursuit.runner',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError(f'First stage failure: {tag}; preserve {log} and inspect')
            result=json.loads(status.read_text());assert result['complete']
        ledger.append(dict(stage=tag,status_path=status_path,status_sha256=sha(status),complete=True))
        write_json(REPORT/'next_study_ledger.json',ledger);print('PURSUIT_SEQUENTIAL_STAGE_COMPLETE',tag,flush=True)
    write_json(REPORT/'next_study_status.json',dict(complete=True,stages=len(ledger),fresh_test_used=False,actual_product_target_verified=False,overall_goal_complete=False))


if __name__=='__main__':main()
