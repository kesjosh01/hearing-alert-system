"""Matched max-MIL loss experiment, followed by the explicit seen-KR branch."""
import json
import subprocess
import sys
from comparison.dataset import write_json,sha
from .paths import ROOT,REPORT


def main():
    assert json.loads((REPORT/'presence_mil_gpu_gate.json').read_text())['passed']
    stages=[('r02_max_ce_control','pursuit.fit'),('r02_presence_mil','pursuit.presence_fit'),('domestic_seen_presence','pursuit.presence_domestic_fit')]
    target=REPORT/'presence_study_frozen.json';dependencies=['pursuit/presence_fit.py','pursuit/presence_domestic_fit.py','pursuit/presence_mil.py','pursuit/fit.py','pursuit/data.py','pursuit/metrics.py','comparison/model.py','comparison/sampling.py','comparison/protocol.py']
    if not target.exists():write_json(target,dict(core_identity={s:sha(ROOT/s) for s in dependencies},specs={s:sha(REPORT/'specs'/(s+'.json')) for s,_ in stages},fresh_test_access=False))
    plan=json.loads(target.read_text());ledger=[]
    for tag,module in stages:
        assert all(sha(ROOT/p)==v for p,v in plan['core_identity'].items());spec=REPORT/'specs'/(tag+'.json');assert sha(spec)==plan['specs'][tag]
        run=ROOT/'runs'/('pursuit_20261114_'+tag)
        if run.exists():
            status=json.loads((run/'status.json').read_text());assert status['complete'] and status['exit_code']==0,'Preserve partial optimizer execution; inspect before resume'
        else:
            log=REPORT/(tag+'_study.log')
            with log.open('x') as f:result=subprocess.run([sys.executable,'-B','-m','pursuit.runner',module,str(spec)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('First presence-loss study failure: '+str(log))
            status=json.loads((run/'status.json').read_text());assert status['complete']
        ledger.append(dict(stage=tag,status_sha256=sha(run/'status.json'),target_results_training_seen=status.get('target_results_training_seen',False),selected_metrics=status['selected_metrics']))
        write_json(REPORT/'presence_study_ledger.json',ledger);print('PRESENCE_LOSS_STUDY_STAGE_COMPLETE',tag,flush=True)
    write_json(REPORT/'presence_study_status.json',dict(complete=True,models=len(ledger),fresh_test_used=False,overall_goal_complete=False,product_targets_verified=False))


if __name__=='__main__':main()
