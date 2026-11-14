"""Paired, source-matched random/pretrained CNN comparison."""
import json
import subprocess
import sys
from comparison.dataset import sha,write_json
from .paths import ROOT,REPORT


def main():
    assert json.loads((REPORT/'prefix_preflight.json').read_text())['passed']
    assert json.loads((REPORT/'round03_pilot_pretraining_pcm_audit.json').read_text())['passed']
    files=['pursuit/prefix_fit.py','pursuit/yamnet_prefix.py','pursuit/strong.py','pursuit/data.py','pursuit/metrics.py','pursuit/cost.py','pursuit/runner.py','pursuit/native.py','pursuit/native_frontend.cc',
           'comparison/model.py','comparison/sampling.py','comparison/protocol.py','comparison/dataset.py','dscnn_baseline/weak_labels.py','dscnn_baseline/common.py','dscnn_baseline/frontend.py']
    target=REPORT/'prefix_study_frozen.json'
    if not target.exists():write_json(target,dict(core_identity={k:sha(ROOT/k) for k in files},transfer_verification_sha256=sha(REPORT/'yamnet_transfer/verification.json'),
        weight_archive_sha256=sha(REPORT/'yamnet_transfer/frozen_variables.npz'),specs={tag:sha(REPORT/'specs'/(tag+'.json')) for tag in ['r03_prefix8_random','r03_prefix8_pretrained']},fresh_test_used=False))
    plan=json.loads(target.read_text());ledger=[]
    for tag in ['r03_prefix8_random','r03_prefix8_pretrained']:
        assert all(sha(ROOT/k)==v for k,v in plan['core_identity'].items()) and sha(REPORT/'yamnet_transfer/frozen_variables.npz')==plan['weight_archive_sha256']
        spec=REPORT/'specs'/(tag+'.json');assert sha(spec)==plan['specs'][tag];run=ROOT/'runs'/('pursuit_20261114_'+tag)
        if run.exists():
            status=json.loads((run/'status.json').read_text());assert status['complete'] and status['exit_code']==0,'Preserve incomplete experiment; inspect before restarting'
        else:
            log=REPORT/(tag+'_study.log')
            with log.open('x') as f:result=subprocess.run([sys.executable,'-B','-m','pursuit.runner','pursuit.prefix_fit',str(spec)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('First prefix experiment failure: '+str(log))
            status=json.loads((run/'status.json').read_text());assert status['complete']
        ledger.append(dict(stage=tag,run=str(run),status_sha256=sha(run/'status.json'),selected_metrics=status['selected_metrics']));write_json(REPORT/'prefix_study_ledger.json',ledger)
        print('PREFIX_SOURCE_MATCHED_MODEL_COMPLETE',tag,flush=True)
    write_json(REPORT/'prefix_study_status.json',dict(complete=True,models=2,overall_goal_complete=False,fresh_test_used=False,actual_product_targets_verified=False))


if __name__=='__main__':main()
