"""Execute the registered round in sequence, preserving partial failures."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from comparison.dataset import write_json,sha
from .paths import ROOT,REPORT


def main():
    p=argparse.ArgumentParser();p.add_argument('round',choices=['round01']);a=p.parse_args()
    assert json.loads((REPORT/'gpu_preflight.json').read_text())['passed']
    assert json.loads((REPORT/'round01_features.json').read_text())['complete']
    frozen=json.loads((REPORT/'round01_training_core_freeze.json').read_text());assert all(sha(ROOT/k)==v for k,v in frozen.items())
    ledger=[]
    for tag in ['r01_freq24','r01_freq48','r01_freq24_firevote']:
        name='pursuit_20261114_'+tag;run=ROOT/'runs'/name
        if run.exists():
            status=json.loads((run/'status.json').read_text());assert status['complete'] and status['exit_code']==0,'Inspect incomplete owner; never restart on an observation timeout'
        else:
            spec=REPORT/'specs'/(tag+'.json');log=REPORT/(name+'.log')
            with log.open('x') as f:
                result=subprocess.run([sys.executable,'-B','-m','pursuit.runner','pursuit.train',str(spec)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if result.returncode:raise RuntimeError(f'First training failure: {log}; inspect before any resume')
            status=json.loads((run/'status.json').read_text())
        assert status['complete'] and sha(run/'classifier.keras')==status['model_sha256']
        ledger.append(dict(name=name,run=str(run),status=status,only_selection_proxies=True))
        write_json(REPORT/'round01_ledger.json',ledger)
    write_json(REPORT/'round01_status.json',dict(complete=True,training_runs=len(ledger),fresh_test_used=False,
                  overall_goal_complete=False,product_targets_verified=False,remaining_public_collection=True,
                  next='Broader data rounds, full YAMNet reference, gain/noise/room augmentation, domestic memorization branch and independent product validation'))
    print('PURSUIT_ROUND01_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
