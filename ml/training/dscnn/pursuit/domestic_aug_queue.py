"""Run the preregistered KR robustness fit after the current GPU study ends."""
import json
import subprocess
import sys
import time
from comparison.dataset import write_json,sha
from .paths import ROOT,REPORT


def main():
    plan=json.loads((REPORT/'domestic_aug_plan.json').read_text());spec=REPORT/'specs/domestic_seen_aug_presence.json'
    target=REPORT/'domestic_aug_queue_status.json'
    if target.exists():raise FileExistsError('Preserve queue ownership; inspect existing execution before any resume')
    write_json(target,dict(state='waiting_for_enriched_study',gpu_execution_started=False,automatic_retry=False))
    started=time.monotonic()
    while True:
        prior=REPORT/'enriched_study_status.json'
        if prior.exists():
            assert json.loads(prior.read_text())['complete'];break
        for tag in ['r04_freq24','r04_prefix40_pretrained','r04_prefix64_pretrained']:
            p=ROOT/'runs'/('pursuit_20261114_'+tag)/'status.json'
            if p.exists():
                try:j=json.loads(p.read_text())
                except json.JSONDecodeError:continue
                if j.get('exit_code') not in (None,0):raise RuntimeError('Preceding GPU stage failed; preserve first failure and stop queued work')
        if time.monotonic()-started>7200:raise TimeoutError('No study completion within queued execution window; inspect, never auto-retry')
        time.sleep(5)
    assert sha(spec)==plan['spec_sha256'] and all(sha(ROOT/p)==v for p,v in plan['code_identity'].items())
    name=json.loads(spec.read_text())['name'];assert not (ROOT/'runs'/name).exists()
    write_json(target,dict(state='launching_owned_gpu_runner',gpu_execution_started=True,automatic_retry=False))
    result=subprocess.run([sys.executable,'-B','-m','pursuit.runner','pursuit.domestic_aug_fit',str(spec)],cwd=ROOT)
    if result.returncode:
        write_json(target,dict(state='first_failure',exit_code=result.returncode,automatic_retry=False))
        raise RuntimeError('First domestic augmentation stage failure; inspect preserved log')
    status=json.loads((ROOT/'runs'/name/'status.json').read_text());assert status['complete'] and status['target_results_training_seen']
    write_json(target,dict(state='complete',run=name,run_status_sha256=sha(ROOT/'runs'/name/'status.json'),
        independent_new_targets=0,actual_product_targets_verified=False))
    print('QUEUED_DOMESTIC_ACOUSTIC_ROBUSTNESS_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
