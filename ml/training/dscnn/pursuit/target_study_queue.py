"""Prepare the target-only data expansion on CPU, then serialize its GPU fit."""
import json
import math
import shutil
import subprocess
import sys
import time
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import CACHE,audio_path
from .yamnet_frontend import CACHE as CACHE64
from .waveform_identity import identity


def until(check,source_failure_log=None):
    started=time.monotonic()
    while not check():
        if source_failure_log and source_failure_log.exists() and 'Traceback (most recent call last)' in source_failure_log.read_text():
            raise RuntimeError('Preceding source collection failed; no automatic retry or alternative route')
        if time.monotonic()-started>7200:raise TimeoutError('Preceding stage did not finish within the queued window')
        time.sleep(5)


def cpu(module,*args):
    executable=ROOT/'.venv/bin/python' if module=='pursuit.yamnet_frontend' else sys.executable
    r=subprocess.run([str(executable),'-B','-m',module,*args],cwd=ROOT)
    if r.returncode:raise RuntimeError('First target-data CPU stage failure: '+module)


def main():
    target=REPORT/'target_study_queue_status.json'
    if target.exists():raise FileExistsError('Preserve queued execution and inspect before any resume')
    plan=json.loads((REPORT/'target_study_queue_plan.json').read_text());spec=REPORT/'specs/r05_prefix64_targets.json'
    assert sha(spec)==plan['spec_sha256'] and all(sha(ROOT/k)==v for k,v in plan['code_identity'].items())
    write_json(target,dict(state='waiting_for_registered_source_shards',gpu_started=False,automatic_retry=False))
    try:
        until(lambda: all((REPORT/'audioset_strong_intake'/f'{i:05d}.json').exists() for i in range(600)),REPORT/'audioset_strong_intake_batch2.log')
        write_json(target,dict(state='freezing_training_only_target_expansion',gpu_started=False,automatic_retry=False))
        cpu('pursuit.target_round','freeze');dataset='round05_train_targets';rows=read_csv(DATA/dataset/'development.csv');required=0
        for row in rows:
            m=identity(row);digest=m['source_sha256'];n=m['sample_count'];count=1+max(0,math.ceil((n-16000)/8000))
            if not (CACHE/digest/'metadata.json').exists():required+=n*2+count*98*40*4+4096
            if not (CACHE64/digest/'metadata.json').exists():required+=count*96*64*4+4096
        free=shutil.disk_usage(DATA).free
        write_json(REPORT/'round05_feature_capacity_plan.json',dict(estimated_new_bytes=required,free_bytes=free,
            safety_factor=2,reserve_bytes=12*2**30,originals_and_existing_caches_preserved=True))
        if free<required*2+12*2**30:raise OSError('Insufficient new-feature capacity with protected-data reserve; no automatic deletion')
        write_json(target,dict(state='preparing_cpu_features_and_identity_audit',gpu_started=False,automatic_retry=False))
        cpu('pursuit.target_round','prepare');cpu('pursuit.yamnet_frontend','prepare',dataset);cpu('pursuit.waveform_identity',dataset)
        audit=json.loads((REPORT/(dataset+'_waveform_identity_audit.json')).read_text())
        assert not audit['cross_role_exact_pcm_clusters']
        write_json(target,dict(state='waiting_for_prior_gpu_studies',gpu_started=False,automatic_retry=False))
        def prior_ready():
            p=REPORT/'domestic_aug_queue_status.json'
            if not p.exists():return False
            try:j=json.loads(p.read_text())
            except json.JSONDecodeError:return False
            if j['state']=='first_failure':raise RuntimeError('Prior GPU stage failed; stop downstream work')
            return j['state']=='complete'
        until(prior_ready)
        assert sha(spec)==plan['spec_sha256'] and all(sha(ROOT/k)==v for k,v in plan['code_identity'].items())
        name=json.loads(spec.read_text())['name'];assert not (ROOT/'runs'/name).exists()
        write_json(target,dict(state='launching_owned_gpu_runner',gpu_started=True,automatic_retry=False))
        r=subprocess.run([sys.executable,'-B','-m','pursuit.runner','pursuit.prefix64_fit',str(spec)],cwd=ROOT)
        if r.returncode:raise RuntimeError('First expanded-target model fitting failure; inspect preserved run')
        status=json.loads((ROOT/'runs'/name/'status.json').read_text());assert status['complete']
        write_json(target,dict(state='complete',run=name,status_sha256=sha(ROOT/'runs'/name/'status.json'),
            fresh_test_used=False,overall_goal_complete=False,actual_product_targets_verified=False))
        print('TARGET_ONLY_EXPANSION_MODEL_COMPLETE',flush=True)
    except Exception as e:
        write_json(target,dict(state='first_failure',error=f'{type(e).__name__}: {e}',automatic_retry=False,originals_and_partials_preserved=True));raise


if __name__=='__main__':main()
