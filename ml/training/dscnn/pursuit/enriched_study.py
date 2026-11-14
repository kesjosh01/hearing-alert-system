"""Sequential enriched-cohort CNN study, stopping on the first stage failure."""
import json
import subprocess
import sys
from comparison.dataset import write_json,sha
from .paths import ROOT,DATA,REPORT

STAGES=[('r04_freq24','pursuit.fit'),('r04_prefix40_pretrained','pursuit.prefix_fit'),
        ('r04_prefix64_pretrained','pursuit.prefix64_fit')]


def plan():
    for tag,module in STAGES:
        p=REPORT/'specs'/(tag+'.json')
        if p.exists():raise FileExistsError('Preserve preregistered study')
        small=tag=='r04_freq24'
        spec=dict(name='pursuit_20261114_'+tag,dataset='round04_enriched',seed=42,
            minimum_negative_coverage=1,policy=[[1,1],[1,1]],
            options=dict(pooling_fraction=.5,learning_rate=.001 if small else .0003,
                epochs=60 if small else 80,patience=12 if small else 15),
            channels=24 if small else 32,
            supervision='Full weak-label bags' if small else 'Published bell-positive windows in TRAIN only; whole recordings for selection',
            module=module,product_targets_verified=False)
        if not small:spec.update(prefix_layers=8,pretrained=True,
            frontend='official 96x64 magnitude-log-Mel' if '64' in tag else 'portable 98x40 power-log-Mel')
        write_json(p,spec)
    write_json(REPORT/'enriched_study_plan.json',dict(stages=STAGES,
        new_membership_chosen_before_new_round_scores=True,old_model_same_cohort_analysis_separate=True,
        fresh_test_access=False,actual_product_criteria_verified=False,
        interpretation='Compare models on identical selection recordings; data, inputs and supervision differ as specified.'))
    print('ENRICHED_CNN_STUDY_PREREGISTERED',flush=True)


def run():
    for file in ['prefix64_preflight.json','round04_enriched_pretraining_pcm_audit.json']:
        assert json.loads((REPORT/file).read_text())['passed']
    assert json.loads((REPORT/'round04_enriched_frontend64_features.json').read_text())['complete']
    dependencies=['pursuit/enriched_study.py','pursuit/fit.py','pursuit/prefix_fit.py','pursuit/prefix64_fit.py',
        'pursuit/generic_update.py','pursuit/yamnet_frontend.py','pursuit/yamnet_prefix.py',
        'pursuit/strong.py','pursuit/data.py','pursuit/native.py','pursuit/metrics.py','pursuit/cost.py',
        'comparison/model.py','comparison/sampling.py','comparison/protocol.py']
    target=REPORT/'enriched_study_frozen.json'
    if not target.exists():write_json(target,dict(core_identity={k:sha(ROOT/k) for k in dependencies},
        specs={t:sha(REPORT/'specs'/(t+'.json')) for t,_ in STAGES},
        cohort_sha256=sha(DATA/'round04_enriched/freeze.json'),
        weight_archive_sha256=sha(REPORT/'yamnet_transfer/frozen_variables.npz'),fresh_test_access=False))
    frozen=json.loads(target.read_text());ledger=[]
    for tag,module in STAGES:
        assert all(sha(ROOT/k)==v for k,v in frozen['core_identity'].items())
        assert sha(DATA/'round04_enriched/freeze.json')==frozen['cohort_sha256']
        assert sha(REPORT/'yamnet_transfer/frozen_variables.npz')==frozen['weight_archive_sha256']
        spec=REPORT/'specs'/(tag+'.json');assert sha(spec)==frozen['specs'][tag]
        directory=ROOT/'runs'/('pursuit_20261114_'+tag)
        if directory.exists():
            j=json.loads((directory/'status.json').read_text());assert j['complete'] and j['exit_code']==0,'Preserve partial run and inspect first error'
        else:
            log=REPORT/(tag+'_study.log')
            with log.open('x') as f:r=subprocess.run([sys.executable,'-B','-m','pursuit.runner',module,str(spec)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError('First enriched-study failure; inspect '+str(log))
            j=json.loads((directory/'status.json').read_text());assert j['complete']
        ledger.append(dict(stage=tag,status_sha256=sha(directory/'status.json'),selected_metrics=j['selected_metrics']))
        write_json(REPORT/'enriched_study_ledger.json',ledger);print('ENRICHED_STUDY_STAGE_COMPLETE',tag,flush=True)
    write_json(REPORT/'enriched_study_status.json',dict(complete=True,models=len(ledger),
        fresh_test_used=False,overall_goal_complete=False,actual_product_criteria_verified=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run']);a=p.parse_args();plan() if a.mode=='plan' else run()
