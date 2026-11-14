"""Sequential one-factor experiments, resumable only from verified completions."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from .protocol import ROOT,REPORT,RUNS,BASE,SEEDS
from .dataset import write_json,write_csv,sha


def score(m):
    return (int(m['guards_passed']),m['target_macro_group_recall'] or 0.,m['domestic_target_macro_group_recall'] or 0.,-(m['foreign_fpr'] or 0.),-m['loss'])


def child(module,args,log):
    cmd=[sys.executable,'-B','-m','comparison.runner',module,*map(str,args)]
    with log.open('x') as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'First failing stage: {module}; see {log}; no automatic retry')


def execute(name,options,teacher,seed=42,role='student'):
    run=RUNS/name
    if run.exists():
        status=json.loads((run/'status.json').read_text())
        if status.get('complete') is not True or status.get('exit_code')!=0:
            raise RuntimeError(f'Existing non-complete run must be inspected, never restarted automatically: {run}')
        meta=json.loads((run/'run.json').read_text())
        assert meta['options']==options and meta['spec'].get('seed',42)==seed and meta['spec'].get('role','student')==role,'Completed run has a different specification'
        return run,status
    spec=dict(name=name,options=options,teacher_run=str(teacher) if teacher else None,seed=seed,role=role)
    path=REPORT/'specs'/(name+'.json');write_json(path,spec)
    child('comparison.train',[path],REPORT/(name+'.log'))
    status=json.loads((run/'status.json').read_text())
    if not status.get('complete'):raise RuntimeError('Child did not complete')
    return run,status


def calibrate(run):
    dest=REPORT/'calibration'/run.name/'results.json'
    if dest.exists():return json.loads(dest.read_text())
    child('comparison.calibration',[run],REPORT/(run.name+'_calibration.log'))
    return json.loads(dest.read_text())


def variant(parent,factor,value):
    o=copy.deepcopy(parent);o[factor]=value
    assert [k for k in o if o[k]!=parent[k]]==[factor],'A comparison must change exactly one setting'
    return o


def record(ledger,name,stage,factor,parent,o,run,status,seed=42):
    m=status['selected_metrics']
    ledger.append(dict(name=name,stage=stage,factor=factor,parent=parent,options=o,run=str(run),seed=seed,
                       selected_epoch=status['selected_epoch'],selection=m,model_sha256=status['model_sha256'],
                       negative_files=status.get('negative_files'),negative_unique_sampled=status.get('negative_unique_sampled'),
                       actual_optimizer_steps=status['actual_optimizer_steps']))
    write_json(REPORT/'ledger.json',ledger)
    write_csv(REPORT/'leaderboard.csv',[dict(name=r['name'],stage=r['stage'],factor=r['factor'],parent=r['parent'],
        target_recall=r['selection']['target_macro_group_recall'],domestic_recall=r['selection']['domestic_target_macro_group_recall'],
        fpr=r['selection']['foreign_fpr'],guard=r['selection']['guards_passed'],epoch=r['selected_epoch'],negative_unique=r['negative_unique_sampled']) for r in ledger])


def early():
    teacher=RUNS/'dscnn_comparison_20261114_teacher_base'
    t=json.loads((teacher/'status.json').read_text());assert t['complete'] and t['exit_code']==0
    ledger=json.loads((REPORT/'ledger.json').read_text()) if (REPORT/'ledger.json').exists() else []
    o=dict(BASE);name='dscnn_comparison_20261114_s1_baseline'
    run,status=execute(name,o,teacher)
    if not any(x['name']==name for x in ledger):record(ledger,name,1,'source_disjoint_selection_and_calibration','historical_models_not_control',o,run,status)
    c=calibrate(run)
    champion=ledger[0]
    # Each subsequent test has the current accepted champion as its single-factor
    # parent. No calibration or fresh-test metric enters this decision.
    experiments=[(2,'negative_sampler','coverage_cycle','s2_coverage'),
                 (2,'hard_negative_multiplier',2.,'s2_hardneg'),
                 (3,'pooling_fraction',.5,'s3_top50'),(3,'pooling_fraction',.25,'s3_top25')]
    for stage,factor,value,tag in experiments:
        if champion['options'][factor]==value:continue
        name='dscnn_comparison_20261114_'+tag
        old=next((x for x in ledger if x['name']==name),None)
        if old:
            entry=old
        else:
            opts=variant(champion['options'],factor,value);run,status=execute(name,opts,teacher)
            record(ledger,name,stage,factor,champion['name'],opts,run,status);entry=ledger[-1]
        adopted=score(entry['selection'])>score(champion['selection'])
        if adopted:champion=entry
        write_json(REPORT/'early_champion.json',dict(champion=champion,stage=stage,selection_only=True,calibration_or_test_used_to_choose=False))
        print('COMPARISON_DECISION',tag,'adopted',adopted,'champion',champion['name'],flush=True)
    write_json(REPORT/'early_complete.json',dict(champion=champion,actual_stages=[1,2,3],not_full_goal_complete=True))
    print('COMPARISON_EARLY_STAGES_COMPLETE',champion['name'],flush=True)


def late():
    from .exposure import audit
    from .protocol import DATA
    early=json.loads((REPORT/'early_complete.json').read_text());champion=early['champion']
    ledger=json.loads((REPORT/'ledger.json').read_text())
    source_freeze=json.loads((DATA/'source_freeze.json').read_text())
    assert all(sha(DATA/name)==digest for name,digest in source_freeze['files'].items())
    # Teacher settings are fixed to BASE; only the addition of training data
    # differs from teacher_base. Student settings never retune the teacher.
    if not (REPORT/'new_training_embeddings.json').exists():
        child('comparison.prepare',['embeddings'],REPORT/'new_training_embeddings_gpu.log')
    teacher_options=dict(BASE,added_training_data=True)
    newteacher,_=execute('dscnn_comparison_20261114_teacher_added',teacher_options,None,role='teacher')
    teachers={False:RUNS/'dscnn_comparison_20261114_teacher_base',True:newteacher}

    def compare(stage,factor,value,tag):
        nonlocal champion
        name='dscnn_comparison_20261114_'+tag
        old=next((x for x in ledger if x['name']==name),None)
        if old:entry=old
        else:
            opts=variant(champion['options'],factor,value);run,status=execute(name,opts,teachers[opts['added_training_data']])
            record(ledger,name,stage,factor,champion['name'],opts,run,status);entry=ledger[-1]
        adopted=score(entry['selection'])>score(champion['selection'])
        if adopted:champion=entry
        write_json(REPORT/'late_progress.json',dict(stage=stage,champion=champion,selection_only=True))
        print('COMPARISON_DECISION',tag,'adopted',adopted,'champion',champion['name'],flush=True)

    for stage,factor,value,tag in [(4,'added_training_data',True,'s4_added'),
                                  (5,'domestic_weight',2.,'s5_domestic2'),
                                  (5,'other_mass',.5,'s5_other50'),
                                  (6,'kd_weight',.25,'s6_kd25'),(6,'kd_weight',0.,'s6_kd0')]:
        compare(stage,factor,value,tag)
    anchor=champion;results={};entries={}
    for architecture in ('temporal24','freq24'):
        options=dict(anchor['options'],architecture=architecture);replicates=[]
        for seed in SEEDS:
            if seed==42 and options==anchor['options']:
                entry=anchor
            else:
                name=f'dscnn_comparison_20261114_s7_{architecture}_seed{seed}'
                old=next((x for x in ledger if x['name']==name),None)
                if old:entry=old
                else:
                    run,status=execute(name,options,teachers[options['added_training_data']],seed)
                    record(ledger,name,7,'architecture_paired_seed',anchor['name'],options,run,status,seed);entry=ledger[-1]
            replicates.append(entry);entries[architecture,seed]=entry
        metrics=[e['selection'] for e in replicates]
        import statistics
        results[architecture]=dict(seeds=SEEDS,runs=[e['run'] for e in replicates],
            target_recall_mean=statistics.mean(m['target_macro_group_recall'] for m in metrics),
            target_recall_std=statistics.stdev(m['target_macro_group_recall'] for m in metrics),
            domestic_recall_mean=statistics.mean(m['domestic_target_macro_group_recall'] or 0 for m in metrics),
            foreign_fpr_mean=statistics.mean(m['foreign_fpr'] for m in metrics),all_guards_passed=all(m['guards_passed'] for m in metrics))
    def architecture_rank(name):
        m=results[name];return (int(m['all_guards_passed']),m['target_recall_mean'],m['domestic_recall_mean'],-m['foreign_fpr_mean'],int(name=='freq24'))
    chosen=max(results,key=architecture_rank);champion=entries[chosen,42]
    write_json(REPORT/'architecture_replicates.json',dict(results=results,chosen=chosen,delivered_seed=42,seed_not_selected_by_score=True,
               paired_settings_except_architecture=True))
    # Learning-rate schedule is a separate one-factor comparison, after the
    # multi-seed architecture decision. Its actual step schedule is logged.
    compare(7,'lr_schedule','plateau','s7_plateau')
    if champion['options']['lr_schedule']=='plateau':
        for seed in (43,44):
            name=f'dscnn_comparison_20261114_s7_plateau_seed{seed}'
            old=next((x for x in ledger if x['name']==name),None)
            if not old:
                run,status=execute(name,champion['options'],teachers[champion['options']['added_training_data']],seed)
                record(ledger,name,7,'lr_schedule_replication',champion['name'],champion['options'],run,status,seed)
    for e in ledger:
        if not (Path(e['run'])/'selected_checkpoint_exposure_audit.json').exists():audit(Path(e['run']))
    frozen=dict(champion=champion,model_sha256=sha(Path(champion['run'])/'classifier.keras'),dataset_freeze_sha256=sha(DATA/'source_freeze.json'),
                selected_without_calibration_or_fresh_test=True,source_disjoint=True,actual_training_stages=[1,2,3,4,5,6,7],
                actual_hyperparameter_comparisons=len(ledger),not_full_goal_complete=True)
    write_json(REPORT/'final_candidate.json',frozen)
    calibrate(Path(champion['run']))
    print('COMPARISON_MODEL_STAGES_COMPLETE',champion['name'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['early','late']);a=p.parse_args()
    early() if a.stage=='early' else late()
