"""Qualify a teacher for a matched compact student, retaining the old reference."""
import argparse
import json
import time
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .operating import measure,key

PLAN=REPORT/'stage2_compact_transfer_comparison_plan.json'
OUT=REPORT/'stage2_teacher_vs_native_control.json'
CONTROL='pursuit_20261114_r06_native40_supervised'
TEACHERS=['pursuit_20261114_r06_full_yamnet_head','pursuit_20261114_r06_full_yamnet_refined_unfused_bn','pursuit_20261114_r06_yamnet_semantic_prior']


def register():
    assert not PLAN.exists()
    write_json(PLAN,dict(created_at_unix=time.time(),control=CONTROL,teachers=TEACHERS,
        dataset_freeze_sha256=sha(DATA/'round06_local_regions/freeze.json'),
        prior_large_reference_comparison_sha256=sha(REPORT/'stage2_teacher_matched_comparison.json'),
        rationale='The new teachers did not improve the earlier larger prefix64 reference. A separate compression experiment may still help the new 54291-parameter native40 student if a teacher is stronger than that exact supervised control.',
        teacher_choice='Highest min(fire/.999,doorbell/.95), then macro source-mean file recall on identical extended selection recordings.',
        transfer_gate='Teacher no worse on either class and strictly higher normalized minimum than the completed native40 supervised control, under identical margin1/single-window selection readout.',
        original_large_reference_results_preserved=True,
        passing_this_gate_does_not_claim_teacher_improves_large_reference=True,
        no_new_kd_optimizer_updates_or_fresh_test_outcomes_observed=True,
        selection_only=True,calibration_or_fresh_used=False,actual_product_targets_verified=False,
        code_sha256=sha(ROOT/'pursuit/stage2_transfer_comparison.py')))
    print('MATCHED_COMPACT_TRANSFER_CRITERIA_REGISTERED',flush=True)


def evaluate():
    assert not OUT.exists()
    plan=json.loads(PLAN.read_text());assert sha(ROOT/'pursuit/stage2_transfer_comparison.py')==plan['code_sha256']
    assert sha(DATA/'round06_local_regions/freeze.json')==plan['dataset_freeze_sha256']
    assert sha(REPORT/'stage2_teacher_matched_comparison.json')==plan['prior_large_reference_comparison_sha256']
    directory=DATA/'round06_local_regions';rows=[r for r in read_csv(directory/'development.csv') if r['role']=='selection'];ids={r['sample_id'] for r in rows}
    records={}
    for name in [CONTROL,*TEACHERS]:
        run=ROOT/'runs'/name;status=json.loads((run/'status.json').read_text())
        assert status['complete'] and not status['fresh_test_used'] and not status['calibration_used']
        assert sha(run/'classifier.keras')==status['model_sha256']
        if name==CONTROL:assert status['actual_optimizer_steps']>0 and status['full_negative_training_view_coverage']
        with np.load(run/'selection_logits.npz',allow_pickle=False) as h:
            assert set(h.files)==ids
            metric=measure(rows,{k:h[k] for k in h.files},mode='margin')
        records[name]=dict(name=name,model_sha256=status['model_sha256'],selection_logits_sha256=sha(run/'selection_logits.npz'),metric=metric)
    chosen=max((records[name] for name in TEACHERS),key=lambda r:key(r['metric']));teacher=chosen['metric'];control=records[CONTROL]['metric']
    stronger=teacher['fire_recall']>=control['fire_recall'] and teacher['doorbell_recall']>=control['doorbell_recall'] and key(teacher)>key(control)
    prior=json.loads((REPORT/'stage2_teacher_matched_comparison.json').read_text())
    write_json(OUT,dict(complete=True,chosen_teacher=chosen['name'],chosen_model_sha256=chosen['model_sha256'],
        control=records[CONTROL],teachers=[records[n] for n in TEACHERS],distillation_gate_passed=bool(stronger),
        criterion=plan['transfer_gate'],teacher_choice=plan['teacher_choice'],selection_recordings=len(rows),
        prior_large_reference_improved=prior['distillation_gate_passed'],prior_large_reference_comparison_sha256=sha(REPORT/'stage2_teacher_matched_comparison.json'),
        conclusion_scope='Qualification for distillation into this exact compact student; the previous larger-model comparison remains a separate negative result.',
        calibration_or_fresh_used=False,actual_product_targets_verified=False,plan_sha256=sha(PLAN)))
    print('MATCHED_COMPACT_TRANSFER_COMPARISON_COMPLETE','KD_ALLOWED',stronger,chosen['name'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','evaluate']);a=p.parse_args();register() if a.mode=='register' else evaluate()
