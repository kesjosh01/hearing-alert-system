"""Compare completed teachers on identical selection IDs, never fresh audio."""
import json
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from .paths import ROOT, DATA, REPORT
from .operating import measure, key


def main():
    target=REPORT/'stage2_teacher_matched_comparison.json'
    assert not target.exists(),'Preserve completed comparison'
    plan_path=REPORT/'stage2_teacher_unfused_study_plan.json'
    if not plan_path.exists():plan_path=REPORT/'stage2_teacher_study_plan.json'
    plan=json.loads(plan_path.read_text())
    old=DATA/'round04_enriched';old_freeze=json.loads((old/'freeze.json').read_text())
    assert all(sha(old/k)==v for k,v in old_freeze['files'].items())
    common=[r for r in read_csv(old/'development.csv') if r['role']=='selection']
    ids={r['sample_id'] for r in common}
    baseline=REPORT/'operating/round04_enriched/pursuit_20261114_r04_prefix64_pretrained.json'
    b=json.loads(baseline.read_text());bp=baseline.with_name(baseline.stem+'_selection_logits.npz')
    assert b['complete'] and sha(bp)==b['logit_cache_sha256']
    assert sha(ROOT/'runs'/b['model']/'classifier.keras')==b['model_sha256']
    with np.load(bp,allow_pickle=False) as h:
        assert set(h.files)==ids
        baseline_margin=measure(common,{k:h[k] for k in h.files},mode='margin')
    comparisons=[]
    for spec in plan['runs']:
        run=ROOT/'runs'/spec['name'];status=json.loads((run/'status.json').read_text())
        assert status['complete'] and status['actual_optimizer_steps']>0 and not status['fresh_test_used']
        assert sha(run/'classifier.keras')==status['model_sha256']
        selection=[r for r in read_csv(run/'used_train_selection.csv') if r['role']=='selection']
        assert ids.issubset(r['sample_id'] for r in selection)
        with np.load(run/'selection_logits.npz',allow_pickle=False) as h:
            assert set(h.files)=={r['sample_id'] for r in selection}
            same=measure(common,{k:h[k] for k in ids},mode='margin')
        comparisons.append(dict(name=spec['name'],model_sha256=status['model_sha256'],
            selection_logits_sha256=sha(run/'selection_logits.npz'),
            full_frozen_selection=status['selected_metrics'],matched_parent_selection=same,
            selected_epoch=status['selected_epoch'],actual_optimizer_steps=status['actual_optimizer_steps']))
    chosen=max(comparisons,key=lambda c:key(c['full_frozen_selection']))
    same=chosen['matched_parent_selection'];reference=b['best']
    stronger=(same['fire_recall']>=reference['fire_recall'] and same['doorbell_recall']>=reference['doorbell_recall'] and key(same)>key(reference))
    write_json(target,dict(complete=True,selection_only=True,common_selection_files=len(common),
        study_plan_sha256=sha(plan_path),
        baseline_model=b['model'],baseline_model_sha256=b['model_sha256'],
        baseline_previous_best_readout=reference,baseline_same_margin_readout=baseline_margin,
        candidates=comparisons,chosen_teacher=chosen['name'],chosen_model_sha256=chosen['model_sha256'],
        selected_on_entire_new_selection_cohort=True,distillation_gate_passed=bool(stronger),
        distillation_gate='No worse on either target and strictly better target-normalized minimum than the old checkpoint with its previous best readout, on identical parent selection IDs.',
        calibration_or_fresh_used=False,foundation_pretraining_independence_certified=False,
        repeated_selection_is_not_independent_test=True,actual_product_targets_verified=False))
    print('STAGE2_MATCHED_TEACHER_COMPARISON_COMPLETE',chosen['name'],'KD_ALLOWED',stronger,flush=True)


if __name__=='__main__':main()
