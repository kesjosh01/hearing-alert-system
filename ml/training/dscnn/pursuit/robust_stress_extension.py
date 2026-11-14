"""Complete the preregistered third-candidate comparison without refitting."""
import json
import numpy as np
from comparison.dataset import write_json, sha
from dscnn_baseline.common import runtime
from .paths import ROOT, REPORT
from .robust_stress import PLAN, DEST, KINDS
from .cpu_integer import validated_interpreter
from . import delivered_integer as di


def main():
    name='domestic_known_enriched_v3'; bundle=REPORT/'candidates'/name
    cfg=json.loads((bundle/'delivered_output_contract.json').read_text())
    assert cfg['output_semantics']=='Quantized softmax probabilities'
    # The decision to add this branch was recorded before reading the first two results.
    prior=REPORT/'domestic_enriched_plan_before_stress_inspection.json'
    assert prior.exists()
    old=json.loads((REPORT/'domestic_holdout_acoustic_evaluation_freeze.json').read_text())
    identities=dict(old['candidates'])
    identities[name]=dict(model_sha256=sha(bundle/'model_int8.tflite'),readout_sha256=sha(bundle/'delivered_output_contract.json'))
    for n,j in identities.items():
        p=REPORT/'candidates'/n
        assert sha(p/'model_int8.tflite')==j['model_sha256'] and sha(p/'delivered_output_contract.json')==j['readout_sha256']
    frozen=REPORT/'domestic_holdout_acoustic_extension_freeze.json'
    assert not frozen.exists()
    write_json(frozen,dict(candidates=identities,original_freeze_sha256=sha(REPORT/'domestic_holdout_acoustic_evaluation_freeze.json'),
        extension_decision_before_old_results_inspection_sha256=sha(prior),plan_sha256=sha(PLAN),
        all_candidate_models_and_readouts_frozen=True,weights_or_thresholds_refitted=False,
        original_result_files_sha256={n:sha(REPORT/'domestic_holdout_acoustic_results'/(n+'.json')) for n in old['candidates']}))
    bank=json.loads((REPORT/'domestic_holdout_acoustic_features.json').read_text());assert bank['complete']
    tf,_=runtime('cpu')
    rep=np.load(REPORT/'domestic_enriched_delivered_v1/representative_training.npy',mmap_mode='r',allow_pickle=False)
    interpreter=validated_interpreter(tf,bundle/'model_int8.tflite',rep,REPORT/'domestic_holdout_acoustic_extension_cpu_gate.json',64)
    details=[];cuts=np.array(cfg['unsigned_probability_cuts'],np.int16)
    for r in bank['receipts']:
        p=DEST/r['source_sha256']/r['kind']/'features.npy';assert sha(p)==r['feature_sha256']
        q=di.predict_q(interpreter,np.load(p,mmap_mode='r',allow_pickle=False))
        scores=di.scores(q,'probability',cfg['output_zero_point'],cfg['policy'])
        flags=(scores.max(0)>=cuts)&(scores.max(0)>0);y=di.LABELS.index(r['label'])
        details.append(dict(sample_id=r['sample_id'],condition=r['kind'],label=r['label'],
            detected=bool(flags[y]),wrong_target_alert=bool(flags[1-y])))
    counts={c:{label:dict(files=sum(r['condition']==c and r['label']==label for r in details),
        detected_files=sum(r['detected'] for r in details if r['condition']==c and r['label']==label),
        wrong_target_alert_files=sum(r['wrong_target_alert'] for r in details if r['condition']==c and r['label']==label))
        for label in di.LABELS[:2]} for c in KINDS}
    write_json(REPORT/'domestic_holdout_acoustic_results'/(name+'.json'),dict(complete=True,identity=identities[name],
        conditions=counts,details=details,plan_sha256=sha(PLAN),signal_recordings_training_seen=True,
        noise_components_training_seen=True,transformation_combinations_not_used_in_training=True,
        thresholds_or_weights_refitted=False,independent_new_recordings=0,actual_device_or_event_reliability_proven=False))
    comparison={}
    for n in identities:
        j=json.loads((REPORT/'domestic_holdout_acoustic_results'/(n+'.json')).read_text())
        assert j['complete'] and j['identity']==identities[n];comparison[n]=j['conditions']
    write_json(REPORT/'domestic_holdout_acoustic_comparison.json',dict(complete=True,results=comparison,
        extension_freeze_sha256=sha(frozen),all_models_frozen_before_new_results_inspection=True,
        target_signals_and_noise_training_seen=True,independent_new_recordings=0,
        actual_product_targets_verified=False))
    print('THREE_FROZEN_DOMESTIC_CANDIDATES_ACOUSTIC_COMPARISON',comparison,flush=True)


if __name__=='__main__':main()
