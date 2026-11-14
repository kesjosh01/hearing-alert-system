"""Explicit user-authorized memorization branch, never an independent KR test."""
import json
from collections import Counter
from dscnn_baseline.common import source,config
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import DATA,REPORT
from .data import prepare


def main():
    out=DATA/'domestic_seen_v1'
    if out.exists():raise FileExistsError('Preserve branch data boundary')
    parent=DATA/'round02';rows=read_csv(parent/'inventory.csv');folds,_=source.load_splits(config()[1])
    existing={r['sample_id'] for r in rows}
    historical=[dict(r,role='train',audio_root='yamnet',previous_campaign_role='historical_test') for r in folds[0]
        if r['split']=='test' and r['target_domain']=='ko_target' and r['sample_id'] not in existing]
    rows+=historical
    positives=[];others=[]
    for r in rows:
        if r['label'] in ('fire_alarm','doorbell') and r['target_domain'] in ('ko_target','ko_style_effect'):
            positives.append(dict(r,role='train',split='train',previous_general_role=r['role']))
        elif r['label']=='other' and r['role'] in ('train','selection','calibration'):
            others.append(r)
    assert positives and all(r['role']=='train' for r in positives)
    rr=positives+others;assert len({r['sample_id'] for r in rr})==len(rr)
    probes=[dict(r,measure='TRAINING_SEEN_POSITIVE_FILE_PROBE_NOT_HELDOUT') for r in positives]
    out.mkdir();write_csv(out/'inventory.csv',rr);write_csv(out/'development.csv',rr);write_csv(out/'known_positive_probes.csv',probes)
    write_json(out/'freeze.json',dict(name=out.name,files={k:sha(out/k) for k in ['inventory.csv','development.csv','known_positive_probes.csv']},
       counts={role:dict(Counter(r['label'] for r in rr if r['role']==role)) for role in ['train','selection','calibration']},
       explicit_user_authorized_domestic_overfitting=True,known_positive_probe_files=len(probes),historical_KR_files_now_seen=[r['sample_id'] for r in historical],
       historical_original_manifests_changed=False,independent_positive_test_files=0,
       target_metric='Memorization of collected domestic-classified recordings only',product_targets_verified=False,
       new_physical_devices_or_other_homes_reliability_claim=False))
    write_json(REPORT/'domestic_branch_boundary.json',dict(branch=str(out),positive_files=len(positives),historical_KR_files_now_training=len(historical),
       source_counts=dict(Counter((r['source_dataset']+' / '+r['label']) for r in positives)),general_round01_and_round02_boundaries_unchanged=True,
       results_must_be_labeled_training_seen=True))
    print('DOMESTIC_SEEN_BRANCH_FROZEN',len(positives),'positive files',len(others),'negative files',flush=True)
    prepare(out.name)


if __name__=='__main__':main()
