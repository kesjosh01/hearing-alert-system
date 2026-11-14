"""A new, additive cohort: source-native targets, Korean speech and confusers.

Membership is frozen before any new-round model scoring. Synthetic examples
form one training-only family; published bell intervals remain training-only.
All historical role bindings and manifests are retained.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from comparison.dataset import read_csv, write_csv, write_json, sha
from .paths import ROOT, DATA, REPORT
from .data import confirmed_new, prepare

NAME = 'round04_enriched'


def real_rows():
    source = DATA / 'realdesed_priority_plan.json'
    out = []
    for r in json.loads(source.read_text())['rows']:
        key = r['key']; m = r['metadata']
        p = DATA / 'realdesed_members' / (key + '.wav')
        receipt = json.loads(p.with_suffix('.json').read_text())
        digest = sha(p)
        assert digest == receipt['audio_sha256'] == r['audio']['sha256']
        group = 'realdesed-creator-' + hashlib.sha256(m['group'].encode()).hexdigest()[:16]
        label = r['label']; role = r['role']
        out.append(dict(sample_id=key, source_dataset='RealDESED-20056072',
            source_recording_id=m['source_id'], creator_group_id=group,
            split_group_id=group+'-'+label, role=role,
            split='train' if role=='train' else 'test' if role=='fresh_test' else 'validation',
            label=label, label_id=1 if label=='doorbell' else 2, target_domain='general',
            sampler_pool='auxiliary_target' if label=='doorbell' else 'overseas_confuser',
            audio_root='dscnn', relative_path=str(p.relative_to(ROOT)), source_sha256=digest,
            source_category='RealDESED / '+m['raw_metadata']['target_classes'],
            source_url='https://zenodo.org/records/20056072', license_id=m['license_name'],
            event_intervals_json=json.dumps([[e['onset'],e['offset']] for e in r['human_bell_events']]),
            event_annotation_source='Official published RealDESED consensus; reviewed validation/test',
            device_model=m['raw_metadata']['recording_device'], device_serial_identity_known=False,
            human_event_timestamps=True, button_activation_ground_truth=False,
            raw_metadata_json=json.dumps(m['raw_metadata'],ensure_ascii=False),
            mirror_revision=receipt['revision']))
    return out


def research_rows():
    plan=json.loads((DATA/'fsd_research_targets.json').read_text())
    receipt=json.loads((REPORT/'hub_subsets/fsd_research_audio/complete.json').read_text())
    assert receipt['complete'] and receipt['revision']==plan['revision']
    out=[]
    for r in plan['rows']:
        p=DATA/'fsd_research_audio'/r['hf_path']; digest=sha(p)
        assert digest==receipt['sha256'][r['hf_path']]
        label=r['label'];role=r['role']
        out.append(dict(r,source_dataset='FSD50K',
            split_group_id=r['creator_group_id']+'-'+label,
            split='train' if role=='train' else 'test' if role=='fresh_test' else 'validation',
            label_id=0 if label=='fire_alarm' else 1, target_domain='general',
            sampler_pool='auxiliary_target',audio_root='dscnn',
            relative_path=str(p.relative_to(ROOT)),source_sha256=digest,
            source_category='FSD50K published target label',license_id=r['license_url']))
    return out


def synthetic_rows():
    receipt=json.loads((REPORT/'hub_subsets/synthetic_smoke/complete.json').read_text())
    assert receipt['complete']
    paths=sorted(k for k in receipt['sha256'] if k.endswith('.wav')); assert len(paths)==250
    out=[]
    for rel in paths:
        p=DATA/'synthetic_smoke'/rel; digest=sha(p);assert digest==receipt['sha256'][rel]
        out.append(dict(sample_id='audiodet-synth-'+p.stem,
            source_dataset='audiodet-synth-smoke',source_recording_id=p.stem,
            creator_group_id='audiodet-synth-smoke-procedural-family',
            split_group_id='audiodet-synth-smoke-procedural-family-fire_alarm',
            role='train',split='train',label='fire_alarm',label_id=0,target_domain='synthetic',
            sampler_pool='auxiliary_target',audio_root='dscnn',
            relative_path=str(p.relative_to(ROOT)),source_sha256=digest,
            source_category='Procedural T3 smoke alarm; one generator family',
            license_id='CC-BY-4.0',source_revision=receipt['revision'],
            source_url='https://huggingface.co/datasets/ShantyCam/audiodet-synth-smoke',
            synthetic=True,human_event_timestamps=False,physical_device_count=0))
    return out


def freeze():
    dest=DATA/NAME
    if dest.exists():raise FileExistsError('Preserve frozen round membership')
    selected=set();receipts=[]
    for bucket,count in [('hard_negatives',25),('background',32)]:
        for i in range(count):
            p=REPORT/'transport'/bucket/f'{i:04d}_complete.json'
            j=json.loads(p.read_text());selected.update(j['paths']);receipts.append(p)
    new=[r for r in confirmed_new() if r['bucket']=='targets' or r['hf_path'] in selected]
    previous=read_csv(DATA/'round03_pilot/inventory.csv')
    old=[r for r in previous if r.get('source_dataset') not in ('FSD50K','RealDESED-20056072')]
    fleurs=read_csv(DATA/'fleurs_ko_audio/eligible.csv')
    assert sha(DATA/'fleurs_ko_audio/eligible.csv')==json.loads((REPORT/'fleurs_intake.json').read_text())['eligible_sha256']
    rows=old+new+real_rows()+research_rows()+synthetic_rows()+fleurs
    ids={};groups=defaultdict(set);creators=defaultdict(set);raw=defaultdict(set)
    for r in rows:
        assert r['sample_id'] not in ids;ids[r['sample_id']]=r
        groups[r['split_group_id']].add(r['role'])
        if r.get('creator_group_id'):creators[r['creator_group_id']].add(r['role'])
        if r.get('source_sha256'):raw[r['source_sha256']].add(r['role'])
    assert all(len(v)==1 for m in [groups,creators,raw] for v in m.values()),'Known source families cross roles'
    for r in previous:
        n=ids[r['sample_id']]
        assert all(str(n.get(k,''))==str(r.get(k,'')) for k in ['role','label','split_group_id','creator_group_id','source_sha256']),r['sample_id']
    dest.mkdir();write_csv(dest/'inventory.csv',rows)
    write_csv(dest/'development.csv',[r for r in rows if r['role']!='fresh_test'])
    write_csv(dest/'fresh_test.csv',[r for r in rows if r['role']=='fresh_test'])
    counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration','fresh_test']}
    write_json(dest/'freeze.json',dict(name=NAME,
        files={k:sha(dest/k) for k in ['inventory.csv','development.csv','fresh_test.csv']},counts=counts,
        source_counts=dict(Counter(r['source_dataset'] for r in rows)),
        previous_round='round03_pilot',all_previous_membership_and_role_bindings_preserved=True,
        exact_prior_freeze_sha256=sha(DATA/'round03_pilot/freeze.json'),
        transport_receipt_hashes={str(p.relative_to(ROOT)):sha(p) for p in receipts},
        model_scores_used_to_choose_new_examples=False,fresh_test_features_or_scores_created=False,
        synthetic_recordings=250,synthetic_independent_device_count=0,synthetic_train_only=True,
        realdesed_published_intervals='Acoustic bouts, not physical button-activation ground truth',
        known_source_creator_raw_byte_role_disjoint=True,physical_device_independence_certified=False,
        foundation_pretraining_exposure_certified_absent=False,
        license_scope='Research: includes legacy ESC-50 and explicit CC-BY-NC FSD targets; not commercially cleared',
        actual_product_targets_verified=False,remaining_collection_continues=True))
    print('ENRICHED_ROUND_FROZEN',NAME,counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare']);a=p.parse_args()
    freeze() if a.mode=='freeze' else prepare(NAME)
