"""Frozen interim cohort while the remaining public collection continues."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import prepare
from .strong import verify

PILOT=REPORT/'round03_pilot_plan.json'


def plan(count):
    if PILOT.exists():raise FileExistsError('Do not revise a scored pilot membership')
    source=DATA/'realdesed_priority_plan.json';j=json.loads(source.read_text());keys=[r['key'] for r in j['rows'][:count]]
    write_json(PILOT,dict(base_round='round02',new_source_plan_sha256=sha(source),exact_keys=keys,selection_before_model_scoring=True,
       count=count,all_remaining_collection_continues=True,strong_labels='Original human-published acoustic event intervals, not device-button activation IDs'))
    write_json(REPORT/'strong_window_cpu_gate.json',verify())
    print('STRONG_DATA_PILOT_PRE_REGISTERED',count,flush=True)


def freeze():
    pilot=json.loads(PILOT.read_text());source=DATA/'realdesed_priority_plan.json';assert sha(source)==pilot['new_source_plan_sha256']
    allrows={r['key']:r for r in json.loads(source.read_text())['rows']};out=DATA/'round03_pilot'
    if out.exists():raise FileExistsError('Preserve frozen pilot')
    missing=[k for k in pilot['exact_keys'] if not (DATA/'realdesed_members'/(k+'.json')).exists()]
    if missing:raise RuntimeError(f'Wait for verified source members: {len(missing)} missing; no dataset written')
    new=[]
    for key in pilot['exact_keys']:
        r=allrows[key];m=r['metadata'];p=DATA/'realdesed_members'/(key+'.wav');digest=sha(p);receipt=json.loads(p.with_suffix('.json').read_text());assert digest==receipt['audio_sha256']==r['audio']['sha256']
        group='realdesed-creator-'+hashlib.sha256(m['group'].encode()).hexdigest()[:16];role=r['role'];label=r['label']
        new.append(dict(sample_id=key,source_dataset='RealDESED-20056072',source_recording_id=m['source_id'],creator_group_id=group,split_group_id=group+'-'+label,
           role=role,split='train' if role=='train' else 'test' if role=='fresh_test' else 'validation',label=label,label_id=1 if label=='doorbell' else 2,
           target_domain='general',sampler_pool='auxiliary_target' if label=='doorbell' else 'overseas_confuser',audio_root='dscnn',relative_path=str(p.relative_to(ROOT)),source_sha256=digest,
           source_category='RealDESED / '+m['raw_metadata']['target_classes'],source_url='https://zenodo.org/records/20056072',license_id=m['license_name'],
           event_intervals_json=json.dumps([[e['onset'],e['offset']] for e in r['human_bell_events']]),event_annotation_source='Official published RealDESED consensus; reviewed validation/test',
           device_model=m['raw_metadata']['recording_device'],device_serial_identity_known=False,human_event_timestamps=True,button_activation_ground_truth=False,
           raw_metadata_json=json.dumps(m['raw_metadata'],ensure_ascii=False),mirror_revision=receipt['revision']))
    rows=read_csv(DATA/'round02/inventory.csv')+new;assert len({r['sample_id'] for r in rows})==len(rows)
    groups=defaultdict(set);digests=defaultdict(set)
    for r in rows:
        if r.get('creator_group_id'):groups[r['creator_group_id']].add(r['role'])
        if r.get('source_sha256'):digests[r['source_sha256']].add(r['role'])
    assert all(len(v)==1 for mapping in [groups,digests] for v in mapping.values())
    out.mkdir();write_csv(out/'inventory.csv',rows);write_csv(out/'development.csv',[r for r in rows if r['role']!='fresh_test']);write_csv(out/'fresh_test.csv',[r for r in rows if r['role']=='fresh_test'])
    counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration','fresh_test']}
    write_json(out/'freeze.json',dict(name=out.name,files={k:sha(out/k) for k in ['inventory.csv','development.csv','fresh_test.csv']},counts=counts,
       pilot_plan_sha256=sha(PILOT),added_files=len(new),new_human_acoustic_bouts=sum(len(json.loads(r['event_intervals_json'])) for r in new),
       event_windows_used_for_training_only=True,validation_and_test_always_full_recordings=True,source_and_known_creator_role_disjoint=True,
       original_audio_unchanged=True,physical_device_reliability_or_monthly_false_rate_proven=False,all_remaining_public_collection_continues=True))
    print('STRONG_SOURCE_PILOT_COHORT_FROZEN',counts,flush=True);prepare(out.name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','freeze']);p.add_argument('--count',type=int,default=64);a=p.parse_args();plan(a.count) if a.mode=='plan' else freeze()
