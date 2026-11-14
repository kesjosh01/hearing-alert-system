"""Index official bell events without altering a scored dataset or its roles."""
import hashlib
import json
from collections import defaultdict,Counter
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .realdesed_members import order


def main():
    target=REPORT/'realdesed_full_bell_index.json'
    if target.exists():raise FileExistsError('Preserve completed source index')
    directory=REPORT/'realdesed_recovery';status=json.loads((directory/'status.json').read_text());assert status['complete']
    catalog=read_csv(directory/'official_source_catalog.csv');assert sha(directory/'official_source_catalog.csv')==status['catalog_sha256']
    grouped=defaultdict(set);mirror=[]
    for line in (DATA/'realdesed_mirror_index/manifest.jsonl').read_text().splitlines():
        r=json.loads(line);mirror.append(r);grouped[r['metadata']['group']].add(r['split'])
    old_roles={g:'fresh_test' if 'test' in splits else ('selection' if int(order(g)[:8],16)%2==0 else 'calibration') if 'validation' in splits else 'train' for g,splits in grouped.items()}
    priority=json.loads((DATA/'realdesed_priority_plan.json').read_text())
    prior={r['audio']['sha256']:r for r in priority['rows']}
    all_rows=[];new_groups=defaultdict(set)
    for split in ['train','validation','test']:
        def table(name):
            r=next(r for r in catalog if r['original_split']==split and r['filename']==name)
            return read_csv(ROOT/r['relative_path'])
        metadata={r['filename']:r for r in table('metadata.csv')};events=defaultdict(list);creators=defaultdict(set)
        for r in table('annotations.csv'):
            if r['class']=='bell_ringing':events[r['filename']].append([float(r['onset']),float(r['offset'])])
        for r in table('annotations_raw.csv'):
            if r['annotator_is_creator'].lower()=='true':creators[r['filename']].add(r['annotator_id'])
        natives={r['filename']:r for r in catalog if r['original_split']==split and r['filename'].endswith('.wav')}
        assert set(metadata)==set(natives)
        for filename,intervals in events.items():
            r=natives[filename];m=metadata[filename];ids=sorted(creators[filename])
            p=prior.get(r['source_sha256']);license=m['license']
            group='realdesed:creator:'+ids[0] if len(ids)==1 else p['metadata']['group'] if p else 'realdesed:recording:'+filename
            new_groups[group].add(split)
            admitted_license='CC0' in license or 'CC BY' in license or 'CC-BY' in license
            all_rows.append(dict(sample_id=p['key'] if p else 'realdesed-official-'+split+'-'+filename[:-4],
                source_dataset='RealDESED-20056072',source_recording_id='realdesed:unpacked/'+split+'.zip.contents/audio/'+filename,
                source_sha256=r['source_sha256'],relative_path=r['relative_path'],audio_root='dscnn',
                original_split=split,label='doorbell',label_id=1,target_domain='general',sampler_pool='auxiliary_target',
                creator_source_group=group,creator_ids_json=json.dumps(ids),creator_identity_known=len(ids)==1,
                creator_group_id='realdesed-creator-'+hashlib.sha256(group.encode()).hexdigest()[:16] if len(ids)==1 else '',
                previously_in_priority=p is not None,event_intervals_json=json.dumps(intervals),
                human_event_timestamps=True,button_activation_ground_truth=False,
                license_id=license,license_eligible=admitted_license,
                source_url='https://zenodo.org/records/20056072',raw_metadata_json=json.dumps(m,ensure_ascii=False),
                device_model=m['recording_device'],device_serial_identity_known=False,
                quarantine_reason='' if len(ids)==1 and admitted_license else 'Creator identity missing/ambiguous or license not eligible; preserve historical role but no new independent-source admission'))
    for r in all_rows:
        g=r['creator_source_group'];splits=new_groups[g]
        role=old_roles[g] if g in old_roles else 'fresh_test' if 'test' in splits else ('selection' if int(order(g)[:8],16)%2==0 else 'calibration') if 'validation' in splits else 'train'
        if not g:role='quarantine'
        r['role']=role;r['split']='train' if role=='train' else 'test' if role=='fresh_test' else 'validation'
        r['split_group_id']=r['creator_group_id']+'-doorbell' if r['creator_identity_known'] else 'realdesed-recording-'+hashlib.sha256(g.encode()).hexdigest()[:16]+'-doorbell'
        if r['source_sha256'] in prior:assert role==prior[r['source_sha256']]['role'],'Never move a trained creator into a fresh role'
    inventory=DATA/'realdesed_official_bells.csv';write_csv(inventory,all_rows)
    new=[r for r in all_rows if not r['previously_in_priority'] and not r['quarantine_reason']]
    write_json(target,dict(complete=True,official_bell_recordings=len(all_rows),published_acoustic_bouts=sum(len(json.loads(r['event_intervals_json'])) for r in all_rows),
        prior_priority_bell_recordings=sum(r['previously_in_priority'] for r in all_rows),
        additional_eligible_bell_recordings=len(new),additional_role_counts=dict(Counter(r['role'] for r in new)),
        inventory_sha256=sha(inventory),all_existing_creator_roles_preserved=True,
        original_audio_copied_or_modified=False,new_features_or_model_scores_created=False,
        scored_round_membership_changed=False,actual_product_targets_verified=False))
    print('OFFICIAL_FULL_BELL_SOURCE_INDEX_COMPLETE',len(all_rows),'additional',len(new),dict(Counter(r['role'] for r in new)),flush=True)


if __name__=='__main__':main()
