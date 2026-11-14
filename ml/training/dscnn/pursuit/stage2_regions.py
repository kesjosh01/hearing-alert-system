"""New development cohort: retain source roles; add explicitly bounded train views."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import re
import shutil
import numpy as np
from comparison.dataset import read_csv,write_csv,write_json,sha
from dscnn_baseline.frontend import window_starts
from .paths import ROOT,DATA,REPORT
from .data import audio_path
from .waveform_identity import identity
from .audioset_annotation_identity import originals

NAME='round06_strong_localized'
PLAN=REPORT/'stage2_regions_plan.json'


def present_indices(starts,intervals):
    starts=np.asarray(starts,np.float64)/16000.;keep=np.zeros(len(starts),bool)
    for a,b in intervals:
        overlap=np.maximum(0,np.minimum(starts+.975,b)-np.maximum(starts,a))
        keep|=overlap>=min(.5,(b-a)*.5)-1e-8
    return np.flatnonzero(keep).tolist()


def absent_indices(starts,intervals,music,guard=.25):
    starts=np.asarray(starts,np.float64)/16000.;keep=np.ones(len(starts),bool)
    for a,b in intervals+music:
        overlap=np.maximum(0,np.minimum(starts+1,b+guard)-np.maximum(starts,max(0,a-guard)))
        keep&=overlap<=1e-9
    return np.flatnonzero(keep).tolist()


def plan():
    if PLAN.exists():raise FileExistsError('Preserve the registered source and window rules')
    write_json(PLAN,dict(parent='round05_train_targets',parent_freeze_sha256=sha(DATA/'round05_train_targets/freeze.json'),
        exact_first_audioset_train_shards=1000,minimum_target_seconds=.2,published_end_tolerance_s=.05,
        training_positive_effective_window_s=.975,target_absent_window_s=1.,absent_guard_s=.25,
        target_absent_views='Original Google-complete annotation sources only; exclude target and Music intervals with guard. Never full-recording other relabels.',
        source_validation_metadata_must_complete_before_fit=True,
        selection_calibration_membership_and_labels_preserved=True,
        old_fresh_recordings_are_historical_only=True,
        new_unseen_test='Separate original Google eval source plan; no original first public test tuning',
        source_license_scope='Research; underlying recording rights and foundation exposure not certified',
        installed_software=False,actual_product_targets_verified=False,
        builder_sha256=sha(ROOT/'pursuit/stage2_regions.py')))
    print('STAGE2_REGION_RULES_REGISTERED',flush=True)


def freeze():
    spec=json.loads(PLAN.read_text());assert sha(ROOT/'pursuit/stage2_regions.py')==spec['builder_sha256']
    directory=DATA/NAME;assert not directory.exists()
    parent=DATA/spec['parent'];assert sha(parent/'freeze.json')==spec['parent_freeze_sha256']
    pf=json.loads((parent/'freeze.json').read_text());assert all(sha(parent/k)==v for k,v in pf['files'].items())
    catalog=REPORT/'stage2_new_source_catalog/eligible_audio_plan.json'
    if not catalog.exists():raise RuntimeError('Wait for the new source metadata catalog; no dataset written')
    cat=json.loads(catalog.read_text());assert cat['complete'] and not cat['model_scores_created']
    valid_ids=set(cat['all_publisher_valid_video_ids'])
    receipt_paths=[REPORT/'audioset_strong_intake'/f'{i:05d}.json' for i in range(spec['exact_first_audioset_train_shards'])]
    if any(not p.exists() for p in receipt_paths):raise RuntimeError('Wait for all registered target-source shards; no dataset written')
    source_plan=json.loads((REPORT/'audioset_strong_intake/plan.json').read_text())
    base=read_csv(parent/'inventory.csv');known_hashes=set();known_pcm=set();source_roles=defaultdict(set)
    for row in base:
        m=identity(row);row['source_sha256']=m['source_sha256'];row['canonical_pcm_sha256']=m['decoded_pcm_sha256']
        known_hashes.add(m['source_sha256']);known_pcm.add(m['decoded_pcm_sha256']);source_roles[m['decoded_pcm_sha256']].add(row['role'])
    registry=ROOT.parent/'yamnet/data/metadata/source_registry.csv'
    old_ids=set(re.findall(r'(?:youtu\.be/|[?&]v=|youtube\.com/shorts/)([A-Za-z0-9_-]{11})',registry.read_text()))
    additions=[];excluded=[];seen_videos={r.get('source_video_id') for r in base if r.get('source_video_id')}
    for i,path in enumerate(receipt_paths):
        receipt=json.loads(path.read_text());assert receipt['complete'] and receipt['source_sha256']==source_plan['files'][i]['sha256']
        for a in receipt['targets']:
            intervals=json.loads(a['event_intervals_json']);why=''
            if a['video_id'] in old_ids or a['video_id'] in seen_videos:why='Previously represented source video'
            elif a['video_id'] in valid_ids:why='Reserved publisher validation source video'
            elif not a['interval_identity_verified_against_original_metadata'] or not intervals:why='No unique original strong target identity'
            elif sum(b-e for e,b in intervals)<spec['minimum_target_seconds']:why='Very short target interval'
            elif a['audio_sha256'] in known_hashes:why='Previously represented source bytes'
            if why:excluded.append(dict(video_id=a['video_id'],reason=why));continue
            file=DATA/'audioset_strong_targets'/a['filename'];assert sha(file)==a['audio_sha256'] and receipt['full_source_sha256_verified']
            row=dict(sample_id='audioset-'+a['video_id']+'-'+a['audio_sha256'][:16],source_dataset='AudioSet-Strong-CLAPv2',
                source_recording_id=a['official_segment_id'],source_video_id=a['video_id'],split_group_id='audioset-video-'+a['video_id']+'-'+a['label'],
                creator_group_id='',creator_identity_known=False,role='train',split='train',label=a['label'],label_id=0 if a['label']=='fire_alarm' else 1,
                target_domain='general',sampler_pool='auxiliary_target',audio_root='dscnn',relative_path=str(file.relative_to(ROOT)),source_sha256=a['audio_sha256'],
                source_category='AudioSet published strong target',event_intervals_json=a['event_intervals_json'],human_event_timestamps=True,
                source_url=a['source_url'],source_revision=a['revision'],license_id='UNDERLYING_AUDIO_TERMS_UNVERIFIED',research_only=True,
                foundation_pretraining_exposure_possible=True,physical_device_identity_known=False,button_activation_ground_truth=False)
            m=identity(row);duration=m['sample_count']/16000.
            if abs(duration-a['audio_duration_s'])>.01 or not all(0<=e<b<=duration+spec['published_end_tolerance_s'] for e,b in intervals):
                excluded.append(dict(video_id=a['video_id'],reason='Original interval/source duration inconsistency'));continue
            if m['decoded_pcm_sha256'] in known_pcm:
                excluded.append(dict(video_id=a['video_id'],reason='Existing canonical PCM'));continue
            row['canonical_pcm_sha256']=m['decoded_pcm_sha256'];additions.append(row)
            known_hashes.add(m['source_sha256']);known_pcm.add(m['decoded_pcm_sha256']);seen_videos.add(a['video_id'])
    rows=base+additions;original=originals('train');views=[];derived=[]
    for row in rows:
        if row['role']!='train' or row['label']=='other':continue
        if row['source_dataset'] not in ('AudioSet-Strong-CLAPv2','RealDESED-20056072'):continue
        intervals=json.loads(row.get('event_intervals_json') or '[]')
        if not intervals:continue
        m=identity(row);starts=window_starts(m['sample_count']);positive=present_indices(starts,intervals)
        if not positive:raise ValueError('No source window can support the published positive interval')
        row['training_region_kind']='published_target_present';row['training_window_indices_json']=json.dumps(positive)
        row['original_source_sample_id']=row['sample_id'];row['original_source_group_id']=row['split_group_id']
        views.append(dict(sample_id=row['sample_id'],kind='published_target_present',window_indices=positive,source_intervals=intervals))
        if row['source_dataset']!='AudioSet-Strong-CLAPv2':continue
        video=row['source_video_id'];events=original[video][row['source_recording_id']]
        music=[[a,b] for mid,a,b in events if mid=='/m/04rlf']
        negative=absent_indices(starts,intervals,music,spec['absent_guard_s'])
        if not negative:continue
        neg=dict(row,sample_id=row['sample_id']+'__target_absent',label='other',label_id=2,
            split_group_id=row['split_group_id']+'__target_absent',sampler_pool='overseas_confuser',
            training_region_kind='published_target_absent',training_window_indices_json=json.dumps(negative),
            original_source_label=row['label'],file_label_not_changed=True,
            label_basis='Only indexed train windows outside published target/Music intervals and guard; not a whole-file negative')
        derived.append(neg);views.append(dict(sample_id=neg['sample_id'],kind='published_target_absent',window_indices=negative,
            source_intervals=intervals,music_intervals=music,guard_s=spec['absent_guard_s']))
    rows+=derived
    assert len({r['sample_id'] for r in rows})==len(rows)
    roles=defaultdict(set)
    for row in rows:roles[row['canonical_pcm_sha256']].add(row['role'])
    assert all(len(v)==1 for v in roles.values()),'Canonical source crosses roles'
    assert all(r['role']=='train' for r in derived)
    directory.mkdir();write_csv(directory/'inventory.csv',rows);write_csv(directory/'development.csv',[r for r in rows if r['role']!='fresh_test'])
    shutil.copyfile(parent/'fresh_test.csv',directory/'historical_reference.csv');write_json(directory/'training_views.json',views)
    files=['inventory.csv','development.csv','historical_reference.csv','training_views.json']
    write_json(directory/'freeze.json',dict(name=NAME,files={k:sha(directory/k) for k in files},parent_freeze_sha256=sha(parent/'freeze.json'),
        plan_sha256=sha(PLAN),new_source_metadata_catalog_sha256=sha(catalog),
        counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration','fresh_test']},
        additional_native_training_recordings=len(additions),additional_labels=dict(Counter(r['label'] for r in additions)),
        derived_training_region_examples=len(derived),derived_views_are_not_independent_recordings=True,
        source_admission_exclusions=excluded,source_receipt_sha256={p.name:sha(p) for p in receipt_paths},
        parent_membership_roles_and_labels_preserved=True,original_file_labels_not_rewritten=True,
        requires_region_aware_stream_loader=True,historical_reference_is_not_new_unseen_test=True,
        new_test_model_scores_created=False,actual_product_targets_verified=False))
    print('STAGE2_REGION_COHORT_FROZEN',len(additions),len(derived),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','freeze']);a=p.parse_args();plan() if a.mode=='plan' else freeze()
