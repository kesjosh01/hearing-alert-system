"""Training-target expansion with unchanged selection/calibration/test sets."""
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter,defaultdict
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import prepare,audio_path
from .waveform_identity import identity

NAME='round05_train_targets'
PLAN=REPORT/'round05_target_expansion_plan.json'


def plan():
    if PLAN.exists():raise FileExistsError('Preserve target admission criteria before new-round scores')
    source=REPORT/'audioset_strong_intake/plan.json';j=json.loads(source.read_text())
    write_json(PLAN,dict(parent='round04_enriched',parent_freeze_sha256=sha(DATA/'round04_enriched/freeze.json'),
        audioset_plan_sha256=sha(source),exact_first_source_shards=600,
        admission='Training-source clips with a unique matching published strong segment and >=0.2 sec total target intervals; no target model scores used',
        minimum_total_published_target_seconds=.2,
        known_youtube_source_ids_excluded_before_fit=True,new_realdesed_known_creator_train_only=True,
        selection_calibration_and_fresh_membership_unchanged=True,
        new_round_model_scores_used_for_membership=False,commercial_clearance_claimed=False))
    print('TRAINING_TARGET_EXPANSION_PLAN_FROZEN',flush=True)


def freeze():
    out=DATA/NAME
    if out.exists():raise FileExistsError('Preserve existing round')
    spec=json.loads(PLAN.read_text());parent=DATA/spec['parent']
    assert sha(parent/'freeze.json')==spec['parent_freeze_sha256']
    source=REPORT/'audioset_strong_intake/plan.json';assert sha(source)==spec['audioset_plan_sha256'];source=json.loads(source.read_text())
    paths=[REPORT/'audioset_strong_intake'/f'{i:05d}.json' for i in range(spec['exact_first_source_shards'])]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:raise RuntimeError(f'Wait for all preregistered source shards: {len(missing)} missing; no cohort written')
    base=read_csv(parent/'inventory.csv');new=[];excluded=[]
    registry=ROOT.parent/'yamnet/data/metadata/source_registry.csv'
    ids=set(re.findall(r'(?:youtu\.be/|[?&]v=|youtube\.com/shorts/)([A-Za-z0-9_-]{11})',registry.read_text()))
    hashes={r.get('source_sha256') or sha(audio_path(r)) for r in base}
    known_pcm={identity(r)['decoded_pcm_sha256'] for r in base}
    for i,p in enumerate(paths):
        r=json.loads(p.read_text());assert r['complete'] and r['source_sha256']==source['files'][i]['sha256']
        for a in r['targets']:
            reason=''
            intervals=json.loads(a['event_intervals_json'])
            if a['video_id'] in ids:reason='Known previous YouTube source ID'
            elif not a['interval_identity_verified_against_original_metadata'] or not intervals:reason='No unique published target interval match'
            elif sum(b-e for e,b in intervals)<spec['minimum_total_published_target_seconds']:reason='Only a very short published target fragment'
            elif a['audio_sha256'] in hashes:reason='Already represented source bytes'
            if reason:excluded.append(dict(video_id=a['video_id'],audio_sha256=a['audio_sha256'],reason=reason));continue
            assert r['full_source_sha256_verified']
            file=DATA/'audioset_strong_targets'/a['filename'];digest=sha(file);assert digest==a['audio_sha256']
            assert all(0<=e<b<=a['audio_duration_s']+.05 for e,b in intervals)
            label=a['label'];group='audioset-video-'+a['video_id']
            candidate=dict(sample_id='audioset-'+a['video_id']+'-'+digest[:16],source_dataset='AudioSet-Strong-CLAPv2',
                source_recording_id=a['official_segment_id'],split_group_id=group+'-'+label,creator_group_id='',creator_identity_known=False,
                role='train',split='train',label=label,label_id=0 if label=='fire_alarm' else 1,
                target_domain='general',sampler_pool='auxiliary_target',audio_root='dscnn',relative_path=str(file.relative_to(ROOT)),source_sha256=digest,
                source_category='AudioSet published target MIDs: '+','.join(a['weak_target_mid_ids']),
                source_url=a['source_url'],source_video_id=a['video_id'],source_revision=a['revision'],
                license_id='UNDERLYING_AUDIO_TERMS_UNVERIFIED',research_only=True,
                event_intervals_json=a['event_intervals_json'],human_event_timestamps=True,
                annotation_source='Published AudioSet strong TSV; matched mirror event metadata, not new human listening',
                original_audio_content_independently_verified_against_youtube=False,button_activation_ground_truth=False,
                foundation_pretraining_exposure_possible=True,physical_device_identity_known=False)
            pcm=identity(candidate)['decoded_pcm_sha256']
            if pcm in known_pcm:
                excluded.append(dict(video_id=a['video_id'],audio_sha256=digest,reason='Exact canonical PCM already represented in parent or earlier admitted source'));continue
            candidate['canonical_pcm_sha256']=pcm;new.append(candidate);hashes.add(digest);known_pcm.add(pcm)
    full_index=DATA/'realdesed_official_bells.csv';index=json.loads((REPORT/'realdesed_full_bell_index.json').read_text());assert sha(full_index)==index['inventory_sha256']
    for r in read_csv(full_index):
        if r['role']=='train' and r['previously_in_priority']=='False' and not r['quarantine_reason']:
            assert r['creator_identity_known']=='True' and sha(ROOT/r['relative_path'])==r['source_sha256']
            if r['source_sha256'] not in hashes:
                pcm=identity(r)['decoded_pcm_sha256']
                if pcm in known_pcm:
                    excluded.append(dict(sample_id=r['sample_id'],reason='Exact canonical PCM already represented'));continue
                new.append(dict(r,canonical_pcm_sha256=pcm));hashes.add(r['source_sha256']);known_pcm.add(pcm)
    rows=base+new;assert len({r['sample_id'] for r in rows})==len(rows)
    groups=defaultdict(set);creators=defaultdict(set);digests=defaultdict(set)
    for r in rows:
        groups[r['split_group_id']].add(r['role'])
        if r.get('creator_group_id'):creators[r['creator_group_id']].add(r['role'])
        if r.get('source_sha256'):digests[r['source_sha256']].add(r['role'])
    assert all(len(v)==1 for m in [groups,creators,digests] for v in m.values())
    assert all(r['role']=='train' and r['label']!='other' for r in new)
    out.mkdir();write_csv(out/'inventory.csv',rows);write_csv(out/'development.csv',[r for r in rows if r['role']!='fresh_test'])
    shutil.copyfile(parent/'fresh_test.csv',out/'fresh_test.csv')
    write_json(out/'freeze.json',dict(name=NAME,files={k:sha(out/k) for k in ['inventory.csv','development.csv','fresh_test.csv']},
        counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration','fresh_test']},
        parent_freeze_sha256=sha(parent/'freeze.json'),target_plan_sha256=sha(PLAN),added_training_targets=len(new),
        added_label_counts=dict(Counter(r['label'] for r in new)),added_source_counts=dict(Counter(r['source_dataset'] for r in new)),
        source_admission_exclusions=excluded,source_receipt_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        all_existing_examples_and_roles_unchanged=True,selection_calibration_and_fresh_membership_unchanged=True,
        source_identity_policy='Exclude new raw-byte or canonical PCM duplicates against all parent roles before fitting; no model score consulted',
        builder_source_sha256=sha(ROOT/'pursuit/target_round.py'),
        audioset_sources_train_only=True,independent_new_devices_not_claimed=True,
        license_scope='Research including legacy NC and unverified underlying AudioSet recording terms; not commercially cleared',
        fresh_test_features_or_scores_created=False,actual_product_targets_verified=False))
    print('TRAINING_TARGETS_EXPANDED_WITH_UNCHANGED_EVALUATION',len(new),dict(Counter(r['label'] for r in new)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','freeze','prepare']);a=p.parse_args()
    plan() if a.mode=='plan' else freeze() if a.mode=='freeze' else prepare(NAME)
