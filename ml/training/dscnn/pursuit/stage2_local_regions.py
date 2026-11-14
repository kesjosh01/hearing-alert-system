"""A new study from files already present when the user stopped collection.

This module has no network operations. Original manifests and the completed
first public benchmark remain unchanged. A new, locally available source set
is partitioned before fitting; publisher pretraining exposure remains unknown.
"""
import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict

from comparison.dataset import read_csv, write_csv, write_json, sha
from dscnn_baseline.frontend import window_starts
from .paths import ROOT, DATA, REPORT
from .stage2_regions import present_indices, absent_indices
from .waveform_identity import identity
from .audioset_annotation_identity import originals, display_names

NAME = 'round06_local_regions'
PLAN = REPORT / 'stage2_local_regions_plan.json'
STOP = REPORT / 'collection_stopped_by_user.json'


def rank(value):
    return hashlib.sha256(('stage2-local-source-20261114:' + value).encode()).hexdigest()


def plan():
    if PLAN.exists():
        raise FileExistsError('Preserve the pre-fit local-source rules')
    stop = json.loads(STOP.read_text())
    assert stop['collection_disabled_by_user'] and stop['all_selected_collectors_stopped']
    receipts = sorted((REPORT / 'audioset_strong_intake').glob('[0-9][0-9][0-9][0-9][0-9].json'))
    assert receipts and all(json.loads(p.read_text())['complete'] for p in receipts)
    write_json(PLAN, dict(
        created_at_unix=time.time(), name=NAME, parent='round05_train_targets',
        parent_freeze_sha256=sha(DATA / 'round05_train_targets/freeze.json'),
        supersedes_unexecuted_region_plan_sha256=sha(REPORT / 'stage2_regions_plan.json'),
        amendment_reason='User stopped all audio/remote metadata collection; incomplete external validation catalog is not resumed. Use only already downloaded, previously unscored source recordings.',
        stop_receipt_sha256=sha(STOP), remote_audio_or_metadata_collection=False,
        existing_audioset_receipts={p.name: sha(p) for p in receipts},
        original_google_train_metadata_sha256=sha(DATA / 'audioset_metadata/audioset_train_strong.tsv'),
        fsd_plan_sha256=sha(DATA / 'fsd_collection_plan.json'),
        new_audioset_roles='Source-video SHA256 rank: [0,.65) train, [.65,.80) selection, [.80,.90) calibration, [.90,1) new fresh test.',
        new_audioset_is_original_google_train=True,
        new_audioset_is_unseen_to_our_supervised_models=True,
        foundation_pretraining_exposure_certified_absent=False,
        physical_device_or_youtube_creator_independence_certified=False,
        fsd_other_caps=dict(train=512, selection=128, calibration=128, fresh_test=512),
        fsd_other_rule='Retain the previously assigned uploader role. Only uploaders/groups absent from the entire parent inventory; exclude target-like metadata; source-SHA/PCM de-duplicate before model scores.',
        minimum_target_seconds=.2, published_end_tolerance_s=.05,
        training_positive_effective_window_s=.975, target_absent_window_s=1., absent_guard_s=.25,
        absent_ambiguous_annotations='Exclude every published alert class, Music, and any class whose display name contains alarm, siren, bell, buzzer or ringing.',
        parent_membership_roles_labels_preserved=True,
        old_first_public_test_is_historical_and_never_tuned=True,
        new_fresh_features_and_scores_forbidden_until_models_and_readouts_frozen=True,
        no_new_feature_arrays_written_to_disk=True, actual_product_targets_verified=False,
        builder_sha256=sha(ROOT / 'pursuit/stage2_local_regions.py'),
        window_rules_sha256=sha(ROOT / 'pursuit/stage2_regions.py')))
    print('LOCAL_ONLY_STAGE2_RULES_REGISTERED', len(receipts), flush=True)


def freeze():
    spec = json.loads(PLAN.read_text())
    assert sha(STOP) == spec['stop_receipt_sha256']
    assert sha(ROOT / 'pursuit/stage2_local_regions.py') == spec['builder_sha256']
    assert sha(ROOT / 'pursuit/stage2_regions.py') == spec['window_rules_sha256']
    directory = DATA / NAME
    assert not directory.exists(), 'Preserve partial or completed cohort'
    parent = DATA / spec['parent']
    assert sha(parent / 'freeze.json') == spec['parent_freeze_sha256']
    pf = json.loads((parent / 'freeze.json').read_text())
    assert all(sha(parent / k) == v for k, v in pf['files'].items())
    base = read_csv(parent / 'inventory.csv')
    known_hashes, known_pcm = set(), set()
    source_identity = {}
    for row in base:
        m = identity(row)
        row['source_sha256'] = m['source_sha256']
        row['canonical_pcm_sha256'] = m['decoded_pcm_sha256']
        known_hashes.add(m['source_sha256'])
        known_pcm.add(m['decoded_pcm_sha256'])
        source_identity[row['sample_id']] = m
    old_creators = {r.get('creator_group_id') for r in base if r.get('creator_group_id')}
    old_groups = {r['split_group_id'] for r in base}
    old_ids = set(re.findall(r'(?:youtu\.be/|[?&]v=|youtube\.com/shorts/)([A-Za-z0-9_-]{11})',
        (ROOT.parent / 'yamnet/data/metadata/source_registry.csv').read_text()))
    seen_videos = {r.get('source_video_id') for r in base if r.get('source_video_id')}
    source_plan = json.loads((REPORT / 'audioset_strong_intake/plan.json').read_text())
    additions, excluded = [], []
    for filename, digest in spec['existing_audioset_receipts'].items():
        path = REPORT / 'audioset_strong_intake' / filename
        assert sha(path) == digest
        receipt = json.loads(path.read_text())
        assert receipt['complete'] and receipt['source_sha256'] == source_plan['files'][int(path.stem)]['sha256']
        for a in receipt['targets']:
            intervals = json.loads(a['event_intervals_json'])
            why = ''
            if a['video_id'] in old_ids or a['video_id'] in seen_videos:
                why = 'Previously represented source video'
            elif not a['interval_identity_verified_against_original_metadata'] or not intervals:
                why = 'No matched original target interval'
            elif sum(b-e for e, b in intervals) < spec['minimum_target_seconds']:
                why = 'Very short published target interval'
            elif a['audio_sha256'] in known_hashes:
                why = 'Previously represented source bytes'
            if why:
                excluded.append(dict(video_id=a['video_id'], reason=why)); continue
            file = DATA / 'audioset_strong_targets' / a['filename']
            assert sha(file) == a['audio_sha256'] and receipt['full_source_sha256_verified']
            value = int(rank(a['video_id']), 16) / 2**256
            role = 'train' if value < .65 else 'selection' if value < .80 else 'calibration' if value < .90 else 'fresh_test'
            row = dict(sample_id='audioset-' + a['video_id'] + '-' + a['audio_sha256'][:16],
                source_dataset='AudioSet-Strong-CLAPv2', source_recording_id=a['official_segment_id'], source_video_id=a['video_id'],
                split_group_id='audioset-video-' + a['video_id'] + '-' + a['label'], creator_group_id='', creator_identity_known=False,
                role=role, split='train' if role == 'train' else 'test' if role == 'fresh_test' else 'validation',
                label=a['label'], label_id=0 if a['label'] == 'fire_alarm' else 1, target_domain='general', sampler_pool='auxiliary_target',
                audio_root='dscnn', relative_path=str(file.relative_to(ROOT)), source_sha256=a['audio_sha256'],
                source_category='AudioSet published strong target', event_intervals_json=a['event_intervals_json'],
                human_event_timestamps=True, source_url=a['source_url'], source_revision=a['revision'],
                license_id='UNDERLYING_AUDIO_TERMS_UNVERIFIED', research_only=True,
                foundation_pretraining_exposure_possible=True, physical_device_identity_known=False,
                button_activation_ground_truth=False, stage2_addition=True)
            m = identity(row); duration = m['sample_count'] / 16000.
            if abs(duration-a['audio_duration_s']) > .01 or not all(0 <= e < b <= duration + spec['published_end_tolerance_s'] for e, b in intervals):
                excluded.append(dict(video_id=a['video_id'], reason='Original interval/source duration mismatch')); continue
            if m['decoded_pcm_sha256'] in known_pcm:
                excluded.append(dict(video_id=a['video_id'], reason='Previously represented canonical PCM')); continue
            row['canonical_pcm_sha256'] = m['decoded_pcm_sha256']
            additions.append(row); source_identity[row['sample_id']] = m
            known_hashes.add(m['source_sha256']); known_pcm.add(m['decoded_pcm_sha256']); seen_videos.add(a['video_id'])
    fsd_plan_path = DATA / 'fsd_collection_plan.json'
    assert sha(fsd_plan_path) == spec['fsd_plan_sha256']
    fsd = json.loads(fsd_plan_path.read_text()); admitted = Counter(); verified = {}
    for path in (REPORT / 'transport').rglob('*_complete.json'):
        r = json.loads(path.read_text()); assert r['revision'] == fsd['revision']; verified.update(r['sha256'])
    for raw in sorted(fsd['rows'], key=lambda r: rank(r['sample_id'])):
        role = raw['role']
        if raw['label'] != 'other' or admitted[role] >= spec['fsd_other_caps'][role]: continue
        if raw['split_group_id'] in old_groups or raw.get('creator_group_id') in old_creators: continue
        text = re.sub(r'[_\-.]+', ' ', (raw['title']+' '+' '.join(raw['tags'])).lower())
        if re.search(r'\b(door\s*bell|fire\s*alarm|smoke\s*(alarm|detector)|siren|feueralarm)\b', text): continue
        file = DATA / 'fsd_audio' / raw['hf_path']; expected = verified.get(raw['hf_path'])
        assert expected and sha(file) == expected
        if expected in known_hashes: continue
        row = dict(raw, audio_root='dscnn', relative_path=str(file.relative_to(ROOT)), source_sha256=expected, stage2_addition=True)
        m = identity(row)
        if m['decoded_pcm_sha256'] in known_pcm: continue
        row['canonical_pcm_sha256'] = m['decoded_pcm_sha256']
        additions.append(row); source_identity[row['sample_id']] = m; admitted[role] += 1
        known_hashes.add(expected); known_pcm.add(m['decoded_pcm_sha256'])
    rows = base + additions
    original = originals('train'); names = display_names(); views, derived = [], []
    ambiguous = {mid for mid, name in names.items() if re.search(r'alarm|siren|bell|buzzer|ringing', name, re.I)} | {'/m/04rlf'}
    for row in rows:
        if row['role'] != 'train' or row['label'] == 'other': continue
        if row['source_dataset'] not in ('AudioSet-Strong-CLAPv2', 'RealDESED-20056072'): continue
        intervals = json.loads(row.get('event_intervals_json') or '[]')
        if not intervals: continue
        starts = window_starts(source_identity[row['sample_id']]['sample_count'])
        positive = present_indices(starts, intervals)
        assert positive, 'Published target has no supporting input window'
        row.update(training_region_kind='published_target_present', training_window_indices_json=json.dumps(positive),
            original_source_sample_id=row['sample_id'], original_source_group_id=row['split_group_id'])
        views.append(dict(sample_id=row['sample_id'], kind='published_target_present', window_indices=positive, source_intervals=intervals))
        if row['source_dataset'] != 'AudioSet-Strong-CLAPv2': continue
        events = original[row['source_video_id']][row['source_recording_id']]
        mask_intervals = [[a,b] for mid,a,b in events if mid in ambiguous]
        negative = absent_indices(starts, intervals, mask_intervals, spec['absent_guard_s'])
        if not negative: continue
        neg = dict(row, sample_id=row['sample_id']+'__target_absent', label='other', label_id=2,
            split_group_id=row['split_group_id']+'__target_absent', sampler_pool='overseas_confuser',
            training_region_kind='published_target_absent', training_window_indices_json=json.dumps(negative),
            original_source_label=row['label'], file_label_not_changed=True,
            label_basis='Indexed train windows outside published target/ambiguous-alert/Music intervals; never a full-file negative')
        derived.append(neg); views.append(dict(sample_id=neg['sample_id'], kind='published_target_absent', window_indices=negative))
    rows += derived
    assert len({r['sample_id'] for r in rows}) == len(rows)
    for field in ['canonical_pcm_sha256', 'split_group_id', 'creator_group_id', 'source_video_id']:
        roles = defaultdict(set)
        for row in rows:
            if row.get(field): roles[row[field]].add(row['role'])
        assert all(len(v) == 1 for v in roles.values()), 'Cross-role identity: ' + field
    new_fresh = [r for r in additions if r['role'] == 'fresh_test']
    assert all(any(r['label'] == label for r in new_fresh) for label in ['fire_alarm','doorbell','other'])
    directory.mkdir()
    write_csv(directory/'inventory.csv', rows)
    write_csv(directory/'development.csv', [r for r in rows if r['role'] != 'fresh_test'])
    write_csv(directory/'fresh_test.csv', new_fresh)
    shutil.copyfile(parent/'fresh_test.csv', directory/'historical_reference.csv')
    write_json(directory/'training_views.json', views)
    files = ['inventory.csv','development.csv','fresh_test.csv','historical_reference.csv','training_views.json']
    write_json(directory/'freeze.json', dict(name=NAME, files={k:sha(directory/k) for k in files}, plan_sha256=sha(PLAN),
        counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration','fresh_test']},
        additional_recording_counts={role:dict(Counter(r['label'] for r in additions if r['role']==role)) for role in ['train','selection','calibration','fresh_test']},
        additional_native_recordings=len(additions), derived_training_region_examples=len(derived),
        new_fresh_test_counts=dict(Counter(r['label'] for r in new_fresh)), source_admission_exclusions=excluded,
        source_receipt_sha256=spec['existing_audioset_receipts'], parent_freeze_sha256=sha(parent/'freeze.json'),
        parent_membership_roles_and_labels_preserved=True, original_file_labels_not_rewritten=True,
        derived_views_are_not_independent_recordings=True, requires_region_aware_stream_loader=True,
        historical_reference_is_not_new_unseen_test=True, all_new_sources_previously_downloaded=True,
        remote_collection=False, new_test_model_scores_created=False, actual_product_targets_verified=False))
    print('LOCAL_ONLY_STAGE2_COHORT_FROZEN', len(additions), len(derived), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['plan','freeze']); a = p.parse_args()
    plan() if a.mode == 'plan' else freeze()
