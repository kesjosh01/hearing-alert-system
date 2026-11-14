"""Freeze new source partitions and audit original annotations before test use."""
import hashlib
import json
import re
import time
from collections import Counter
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .audioset_intake import RemoteFile,FA,DB
from .audioset_annotation_identity import originals,display_names,match_row,video_identity

OUT=REPORT/'stage2_new_source_catalog'
MUSIC='/m/04rlf'


def main():
    import pyarrow.parquet as pq
    OUT.mkdir(exist_ok=True);plan_path=OUT/'plan.json'
    probe_path=REPORT/'audioset_new_validation_probe/mirror_partitions_metadata.json'
    probe=json.loads(probe_path.read_text())
    source={s:originals(s) for s in ['train','eval']};names=display_names()
    registry=ROOT.parent/'yamnet/data/metadata/source_registry.csv'
    old_ids=set(re.findall(r'(?:youtu\.be/|[?&]v=|youtube\.com/shorts/)([A-Za-z0-9_-]{11})',registry.read_text()))
    for r in read_csv(DATA/'round05_train_targets/inventory.csv'):
        if r.get('source_video_id'):old_ids.add(r['source_video_id'])
    if not plan_path.exists():
        write_json(plan_path,dict(repository=probe['repository'],revision=probe['revision'],
            catalog_sha256=sha(probe_path),original_metadata_sha256={s:sha(DATA/'audioset_metadata'/f'audioset_{s}_strong.tsv') for s in source},
            roles=dict(valid='selection',test='new_fresh_test'),
            old_source_video_ids_excluded=sorted(old_ids),
            new_test_excludes_all_original_google_train_video_ids=True,
            admission='Unique original source event correspondence, one target class; >=0.2 sec of target intervals. Music and generic alarm/bell ambiguity excluded from eligible-other samples before any scores.',
            selection_other_cap=32,new_test_other_cap=256,
            source_sample_order='sha256(stage2-source-20261114:video_id), independent of model scores',
            source_creator_or_foundation_pretraining_independence_certified=False,
            old_public_test_not_reused_as_new_test=True,new_model_scores_created=False,
            metadata_only=True,source_program_sha256=sha(ROOT/'pursuit/stage2_metadata_catalog.py')))
    p=json.loads(plan_path.read_text());assert sha(probe_path)==p['catalog_sha256']
    assert sha(ROOT/'pursuit/stage2_metadata_catalog.py')==p['source_program_sha256']
    ambiguous={'alarm','bell','buzzer','chime','ringing'}
    for group in sorted(probe['records'],key=lambda x:x['publisher_partition']):
        split=group['publisher_partition'];expected='eval' if split=='test' else 'train'
        folder=OUT/split;folder.mkdir(exist_ok=True)
        for i,item in enumerate(group['shard_catalog']):
            receipt=folder/f'{i:04d}.json'
            if receipt.exists():
                j=json.loads(receipt.read_text());assert j['source_sha256']==item['lfs_sha256'] and j['complete'];continue
            url=f'https://huggingface.co/datasets/{p["repository"]}/resolve/{p["revision"]}/{item["path"]}'
            remote=RemoteFile(url,item['bytes'],item['lfs_sha256'])
            rows=pq.ParquetFile(remote).read(columns=['index','raw_text']).to_pylist();results=[]
            for k,row in enumerate(rows):
                partition,video=video_identity(row['index']);reason=''
                if partition!=expected:reason='Publisher partition does not match source index'
                identity=match_row(row,source[partition],names)
                if not identity['unique_original_identity_verified']:reason='No unique original annotation identity'
                if video in old_ids:reason='Previously represented development or observed source video'
                if split=='test' and video in source['train']:reason='Source video also appears in original training metadata'
                events=source[partition][video][identity['matching_original_segments'][0]] if identity['unique_original_identity_verified'] else []
                mids={m for m,_,_ in events};labels=({'fire_alarm'} if mids&FA else set())|({'doorbell'} if mids&DB else set())
                weak=set(re.findall(r'\((/[^ )]+)\)',' '.join(row['raw_text'])))
                label='other';intervals=[]
                if len(labels)>1:reason=reason or 'Concurrent target classes need a separate multilabel protocol'
                elif labels:
                    label=next(iter(labels));target=FA if label=='fire_alarm' else DB
                    intervals=[[a,b] for m,a,b in events if m in target]
                    if sum(b-a for a,b in intervals)<.2:reason=reason or 'Very short target fragment'
                else:
                    if weak&(FA|DB):reason=reason or 'Weak/strong target disagreement; not an eligible other'
                    if MUSIC in mids:reason=reason or 'Musical segments do not have complete fine-grained annotations'
                    if {names[m].lower() for m in mids}&ambiguous:reason=reason or 'Generic alarm or bell source ambiguous'
                results.append(dict(index=row['index'],video_id=video,source_row=k,label=label,
                    original_segment_id=identity['matching_original_segments'][0] if identity['unique_original_identity_verified'] else '',
                    event_intervals=intervals,schema=identity['schema'],quarantine_reason=reason,
                    raw_text_sha256=hashlib.sha256(json.dumps(row['raw_text'],ensure_ascii=False).encode()).hexdigest()))
            write_json(receipt,dict(complete=True,source_path=item['path'],source_bytes=item['bytes'],source_sha256=item['lfs_sha256'],
                metadata_transfer_bytes=remote.fetched,metadata_only=True,audio_decoded_or_model_scored=False,rows=results))
            if i%20==0:print('STAGE2_SOURCE_METADATA_AUDITED',split,i+1,len(group['shard_catalog']),flush=True)
    output={};excluded={};valid_video_ids=set()
    for split in ['test','valid']:
        all_rows=[]
        for path in sorted((OUT/split).glob('[0-9]*.json')):
            j=json.loads(path.read_text());assert j['complete']
            for row in j['rows']:
                row=dict(row,source_path=j['source_path'],source_bytes=j['source_bytes'],source_sha256=j['source_sha256'])
                all_rows.append(row)
                if split=='valid':valid_video_ids.add(row['video_id'])
        eligible=[r for r in all_rows if not r['quarantine_reason']]
        other=sorted([r for r in eligible if r['label']=='other'],key=lambda r:hashlib.sha256(('stage2-source-20261114:'+r['video_id']).encode()).hexdigest())
        cap=p['selection_other_cap'] if split=='valid' else p['new_test_other_cap']
        output[split]=[r for r in eligible if r['label']!='other']+other[:cap]
        excluded[split]=dict(Counter(r['quarantine_reason'] for r in all_rows if r['quarantine_reason']))
    assert not ({r['video_id'] for r in output['test']}&{r['video_id'] for r in output['valid']})
    write_json(OUT/'eligible_audio_plan.json',dict(complete=True,repository=p['repository'],revision=p['revision'],
        plan_sha256=sha(plan_path),roles=p['roles'],selected=output,exclusions=excluded,
        all_publisher_valid_video_ids=sorted(valid_video_ids),
        class_counts={s:dict(Counter(r['label'] for r in rows)) for s,rows in output.items()},
        native_audio_identity_audit_still_required=True,model_scores_created=False,actual_product_targets_verified=False))
    print('STAGE2_SOURCE_METADATA_CATALOG_COMPLETE',{s:dict(Counter(r['label'] for r in rows)) for s,rows in output.items()},flush=True)


if __name__=='__main__':main()
