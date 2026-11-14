"""Hash-only canonical PCM audit, including unopened evaluation recordings.

No fresh-test features, listening previews or model scores are generated.
The purpose is source identity: a recording duplicated in development cannot
count as an independent held-out example, regardless of its class or score.
"""
import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from comparison.dataset import read_csv,write_json,write_csv,sha
from .paths import ROOT,DATA,REPORT
from .data import audio_path,CACHE

DEST=DATA/'canonical_pcm_identity_v1'


def identity(row):
    path=audio_path(row);source=sha(path)
    if row.get('source_sha256'):assert source==row['source_sha256']
    receipt=DEST/(source+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==source;return r
    old=CACHE/source/'metadata.json'
    if old.exists():
        r=json.loads(old.read_text());assert r['source_sha256']==source
        result=dict(source_sha256=source,decoded_pcm_sha256=r['decoded_pcm_sha256'],sample_count=r['sample_count'],
            source='Existing canonical native preparation receipt',fresh_features_created=False)
    else:
        pcm=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-ac','1','-ar','16000','-c:a','pcm_s16le','-f','s16le','-'],stderr=subprocess.PIPE)
        assert len(pcm)%2==0 and len(pcm)>0
        result=dict(source_sha256=source,decoded_pcm_sha256=hashlib.sha256(pcm).hexdigest(),sample_count=len(pcm)//2,
            source='Hash-only canonical PCM stream; no waveform or model features retained',fresh_features_created=False)
    # This function is called once per unique source container by the caller.
    write_json(receipt,result);return result


def main(name):
    directory=DATA/name;freeze=json.loads((directory/'freeze.json').read_text())
    assert all(sha(directory/k)==v for k,v in freeze['files'].items())
    target=REPORT/(name+'_waveform_identity_audit.json')
    if target.exists():raise FileExistsError('Preserve completed source audit')
    rows=read_csv(directory/'inventory.csv');DEST.mkdir(exist_ok=True)
    # Establish that direct hash-only decoding exactly matches the original
    # cached waveform convention before inspecting any held-out source.
    parity=[]
    for row in [r for r in rows if r['role']!='fresh_test'][:3]:
        source=sha(audio_path(row));old=json.loads((CACHE/source/'metadata.json').read_text())
        pcm=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(audio_path(row)),'-ac','1','-ar','16000','-c:a','pcm_s16le','-f','s16le','-'],stderr=subprocess.PIPE)
        assert hashlib.sha256(pcm).hexdigest()==old['decoded_pcm_sha256'];parity.append(row['sample_id'])
    unique={str(audio_path(r)):r for r in rows};records={}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,(row,r) in enumerate(zip(unique.values(),executor.map(identity,unique.values())),1):
            records[str(audio_path(row))]=r
            if i%500==0:print('CANONICAL_PCM_SOURCE_AUDIT',i,len(unique),flush=True)
    groups=defaultdict(list)
    for row in rows:groups[records[str(audio_path(row))]['decoded_pcm_sha256']].append(row)
    crossing=[];contaminated=set()
    for digest,rr in groups.items():
        if len({r['role'] for r in rr})>1:
            crossing.append(dict(decoded_pcm_sha256=digest,members=[dict(sample_id=r['sample_id'],role=r['role'],label=r['label']) for r in rr]))
            if any(r['role']!='fresh_test' for r in rr):contaminated.update(r['sample_id'] for r in rr if r['role']=='fresh_test')
    fresh=[r for r in rows if r['role']=='fresh_test'];eligible=[r for r in fresh if r['sample_id'] not in contaminated]
    out=DATA/(name+'_identity_audit');out.mkdir(exist_ok=False)
    write_csv(out/'eligible_fresh_recordings.csv',eligible)
    write_csv(out/'quarantined_fresh_recordings.csv',[dict(r,exclusion_reason='Exact canonical PCM also in development; identity audit before any fresh model inference') for r in fresh if r['sample_id'] in contaminated])
    write_json(target,dict(complete=True,source_files=len(rows),canonical_pcm_families=len(groups),
        cross_role_exact_pcm_clusters=crossing,fresh_recordings=len(fresh),eligible_fresh_recordings=len(eligible),
        excluded_fresh_recordings=len(contaminated),canonical_decoding_parity_sample_ids=parity,
        hash_only=True,fresh_features_created=False,fresh_model_inferences=0,
        quarantine_based_on_identity_not_model_scores=True,original_manifests_or_audio_modified=False,
        physical_device_or_creator_independence_certified=False,
        eligible_manifest_sha256=sha(out/'eligible_fresh_recordings.csv'),original_freeze_sha256=sha(directory/'freeze.json')))
    print('CANONICAL_PCM_SOURCE_AUDIT_COMPLETE',len(rows),'fresh',len(fresh),'identity_quarantined',len(contaminated),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();main(a.name)
