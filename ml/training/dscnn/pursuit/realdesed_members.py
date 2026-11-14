"""Prioritize source-native labeled records using public TAR byte ranges.

Only standard public HTTP 206 responses are accepted. Every complete native
audio member must match the SHA256 in the pinned, verified source manifest.
This is not a bypass of a download restriction or a substitute for the ongoing
official full-archive collection.
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from collections import Counter,defaultdict
from pathlib import Path
from comparison.dataset import read_csv,sha,write_json
from .paths import DATA,REPORT

INDEX=DATA/'realdesed_mirror_index'
DEST=DATA/'realdesed_members'
PLAN=DATA/'realdesed_priority_plan.json'


def order(x):return hashlib.sha256(('pursuit-realdesed-before-scores:'+x).encode()).hexdigest()


def plan():
    if PLAN.exists():raise FileExistsError('Preserve participant roles and pre-score selection')
    receipt=json.loads((REPORT/'hub_subsets/realdesed_mirror_index/complete.json').read_text())
    assert all(sha(INDEX/k)==v for k,v in receipt['sha256'].items())
    release=json.loads((INDEX/'release.json').read_text());assert sha(INDEX/'manifest.jsonl')==release['manifest']['sha256']
    with (INDEX/'manifest.jsonl').open() as h:rows=[json.loads(l) for l in h]
    grouped=defaultdict(set)
    for r in rows:grouped[r['metadata']['group']].add(r['split'])
    roles={}
    for group,splits in grouped.items():
        if 'test' in splits:role='fresh_test'
        elif 'validation' in splits:role='selection' if int(order(group)[:8],16)%2==0 else 'calibration'
        else:role='train'
        roles[group]=role
    selected=[];hard=[];confusers={'phone_ringing','microwave','keychain','door_open_close','cutlery_dishes','light_switch'}
    for r in rows:
        m=r['metadata'];events=m['targets']['events'];bell=[e for e in events if 'bell_ringing' in e['labels']]
        if bell:label='doorbell';bucket='targets'
        elif confusers.intersection(m['raw_metadata']['target_classes'].split(';')):label='other';bucket='confusers'
        else:continue
        license=m['license_name'];assert 'CC0' in license or 'CC BY' in license or 'CC-BY' in license
        x=dict(r,role=roles[m['group']],label=label,bucket=bucket,human_bell_events=bell)
        (selected if bell else hard).append(x)
    # Keep a bounded priority confuser set; the official collection continues
    # for the complete corpus. No model scores determine this subset.
    hard.sort(key=lambda r:order(r['key']));seen=set();priority=[]
    for r in hard:
        group=r['metadata']['group']
        if group in seen:continue
        priority.append(r);seen.add(group)
        if len(priority)>=256:break
    selected.sort(key=lambda r:order(r['key']));selected+=priority
    write_json(PLAN,dict(repository=receipt['repository'],revision=receipt['revision'],rows=selected,role_counts={role:dict(Counter(r['label'] for r in selected if r['role']==role)) for role in set(roles.values())},
       source_manifest_sha256=sha(INDEX/'manifest.jsonl'),metadata_receipt_sha256=sha(REPORT/'hub_subsets/realdesed_mirror_index/complete.json'),
       participant_group_role_disjoint=True,model_scores_used=False,official_archive_collection_continues=True,
       bell_definition='Official annotation guidelines: sound emitted by a doorbell system',unknown_fire_events_not_strongly_annotated=True))
    print('REALDESED_PRIORITY_SOURCE_PLAN_FROZEN',len(selected),'files',sum(len(r['human_bell_events']) for r in selected),'doorbell events',flush=True)


def transfer():
    j=json.loads(PLAN.read_text());DEST.mkdir(parents=True,exist_ok=True);receipts=[];out=REPORT/'realdesed_priority';out.mkdir(exist_ok=True)
    for i,r in enumerate(j['rows'],1):
        a=r['audio'];p=DEST/(r['key']+'.wav');receipt=p.with_suffix('.json')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['audio_sha256']==a['sha256'] and sha(p)==a['sha256'];receipts.append(old);continue
        if p.exists():raise FileExistsError('Unreceipted native record must be inspected')
        assert shutil.disk_usage(DEST).free>26*2**30,'Preserve capacity for the official source collection'
        start=r['audio_offset'];end=start+a['stored_bytes']-1
        url=f"https://huggingface.co/datasets/{j['repository']}/resolve/{j['revision']}/{r['shard']}"
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'hearing-alert-public-audio-research/1.0'})
        partial=p.with_suffix('.wav.part')
        if partial.exists():raise FileExistsError('Partial range fetch preserved; inspect before retry')
        try:
            with urllib.request.urlopen(request,timeout=45) as h:
                if h.status!=206 or not h.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('Server did not honor ordinary public byte range; no full-file fallback or bypass')
                with partial.open('xb') as f:
                    remaining=a['stored_bytes']
                    while remaining:
                        b=h.read(min(2**20,remaining))
                        if not b:raise EOFError('Incomplete range; preserve partial')
                        f.write(b);remaining-=len(b)
            assert partial.stat().st_size==a['stored_bytes'] and sha(partial)==a['sha256'],'Pinned manifest native audio checksum mismatch'
            partial.rename(p)
            old=dict(key=r['key'],role=r['role'],audio_sha256=a['sha256'],stored_bytes=a['stored_bytes'],repository=j['repository'],revision=j['revision'],shard=r['shard'],byte_range=[start,end],source_native_unchanged=True)
            write_json(receipt,old);receipts.append(old)
            write_json(out/'status.json',dict(complete=len(receipts)==len(j['rows']),files=len(receipts),planned_files=len(j['rows']),bytes=sum(x['stored_bytes'] for x in receipts)))
            print('REALDESED_NATIVE_MEMBER_VERIFIED',i,len(j['rows']),r['role'],r['label'],flush=True)
            time.sleep(2)
        except Exception as e:
            write_json(out/'first_failure.json',dict(error=f'{type(e).__name__}: {e}',key=r['key'],partials_preserved=True,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','transfer']);a=p.parse_args();plan() if a.mode=='plan' else transfer()
