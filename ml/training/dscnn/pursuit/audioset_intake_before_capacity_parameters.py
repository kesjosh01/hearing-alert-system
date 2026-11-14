"""Read public pinned Parquet metadata, retain only target audio in new files.

Existing pyarrow is used unchanged. Source-file bytes for a selected shard are
verified against its Hub LFS SHA256 in memory; no source or cache is deleted.
All additions are training-only research candidates, never an unseen test.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.request
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import quote
import pyarrow.parquet as pq
from comparison.dataset import write_json,sha
from .paths import DATA,REPORT

REPO='CLAPv2/audioset_strong'
DEST=DATA/'audioset_strong_targets'
OUT=REPORT/'audioset_strong_intake'
FA={'/m/012n7d','/m/012ndj','/m/01y3hg','/m/03kmc9','/m/04qvtq','/m/0c3f7m','/m/0dgbq'}
DB={'/m/03wwcy','/m/07r67yg'}


def get(url,headers=None):
    request=urllib.request.Request(url,headers=dict({'Accept-Encoding':'identity'},**(headers or {})))
    return urllib.request.urlopen(request,timeout=60)


class RemoteFile(io.RawIOBase):
    def __init__(self,url,size,digest):
        self.url=url;self.size=size;self.digest=digest;self.position=0;self.parts=[];self.fetched=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        p=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if p<0:raise ValueError('Negative remote file position')
        self.position=p;return p
    def read(self,n=-1):
        start=self.position;end=self.size if n<0 else min(self.size,start+n)
        if end<=start:return b''
        for begin,body in self.parts:
            if begin<=start and begin+len(body)>=end:
                self.position=end;return body[start-begin:end-begin]
        with get(self.url,{'Range':f'bytes={start}-{end-1}'}) as h:
            body=h.read();status=h.status;content_range=h.headers.get('Content-Range','')
        self.fetched+=len(body)
        if status==200:
            assert len(body)==self.size and hashlib.sha256(body).hexdigest()==self.digest
            self.parts=[(0,body)];self.position=end;return body[start:end]
        assert status==206 and content_range==f'bytes {start}-{end-1}/{self.size}' and len(body)==end-start,(status,content_range,start,end)
        self.parts.append((start,body));self.position=end;return body
    def full(self):
        for begin,body in self.parts:
            if begin==0 and len(body)==self.size:return body
        with get(self.url) as h:body=h.read()
        self.fetched+=len(body)
        assert len(body)==self.size and hashlib.sha256(body).hexdigest()==self.digest
        return body


def make_plan():
    plan=OUT/'plan.json'
    if plan.exists():return json.loads(plan.read_text())
    identity=json.loads((REPORT/'hub_subsets/audioset_strong_probe/identity.json').read_text())
    assert identity['id']==REPO and not identity['private'] and not identity['gated']
    revision=identity['sha'];items=[]
    url=f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/data/train?limit=1000'
    while url:
        with get(url) as h:
            rows=json.load(h);link=h.headers.get('Link','')
        for r in rows:
            if r.get('type')=='file' and r['path'].endswith('.parquet'):
                assert r['lfs']['oid'] and len(r['lfs']['oid'])==64
                items.append(dict(path=r['path'],size=r['size'],sha256=r['lfs']['oid']))
        match=re.search(r'<([^>]+)>;\s*rel="next"',link);url=match.group(1) if match else None
    assert items and len({r['path'] for r in items})==len(items)
    result=dict(repository=REPO,revision=revision,files=sorted(items,key=lambda r:r['path']),
        model_scores_used=False,admission='Training-only research candidates; foundation pretraining exposure possible',
        underlying_source_license_individually_cleared=False,new_apps_or_environment_installed=False)
    write_json(plan,result);return result


def official_events():
    rows=defaultdict(lambda:defaultdict(list))
    with (DATA/'audioset_metadata/audioset_train_strong.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            video=r['segment_id'].rsplit('_',1)[0]
            rows[video][r['segment_id']].append((float(r['start_time_seconds']),float(r['end_time_seconds']),r['label']))
    return rows


def matched_events(row,official):
    prefix='fsx_yuchenhui_audioset_strong_processed_train_';index=row['index']
    assert index.startswith(prefix);video=index[len(prefix):]
    assert re.fullmatch(r'[A-Za-z0-9_-]{11}',video)
    texts=row['raw_text'];mids=set(re.findall(r'\((/[^ )]+)\)', ' '.join(texts)))
    labels=({'fire_alarm'} if mids&FA else set())|({'doorbell'} if mids&DB else set())
    if len(labels)!=1:return None
    events=[]
    for text in texts:
        if text.startswith('Strong events: '):
            for part in text[len('Strong events: '):].split(' | '):
                m=re.fullmatch(r'\(([0-9.]+)s - ([0-9.]+)s\) .* \((/[^ )]+)\)',part)
                if m:events.append((float(m[1]),float(m[2]),m[3]))
    candidates=[]
    for segment,source in official.get(video,{}).items():
        if events and len(events)==len(source) and all(any(mid==c and abs(a-x)<=.002 and abs(b-y)<=.002 for x,y,c in source) for a,b,mid in events):
            candidates.append((segment,source))
    label=next(iter(labels));target=FA if label=='fire_alarm' else DB
    source=candidates[0][1] if len(candidates)==1 else []
    intervals=[[a,b] for a,b,mid in source if mid in target]
    return dict(video_id=video,label=label,role='train',raw_text=texts,
        official_segment_id=candidates[0][0] if len(candidates)==1 else '',
        event_intervals_json=json.dumps(intervals),human_event_timestamps=bool(intervals),
        interval_identity_verified_against_original_metadata=len(candidates)==1,
        weak_target_mid_ids=sorted(mids&target))


def main(limit):
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);plan=make_plan();official=official_events();done=0
    for i,item in enumerate(plan['files']):
        receipt=OUT/(f'{i:05d}.json')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['complete'] and old['source_sha256']==item['sha256']
            assert all(sha(DEST/r['filename'])==r['audio_sha256'] for r in old['targets']);continue
        if done>=limit:break
        if shutil.disk_usage(DEST).free < item['size']+12*2**30:raise OSError('Retain original data and free-capacity reserve')
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{plan["revision"]}/'+quote(item['path'],safe='/')
        remote=RemoteFile(url,item['size'],item['sha256']);meta=pq.ParquetFile(remote).read(columns=['index','raw_text']).to_pylist()
        matches=[(j,r,matched_events(r,official)) for j,r in enumerate(meta)]
        matches=[(j,r,m) for j,r,m in matches if m is not None]
        targets=[]
        if matches:
            body=remote.full();table=pq.read_table(io.BytesIO(body),columns=['index','audio','audio_len'])
            for j,row,m in matches:
                a=table.column('audio')[j].as_py()['bytes'];digest=hashlib.sha256(a).hexdigest()
                assert a[:4]==b'RIFF' and a[8:12]==b'WAVE';filename='audioset-'+m['video_id']+'-'+digest[:16]+'.wav';p=DEST/filename
                if p.exists():assert sha(p)==digest
                else:
                    with p.open('xb') as f:f.write(a)
                targets.append(dict(m,filename=filename,audio_sha256=digest,audio_bytes=len(a),audio_duration_s=table.column('audio_len')[j].as_py(),
                    repository=REPO,revision=plan['revision'],source_parquet=item['path'],source_row=j,source_sha256=item['sha256'],
                    source_url='https://research.google.com/audioset/download_strong.html',
                    underlying_audio_license_individually_verified=False,independent_test=False,physical_device_identity_known=False))
        write_json(receipt,dict(complete=True,repository=REPO,revision=plan['revision'],source_path=item['path'],source_sha256=item['sha256'],
            source_rows=len(meta),metadata_transfer_bytes=remote.fetched,full_source_sha256_verified=bool(matches),targets=targets))
        done+=1;print('AUDIOSET_TARGET_SHARD_COMPLETE',i,len(meta),len(targets),flush=True)
    receipts=[json.loads(p.read_text()) for p in sorted(OUT.glob('[0-9]*.json'))]
    targets=[r for q in receipts for r in q['targets']]
    write_json(OUT/'status.json',dict(complete=len(receipts)==len(plan['files']),source_shards_scanned=len(receipts),total_source_shards=len(plan['files']),
        source_rows_scanned=sum(q['source_rows'] for q in receipts),target_files=len(targets),label_counts=dict(Counter(r['label'] for r in targets)),
        published_interval_files=sum(r['human_event_timestamps'] for r in targets),
        source_full_sha256_checked_before_audio_retention=True,only_train_source_partition=True,
        fresh_test_data_created=False,model_training_data_automatically_changed=False,product_targets_verified=False))
    print('AUDIOSET_TARGET_COLLECTION_BATCH_COMPLETE',len(receipts),len(targets),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=200);a=p.parse_args();main(a.limit)
