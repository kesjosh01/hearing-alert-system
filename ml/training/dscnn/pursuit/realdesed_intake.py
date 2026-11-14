"""Official licensed public audio only; no setup scripts or dependency changes."""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path,PurePosixPath
from comparison.dataset import sha,write_json
from .paths import DATA,REPORT

DEST=DATA/'realdesed'
META=DATA/'metadata/realdesed_20056072.json'


def download(item):
    name=item['key'];p=DEST/name;receipt=REPORT/'realdesed_transport'/(name+'.json')
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['complete'] and sha(p)==old['sha256'];return p
    partial=DEST/(name+'.part');size=int(item['size'])
    # Leave room for the concurrent FSD collection, extracted audio and caches.
    required=2*size+26*2**30
    if shutil.disk_usage(DEST).free<required:raise OSError('Capacity reserve: preserve all data; no automatic cleanup')
    if p.exists():raise FileExistsError('Unreceipted final file requires inspection')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise ValueError('Partial exceeds official size')
    headers={'User-Agent':'hearing-alert-public-audio-research/1.0'}
    if offset:headers['Range']=f'bytes={offset}-'
    started=time.monotonic();last=offset
    write_json(REPORT/'realdesed_transport/active.json',dict(pid=os.getpid(),file=name,phase='download',expected_bytes=size,partial_bytes=offset,official_record=20056072))
    if offset<size:
        req=urllib.request.Request(item['links']['self'],headers=headers)
        with urllib.request.urlopen(req,timeout=45) as response:
            if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                raise ValueError('Range not accepted; preserve partial rather than append wrong bytes')
            with partial.open('ab' if offset else 'xb') as h:
                while True:
                    chunk=response.read(2**20)
                    if not chunk:break
                    h.write(chunk);offset+=len(chunk)
                    if offset-last>=64*2**20:
                        write_json(REPORT/'realdesed_transport/active.json',dict(pid=os.getpid(),file=name,phase='download',expected_bytes=size,partial_bytes=offset,elapsed_seconds=time.monotonic()-started))
                        print('REALDESED_BYTES',name,offset,size,flush=True);last=offset
                    if offset>size:raise ValueError('Remote response exceeds official size')
    assert partial.stat().st_size==size
    md5=hashlib.md5()
    with partial.open('rb') as h:
        for chunk in iter(lambda:h.read(2**20),b''):md5.update(chunk)
    assert 'md5:'+md5.hexdigest()==item['checksum'],'Official archive checksum mismatch'
    partial.rename(p)
    write_json(receipt,dict(complete=True,file=name,bytes=size,official_checksum=item['checksum'],sha256=sha(p),url=item['links']['self'],elapsed_seconds=time.monotonic()-started))
    print('REALDESED_DOWNLOAD_VERIFIED',name,size,flush=True);return p


def extract(p):
    receipt=REPORT/'realdesed_transport'/(p.name+'.extracted.json')
    if receipt.exists():
        old=json.loads(receipt.read_text());assert all(sha(DEST/k)==v for k,v in old['files'].items());return
    with zipfile.ZipFile(p) as z:
        members=[i for i in z.infolist() if not i.is_dir()]
        for i in members:
            rel=PurePosixPath(i.filename)
            if rel.is_absolute() or '..' in rel.parts or (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe archive member')
            if Path(i.filename).suffix.lower() not in {'.wav','.flac','.ogg','.mp3','.csv','.json','.txt','.md'}:raise ValueError('Audio/metadata files only')
        if shutil.disk_usage(DEST).free<sum(i.file_size for i in members)+26*2**30:raise OSError('Extraction capacity reserve')
        records={}
        for i in members:
            out=DEST/i.filename;out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():raise FileExistsError('Preserve partial extraction; inspect before any resume')
            with z.open(i) as src,out.open('xb') as dst:shutil.copyfileobj(src,dst)
            records[i.filename]=sha(out)
    write_json(receipt,dict(complete=True,archive_sha256=sha(p),files=records,uncompressed_bytes=sum(i.file_size for i in members)))
    print('REALDESED_EXTRACTION_VERIFIED',p.name,len(records),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('splits',nargs='+',choices=['train','validation','test']);args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);j=json.loads(META.read_text());assert j['id']==20056072 and j['metadata']['license']['id']=='cc-by-4.0' and j['metadata']['access_right']=='open'
    items={i['key']:i for i in j['files']}
    plan=REPORT/'realdesed_transport/plan.json'
    if not plan.exists():write_json(plan,dict(official_record=20056072,record_metadata_sha256=sha(META),splits=args.splits,
        files=[items[s+'.zip'] for s in args.splits],raw_audio_only=True,setup_scripts_executed=False,packages_installed=False,
        test_audio_not_used_for_model_development=True,license='Dataset CC-BY 4.0; per-file CC0 or CC-BY must also be retained'))
    try:
        for split in args.splits:extract(download(items[split+'.zip']))
        write_json(REPORT/'realdesed_transport/status.json',dict(complete=True,splits=args.splits,actual_product_validation=False))
    except Exception as e:
        write_json(REPORT/'realdesed_transport/failure.json',dict(error=f'{type(e).__name__}: {e}',partials_preserved=True,automatic_retry=False));raise


if __name__=='__main__':main()
