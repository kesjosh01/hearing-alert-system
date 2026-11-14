"""Pinned exact-file data transfers using the existing, unchanged HF CLI."""
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from comparison.dataset import sha,write_json
from .paths import ROOT,DATA,REPORT
from .fsd_intake import HF


def transfer(repo,paths,tag):
    dest=DATA/tag;out=REPORT/'hub_subsets'/tag;out.mkdir(parents=True,exist_ok=True)
    identity=out/'identity.json'
    if identity.exists():info=json.loads(identity.read_text())
    else:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=45) as h:info=json.load(h)
        assert info['id']==repo and not info['private'] and not info['gated'];write_json(identity,info)
    revision=info['sha'];completed=out/'complete.json'
    if completed.exists():
        old=json.loads(completed.read_text());assert old['paths']==paths and old['revision']==revision and all(sha(dest/k)==v for k,v in old['sha256'].items());return
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
       HF_HUB_CACHE=str(ROOT/'.cache/pursuit_hf/hub'),HF_XET_CACHE=str(ROOT/'.cache/pursuit_hf/xet'))
    cmd=[str(HF),'download',repo,*paths,'--repo-type','dataset','--revision',revision,'--local-dir',str(dest),'--max-workers','2']
    dry=out/'dry.json'
    if not dry.exists():
        with dry.open('x') as f,(out/'dry.stderr').open('x') as e:result=subprocess.run(cmd+['--dry-run','--json'],stdout=f,stderr=e,env=env,cwd=ROOT)
        if result.returncode:raise RuntimeError('Dry-run failed; inspect preserved record')
    planned=json.loads(dry.read_text());assert {x['file'] for x in planned}==set(paths)
    size=0
    for x in planned:
        m=re.fullmatch(r'([0-9.]+)([KMGTP]?)',str(x['size']));assert m
        size+=float(m[1])*(1000**('KMGTP'.index(m[2])+1) if m[2] else 1)
    free=shutil.disk_usage(DATA).free;assert free>=2*size+26*2**30,'Keep space for all ongoing source transfers and original archives'
    write_json(out/'plan.json',dict(repository=repo,revision=revision,paths=paths,estimated_bytes=int(size),free_bytes=free,existing_cli_unchanged=True))
    with (out/'download.log').open('a') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    if result.returncode:raise RuntimeError('Transfer failed; partial data preserved, no automatic retry')
    checksums={}
    for name in paths:
        p=dest/name;checksums[name]=sha(p)
        remote=Path(str(dest/'.cache/huggingface/download'/name)+'.metadata').read_text().splitlines();assert remote[0]==revision
        if len(remote[1])==64:assert checksums[name]==remote[1]
    write_json(completed,dict(complete=True,repository=repo,revision=revision,paths=paths,sha256=checksums,exit_code=0))
    print('PURSUIT_PINNED_DATA_SUBSET_COMPLETE',tag,len(paths),flush=True)


if __name__=='__main__':
    files=['README.md','release.json','manifest.jsonl','metadata.parquet','provenance/source-selection.json']
    for split in ['train','validation','test']:
        files += [f'provenance/source/unpacked/{split}.zip.contents/{name}' for name in ['metadata.csv','annotations.csv','annotations_raw.csv']]
    transfer('ltuncay/chime-realdesed',files,'realdesed_mirror_index')
