"""Build a new source split without changing any historical data or membership."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .protocol import DATA, ROOT, REPORT, VERSION, methods


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def order(key):return hashlib.sha256(('20261114:'+str(key)).encode()).hexdigest()


def validate(rows):
    assert len({r['sample_id'] for r in rows})==len(rows),'Repeated sample ID'
    groups=defaultdict(set);recordings=defaultdict(set);digests=defaultdict(set);creators=defaultdict(set)
    for r in rows:
        groups[r['split_group_id']].add(r['role'])
        recordings[(r['source_dataset'],r['source_recording_id'])].add(r['role'])
        if r.get('source_sha256'):digests[r['source_sha256']].add(r['role'])
        if r.get('creator_group_id'):creators[r['creator_group_id']].add(r['role'])
        assert r['role'] in ('train','selection','calibration','fresh_test')
    for kind,m in [('group',groups),('recording',recordings),('hash',digests),('creator',creators)]:
        assert not any(len(v)>1 for v in m.values()),f'Cross-role {kind} overlap'
    return rows


def build():
    dest=DATA/'development.csv'
    if dest.exists():raise FileExistsError('Preserve frozen split; inspect before any revision')
    rows=read_csv(ROOT/'data/expansion_20261114/refit.csv')
    groups=defaultdict(list)
    for r in rows:groups[r['split_group_id']].append(r)
    assignments={};strata=defaultdict(list)
    for gid,rr in groups.items():
        r=rr[0]
        if r['label']=='other' and r['split']=='train' and r['sampler_pool']!='domestic_doorlock':
            assignments[gid]='train'
        elif r['label']=='other' and r['split']=='validation':
            strata[('other','foreign_validation')].append(gid)
        elif r['label']=='doorbell' and r['source_dataset']=='GongU':
            # Previously collected related catalog series stay training-only.
            assignments[gid]='train'
        else:
            domain='domestic' if r['sampler_pool'] in ('domestic_target','domestic_doorlock') else 'overseas'
            strata[(r['label'],domain)].append(gid)
    for (label,pool),gids in sorted(strata.items()):
        gids=sorted(gids,key=order)
        if pool=='foreign_validation':
            for i,gid in enumerate(gids):assignments[gid]='selection' if i%2==0 else 'calibration'
        else:
            n=len(gids);held=max(1,round(n*.15)) if n>=3 else 0
            for i,gid in enumerate(gids):assignments[gid]='selection' if i<held else 'calibration' if i<2*held else 'train'
    for r in rows:
        r['role']=assignments[r['split_group_id']];r['previous_split']=r['split']
        r['split']='train' if r['role']=='train' else 'validation'
        r['comparison_version']=VERSION
    validate(rows)
    for role in ('selection','calibration'):
        assert {r['label'] for r in rows if r['role']==role}=={'fire_alarm','doorbell','other'}
    write_csv(dest,rows)
    summary={role:{'clips':dict(Counter(r['label'] for r in rows if r['role']==role)),
                   'groups':{label:len({r['split_group_id'] for r in rows if r['role']==role and r['label']==label}) for label in ('fire_alarm','doorbell','other')}} for role in ('train','selection','calibration')}
    write_json(DATA/'split_audit.json',dict(version=VERSION,counts=summary,source_disjoint=True,historical_test_files=0,development_sha256=sha(dest),parent_sha256=sha(ROOT/'data/expansion_20261114/refit.csv')))
    write_json(REPORT/'protocol.json',methods())
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':build()
