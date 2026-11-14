"""Expand a versioned cohort only from already verified public-audio shards."""
import argparse
import json
from collections import Counter,defaultdict
from .paths import ROOT,DATA,REPORT
from .data import confirmed_new,prepare
from comparison.dataset import read_csv,write_csv,write_json,sha


def freeze(name,hard,background):
    out=DATA/name
    if out.exists():raise FileExistsError('Do not overwrite an existing data boundary')
    receipt_paths=[];selected=set()
    for bucket,count in [('hard_negatives',hard),('background',background)]:
        for i in range(count):
            p=REPORT/'transport'/bucket/f'{i:04d}_complete.json';j=json.loads(p.read_text());selected.update(j['paths']);receipt_paths.append(p)
    new=[r for r in confirmed_new() if r['bucket']=='targets' or r['hf_path'] in selected]
    previous=read_csv(DATA/'round01/inventory.csv')
    old=[r for r in previous if r.get('source_dataset')!='FSD50K']
    # The source dataset identifier is explicitly checked rather than inferred
    # from a filename or from a directory name.
    new_ids={r['sample_id'] for r in new}
    old=[r for r in old if r['sample_id'] not in new_ids]
    rows=old+new;ids=set();groups=defaultdict(set);creators=defaultdict(set);digests=defaultdict(set)
    for r in rows:
        assert r['sample_id'] not in ids;ids.add(r['sample_id']);groups[r['split_group_id']].add(r['role'])
        if r.get('creator_group_id'):creators[r['creator_group_id']].add(r['role'])
        if r.get('source_sha256'):digests[r['source_sha256']].add(r['role'])
    assert all(len(v)==1 for m in [groups,creators,digests] for v in m.values())
    previous_ids={r['sample_id'] for r in previous};assert previous_ids.issubset(ids),'Expansion must preserve all prior examples'
    out.mkdir();write_csv(out/'inventory.csv',rows);write_csv(out/'development.csv',[r for r in rows if r['role']!='fresh_test'])
    write_csv(out/'fresh_test.csv',[r for r in rows if r['role']=='fresh_test'])
    counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration','fresh_test']}
    write_json(out/'freeze.json',dict(name=name,files={k:sha(out/k) for k in ['inventory.csv','development.csv','fresh_test.csv']},counts=counts,
      previous_round='round01',previous_round_files_preserved=True,new_fsd_files=len(new),hard_negative_shards=hard,background_shards=background,
      transport_receipt_hashes={str(p.relative_to(ROOT)):sha(p) for p in receipt_paths},
      model_scores_used_to_choose_new_examples=False,fresh_test_features_or_model_scores_created=False,
      labels='Weak recording labels, no human event onsets',independence='Known recording, raw bytes and FSD uploader families; physical-device independence and pretraining exposure not certified'))
    print('PURSUIT_EXPANDED_ROUND_FROZEN',name,counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare']);p.add_argument('name');p.add_argument('--hard',type=int,default=13);p.add_argument('--background',type=int,default=10);a=p.parse_args()
    freeze(a.name,a.hard,a.background) if a.mode=='freeze' else prepare(a.name)
