"""Keep the same 43 KR targets; add already verified household/speech negatives."""
import json
from collections import Counter,defaultdict
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import DATA,REPORT
from .waveform_identity import identity

NAME='domestic_seen_enriched_v1'


def main():
    out=DATA/NAME
    if out.exists():raise FileExistsError('Preserve the known-source branch boundary')
    parent=DATA/'domestic_seen_v1';general=DATA/'round04_enriched'
    for d in [parent,general]:
        j=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in j['files'].items())
    positives=read_csv(parent/'known_positive_probes.csv');assert len(positives)==43 and all(r['role']=='train' for r in positives)
    old=read_csv(parent/'development.csv');negative={r['sample_id']:r for r in read_csv(general/'development.csv') if r['label']=='other'}
    for r in old:
        if r['label']=='other':
            if r['sample_id'] in negative:assert negative[r['sample_id']]['role']==r['role']
            else:negative[r['sample_id']]=r
    # Source-identity decisions are made before model fitting, without scores.
    positive_pcm={identity(r)['decoded_pcm_sha256'] for r in positives};rejected=[];accepted=[]
    for r in negative.values():
        if identity(r)['decoded_pcm_sha256'] in positive_pcm:
            rejected.append(dict(sample_id=r['sample_id'],reason='Exact canonical signal also has an approved positive label; quarantine other admission'))
        else:accepted.append(r)
    rows=positives+accepted;assert len({r['sample_id'] for r in rows})==len(rows)
    groups=defaultdict(set);creators=defaultdict(set);pcm=defaultdict(set)
    for r in rows:
        groups[r['split_group_id']].add(r['role'])
        if r.get('creator_group_id'):creators[r['creator_group_id']].add(r['role'])
        pcm[identity(r)['decoded_pcm_sha256']].add(r['role'])
    assert all(len(v)==1 for m in [groups,creators,pcm] for v in m.values())
    out.mkdir();write_csv(out/'inventory.csv',rows);write_csv(out/'development.csv',rows);write_csv(out/'known_positive_probes.csv',positives)
    counts={role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ['train','selection','calibration']}
    write_json(out/'freeze.json',dict(name=NAME,files={k:sha(out/k) for k in ['inventory.csv','development.csv','known_positive_probes.csv']},
        counts=counts,original_positive_source_count=43,independent_positive_test_files=0,
        target_scope='Intentionally training-seen KR sources; same original and three waveform conditions as previous branch',
        negative_addition='Already source-audited R04 negatives, including Korean FLEURS read speech and RealDESED confusers',
        parent_freeze_sha256=sha(parent/'freeze.json'),general_negative_parent_sha256=sha(general/'freeze.json'),
        quarantine=rejected,no_new_public_fresh_test_used=True,original_audio_and_other_boundaries_changed=False,
        original_positive_files_preserved=True,actual_product_targets_verified=False))
    print('DOMESTIC_KNOWN_TARGETS_WITH_ENRICHED_NEGATIVES_FROZEN',counts,flush=True)


if __name__=='__main__':main()
