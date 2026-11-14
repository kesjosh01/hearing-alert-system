"""Freeze transport-successful sources, never consulting model scores."""
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path
from .dataset import DATA,ROOT,REPORT,read_csv,write_csv,write_json,sha,validate


def main():
    if (DATA/'fresh_test.csv').exists():raise FileExistsError('Never silently change final-test membership')
    dest=DATA/'new_public';new=[]
    for name in ('downloads.json','soundbible_downloads.json','bsb_downloads.json'):
        p=dest/name
        if p.exists():new+=json.loads(p.read_text())
    old=read_csv(ROOT/'data/expansion_20261114/refit.csv')
    # Historical metadata only; no model predictions or new test features.
    oldmeta=ROOT.parent/'yamnet/data/metadata'
    oldkeys={(r['source_dataset'],str(int(r['source_recording_id'])) if r['source_recording_id'].isdigit() else r['source_recording_id']) for r in old}
    oldhash={};oldpcm={}
    for p in (ROOT/'.cache/features').glob('*/metadata.json'):
        j=json.loads(p.read_text());oldhash[j['source_sha256']]=str(p)
    for base in (ROOT/'.cache/features',ROOT/'.cache/historical_holdout'):
        for p in base.rglob('audio.wav'):
            with wave.open(str(p)) as h:b=h.readframes(h.getnframes())
            oldpcm[hashlib.sha256(b).hexdigest()]=str(p)
    accepted=[];excluded=[];newhash={};newpcm={}
    for r in sorted(new,key=lambda r:(r['role']!='train',r['sample_id'])):
        audio=ROOT/r['relative_path']
        if sha(audio)!=r['source_sha256']:raise ValueError('Downloaded public bytes changed')
        key=(r['source_dataset'],str(int(r['source_recording_id'])) if r['source_recording_id'].isdigit() else r['source_recording_id'])
        reason='known recording ID' if key in oldkeys else 'exact source bytes' if r['source_sha256'] in oldhash else 'exact decoded PCM' if r['decoded_pcm_sha256'] in oldpcm else 'new-source duplicate' if r['source_sha256'] in newhash or r['decoded_pcm_sha256'] in newpcm else None
        if reason:excluded.append(dict(id=r['sample_id'],reason=reason));continue
        newhash[r['source_sha256']]=r['sample_id'];newpcm[r['decoded_pcm_sha256']]=r['sample_id']
        r.update(original_label=r['label'],comparison_version='source-disjoint-sequential-comparison-20261114-v1')
        accepted.append(r)
    for r in accepted:
        # Field lengths are plain publisher metadata, not executable commands.
        assert r['role'] in ('train','fresh_test')
    validate(read_csv(DATA/'development.csv')+accepted)
    training=[r for r in accepted if r['role']=='train'];test=[r for r in accepted if r['role']=='fresh_test']
    assert {r['label'] for r in test}=={'fire_alarm','doorbell','other'},'Fresh test needs all three classes'
    write_csv(DATA/'new_training.csv',training);write_csv(DATA/'fresh_test.csv',test)
    write_json(dest/'accepted.json',accepted);write_json(dest/'excluded_duplicates.json',excluded)
    counts={role:{lab:dict(clips=sum(r['label']==lab for r in accepted if r['role']==role),groups=len({r['split_group_id'] for r in accepted if r['role']==role and r['label']==lab})) for lab in ('fire_alarm','doorbell','other')} for role in ('train','fresh_test')}
    freeze=dict(counts=counts,files={name:sha(DATA/name) for name in ('development.csv','new_training.csv','fresh_test.csv')},
                sources_sha256=sha(dest/'accepted.json'),source_overlap_checks=dict(recording_id=True,exact_bytes=True,exact_decoded_pcm=True,creator_or_device_series_roles=True),
                excluded=excluded,new_test_model_inferences=0,no_model_scores_used_for_inclusion=True,
                foundation_pretraining_exposure_unknown=True,physical_device_country_verified=False,human_listening_verified=False,
                independence_limit='New publisher recording identities and conservative related-source groups; all physical device/session reuse or edited excerpts cannot be certified from public metadata.')
    write_json(DATA/'source_freeze.json',freeze)
    text=['# Public audio credits','', 'Original public files preserved. Converted copies are used for local model research. Creator descriptions provide weak file labels; no human event annotations or verified Korean physical-device provenance is claimed.','']
    for r in accepted:text.append(f"- {r['sample_id']}: {r['title']} — {r.get('author','see source')}; {r['license_id']}; {r['source_url']}; role={r['role']}")
    (DATA/'PUBLIC_DATA_CREDITS.md').write_text('\n'.join(text)+'\n')
    print('NEW_SOURCES_FROZEN',json.dumps(counts),'excluded',len(excluded),flush=True)


if __name__=='__main__':main()
