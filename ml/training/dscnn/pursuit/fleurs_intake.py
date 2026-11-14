"""Native Korean speech confusers; preserve archives and source split records."""
import csv
import hashlib
import json
import re
import shutil
import struct
import tarfile
from collections import Counter,defaultdict
from pathlib import PurePosixPath
from comparison.dataset import sha,write_csv,write_json
from .paths import ROOT,DATA,REPORT

SOURCE=DATA/'fleurs_ko'
DEST=DATA/'fleurs_ko_audio'
KEYWORDS=re.compile('화재|경보|사이렌|초인종|알람|대피|불이 났|비상 방송')


def wav_header(path):
    with path.open('rb') as f:
        h=f.read(12);assert h[:4]==b'RIFF' and h[8:]==b'WAVE';fmt=None;audio_hash=None;data_bytes=None
        while True:
            h=f.read(8)
            if not h:break
            assert len(h)==8;kind,size=struct.unpack('<4sI',h)
            if kind==b'fmt ':
                b=f.read(size);assert len(b)>=16;fmt=struct.unpack('<HHIIHH',b[:16])
            elif kind==b'data':
                digest=hashlib.sha256();remaining=size
                while remaining:
                    b=f.read(min(2**20,remaining));assert b;digest.update(b);remaining-=len(b)
                audio_hash=digest.hexdigest();data_bytes=size
            else:f.seek(size,1)
            if size%2:f.seek(1,1)
    assert fmt is not None and data_bytes is not None
    encoding,channels,rate,_,align,bits=fmt;assert rate==16000 and channels==1 and encoding in (1,3,65534) and data_bytes%align==0
    return dict(sample_count=data_bytes//align,rate=rate,channels=channels,bits=bits,wav_format_code=encoding,native_sample_bytes_sha256=audio_hash)


def main():
    receipt=json.loads((REPORT/'hub_subsets/fleurs_ko/complete.json').read_text());assert receipt['complete'] and all(sha(SOURCE/k)==v for k,v in receipt['sha256'].items())
    target=REPORT/'fleurs_intake.json'
    if target.exists():raise FileExistsError('Preserve completed native-source intake')
    DEST.mkdir(exist_ok=True);all_rows=[]
    for original_split in ['train','dev','test']:
        table=SOURCE/f'data/ko_kr/{original_split}.tsv';metadata={}
        with table.open(encoding='utf-8',newline='') as h:
            for row in csv.reader(h,delimiter='\t',quoting=csv.QUOTE_NONE):
                assert len(row)==7 and row[1] not in metadata;metadata[row[1]]=row
        directory=DEST/original_split;directory.mkdir(exist_ok=True)
        archive=SOURCE/f'data/ko_kr/audio/{original_split}.tar.gz';archive_sha=sha(archive);seen=set()
        if shutil.disk_usage(DEST).free<archive.stat().st_size*2+26*2**30:raise OSError('Preserve remaining source-transfer capacity')
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if member.isdir():continue
                rel=PurePosixPath(member.name)
                if not member.isfile() or rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source archive member')
                if not rel.name.endswith('.wav'):raise ValueError('Native audio only; never extract/execute software')
                assert rel.name in metadata and rel.name not in seen;seen.add(rel.name);row=metadata[rel.name];p=directory/rel.name
                if p.exists():raise FileExistsError('Preserve partial extraction; inspect before resuming')
                with tar.extractfile(member) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
                info=wav_header(p);assert info['sample_count']==int(row[5])
                role='train' if original_split=='train' else 'fresh_test' if original_split=='test' else 'calibration' if int(hashlib.sha256(row[0].encode()).hexdigest()[:8],16)%2 else 'selection'
                reason='Potential spoken alarm/evacuation content; no target-absence claim' if KEYWORDS.search(row[2]) else ''
                all_rows.append(dict(sample_id='fleurs-ko-'+original_split+'-'+p.stem,source_dataset='FLEURS-ko',source_recording_id=p.stem,split_group_id='fleurs-ko-prompt-'+row[0],creator_group_id='',speaker_identity_known=False,
                  role=role,split='train' if role=='train' else 'test' if role=='fresh_test' else 'validation',original_split=original_split,label='other',label_id=2,
                  target_domain='ko_speech_confuser',sampler_pool='overseas_general',audio_root='dscnn',relative_path=str(p.relative_to(ROOT)),source_sha256=sha(p),
                  source_category='Korean ordinary read speech',prompt_id=row[0],transcription=row[2],gender_metadata=row[6],language='ko_kr',
                  source_url='https://huggingface.co/datasets/google/fleurs',license_id='CC-BY-4.0',attribution='Google FLEURS / XTREME-S; retain original dataset citation and speaker-recording conditions',
                  source_revision=receipt['revision'],archive_sha256=archive_sha,quarantine_reason=reason,human_event_timestamps=False,physical_recording_country_verified=False,**info))
        assert seen==set(metadata)
        print('FLEURS_NATIVE_SPLIT_EXTRACTED',original_split,len(seen),flush=True)
    prompt_roles=defaultdict(set);pcm_roles=defaultdict(set)
    for r in all_rows:prompt_roles[r['prompt_id']].add(r['role']);pcm_roles[r['native_sample_bytes_sha256']].add(r['role'])
    # Shared read-text prompts in dev's selection/calibration roles are related;
    # put them in a single deterministic role. Original train/test collisions
    # remain quarantined before model scoring, never quietly relabeled.
    for r in all_rows:
        roles=prompt_roles[r['prompt_id']]
        if len(roles)>1 or len(pcm_roles[r['native_sample_bytes_sha256']])>1:r['quarantine_reason']='Shared source text or exact native samples cross proposed roles'
    eligible=[r for r in all_rows if not r['quarantine_reason']];write_csv(DEST/'inventory.csv',all_rows);write_csv(DEST/'eligible.csv',eligible)
    write_json(target,dict(complete=True,raw_recordings=len(all_rows),eligible_recordings=len(eligible),quarantined=len(all_rows)-len(eligible),
        role_counts={role:dict(Counter(r['label'] for r in eligible if r['role']==role)) for role in ['train','selection','calibration','fresh_test']},
        total_native_hours=sum(r['sample_count'] for r in all_rows)/16000/3600,known_prompt_and_native_samples_checked=True,unknown_speaker_independence_certified=False,
        inventory_sha256=sha(DEST/'inventory.csv'),eligible_sha256=sha(DEST/'eligible.csv'),source_download_receipt_sha256=sha(REPORT/'hub_subsets/fleurs_ko/complete.json'),
        physical_device_or_household_month_validation=False,software_or_environment_installed=False))
    print('KOREAN_SPEECH_SOURCE_INTAKE_COMPLETE',len(all_rows),len(eligible),flush=True)


if __name__=='__main__':main()
