"""Source-aware FSD50K intake using licensed audio, human labels and metadata.

The full collection queue is separate from resumable transport batches. This
does not claim that every public sound on the Internet has been collected.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter,defaultdict
from pathlib import Path
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT

HF=Path('/home/jhh0415/miniconda3/envs/lerobot_so101/bin/hf')
REPOSITORY='Fhrozen/FSD50k'
SIRENS={'Siren','Police_car_(siren)','Ambulance_(siren)','Fire_engine_and_fire_truck_(siren)','Civil_defense_siren'}
HARD={'Alarm_clock','Church_bell','Bell','Bicycle_bell','Cowbell','Jingle_bell','Telephone_bell_ringing','Ringtone',
      'Beep_and_bleep','Buzzer','Creak','Squeak','Frog','Crying_and_sobbing','Baby_cry_and_infant_cry','Door',
      'Microwave_oven','Fireworks','Car_alarm','Smoke_detector_and_smoke_alarm'}
CONTEXT={'Speech','Male_speech_and_man_speaking','Female_speech_and_woman_speaking','Child_speech_and_kid_speaking',
         'Speech_synthesizer','Hubbub_and_speech_noise_and_speech_babble','Television','Vacuum_cleaner','Washing_machine',
         'Water','Water_tap_and_faucet','Rain','Wind','Dog','Bark','Cat','Meow','Dishes_and_pots_and_pans',
         'Cutlery_and_silverware','Frying_(food)','Electric_shaver_and_electric_razor','Blender','Domestic_sounds_and_home_sounds',
         'Bird','Traffic_noise_and_roadway_noise','Engine','Printer','Computer_keyboard','Typing','Music'}


def rank(value):return hashlib.sha256(('pursuit-source-role-20261114:'+value).encode()).hexdigest()


def plan():
    destination=DATA/'fsd_collection_plan.json'
    if destination.exists():raise FileExistsError('Preserve frozen queue; do not reassign roles after scores')
    meta=DATA/'metadata';info={};ground={};collection={};original={}
    for split in ('dev','eval'):
        ii=json.loads((meta/f'FSD50K.metadata/{split}_clips_info_FSD50K.json').read_text());info.update(ii)
        for r in read_csv(meta/f'FSD50K.ground_truth/{split}.csv'):
            ground[r['fname']]=set(r['labels'].split(','));original[r['fname']]=split
        for r in read_csv(meta/f'FSD50K.metadata/collection/collection_{split}.csv'):
            collection[r['fname']]=set(r['labels'].split(','))
    old=read_csv(ROOT.parent/'yamnet/data/metadata/esc50_manifest.csv')
    seen={r['source_recording_id'] for r in old}
    seen_authors={info[s]['uploader'] for s in seen if s in info}
    accepted=[];quarantine=[];excluded=[];authors={}
    for sid,i in info.items():
        author=i['uploader'];source=original[sid];labels=ground[sid]|collection.get(sid,set());fine=collection.get(sid,set())
        title=re.sub(r'[_\-.]+',' ',i['title'].lower());tags=' '.join(i.get('tags',[])).lower();description=i.get('description','')
        license_url=i['license'];licensed=('/licenses/by/' in license_url or '/publicdomain/zero/' in license_url)
        if sid in seen:
            excluded.append(dict(freesound_id=sid,reason='Same original Freesound recording already in ESC-50'));continue
        fire=bool(labels&SIRENS or 'Fire_alarm' in fine or 'Smoke_detector_and_smoke_alarm' in fine)
        bell='Doorbell' in labels
        # FSD50K merged tiny classes into Alarm. Accept explicit creator titles
        # with that human Alarm parent; do not pretend these are event intervals.
        metadata_fire=bool('Alarm' in labels and re.search(r'\b(fire\s*alarm|smoke\s*(alarm|detector)|feueralarm|brandalarm)\b',title+' '+tags))
        metadata_bell=bool(labels&{'Alarm','Bell','Buzzer','Domestic_sounds_and_home_sounds'} and re.search(r'\b(door\s*bell|door\s*chime|türklingel)\b',title+' '+tags))
        fire=fire or metadata_fire;bell=bell or metadata_bell
        if fire and bell:
            quarantine.append(dict(freesound_id=sid,title=i['title'],reason='Multiple target classes; reserve for future multilabel policy'));continue
        if fire:label='fire_alarm';bucket='targets';basis='Human fine class or explicit creator fire/smoke title with human Alarm parent'
        elif bell:label='doorbell';bucket='targets';basis='Human Doorbell class or explicit creator doorbell title with compatible human parent'
        elif ('Alarm' in labels and not fine&{'Alarm_clock','Car_alarm'}) or 'Buzzer' in fine:
            quarantine.append(dict(freesound_id=sid,title=i['title'],reason='Ambiguous Alarm/Buzzer is never automatically a negative'));continue
        elif labels&HARD:label='other';bucket='hard_negatives';basis='Human non-target confuser class, no known target labels'
        elif labels&CONTEXT:label='other';bucket='background';basis='Human context class, no known target/ambiguous alarm labels'
        else:continue
        if not licensed:
            excluded.append(dict(freesound_id=sid,reason='CC-BY-NC or Sampling+ held out from product-directed intake',license_url=license_url));continue
        if author not in authors:
            if source=='eval' and author not in seen_authors:role='fresh_test'
            else:
                slot=int(rank(author)[:8],16)%100;role='selection' if slot<10 else 'calibration' if slot<20 else 'train'
            authors[author]=role
        role=authors[author]
        accepted.append(dict(sample_id='fsd50k-'+sid,freesound_id=sid,source_dataset='FSD50K',source_recording_id=sid,
          split_group_id='fsd50k-uploader-'+hashlib.sha256(author.encode()).hexdigest()[:16]+'-'+label,
          creator_group_id='fsd50k-uploader-'+hashlib.sha256(author.encode()).hexdigest()[:16],uploader=author,
          role=role,split='train' if role=='train' else 'test' if role=='fresh_test' else 'validation',label=label,
          label_id={'fire_alarm':0,'doorbell':1,'other':2}[label],target_domain='general',
          sampler_pool='auxiliary_target' if label!='other' else 'overseas_confuser' if bucket=='hard_negatives' else 'overseas_general',
          source_category=','.join(sorted(fine or labels)),bucket=bucket,hf_path=f'clips/{source}/{sid}.wav',
          source_url=f'https://freesound.org/s/{sid}/',license_id='CC0-1.0' if '/zero/' in license_url else 'CC-BY-3.0',
          license_url=license_url,dataset_license='CC-BY-4.0',title=i['title'],description=description,tags=i.get('tags',[]),
          labels=sorted(labels),fine_labels=sorted(fine),label_basis=basis,human_event_timestamps=False,
          creator_description_not_new_human_listening=True,physical_device_country_verified=False,
          source_identity_previously_used=False,original_fsd_partition=source))
    roles=defaultdict(set)
    for r in accepted:roles[r['creator_group_id']].add(r['role'])
    assert all(len(v)==1 for v in roles.values())
    accepted.sort(key=lambda r:({'targets':0,'hard_negatives':1,'background':2}[r['bucket']],rank(r['sample_id'])))
    identity=json.loads((meta/'hf_repo_identity.json').read_text())
    output=dict(repository=REPOSITORY,revision=identity['sha'],metadata_receipt_sha256=sha(meta/'download_receipts.json'),
       rows=accepted,counts=dict(Counter(r['bucket'] for r in accepted)),role_counts={role:dict(Counter(r['label'] for r in accepted if r['role']==role)) for role in ['train','selection','calibration','fresh_test']},
       excluded=excluded,quarantine=quarantine,source_assignment_before_new_model_scores=True,old_esc_recording_ids_removed=True,
       remaining_collection='Other public catalogs and later background rounds remain open; this queue is not an Internet completeness claim.')
    write_json(destination,output);write_json(REPORT/'fsd_intake_summary.json',{k:v for k,v in output.items() if k!='rows'})
    print('FSD_QUEUE_FROZEN',output['counts'],output['role_counts'],flush=True)


def transfer(bucket):
    plan=json.loads((DATA/'fsd_collection_plan.json').read_text());rr=[r for r in plan['rows'] if r['bucket']==bucket]
    dest=DATA/'fsd_audio';folder=REPORT/'transport'/bucket;folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1',
             HF_HUB_CACHE=str(ROOT/'.cache/pursuit_hf/hub'),HF_XET_CACHE=str(ROOT/'.cache/pursuit_hf/xet'),HF_HUB_DISABLE_XET='1')
    spec=json.loads((DATA/'metadata/hf_repo_identity.json').read_text());assert spec['sha']==plan['revision'] and HF.is_file()
    # Modest shards keep command size and resolver bursts bounded. A future
    # resume reuses each exact shard and destination; no second owner may run.
    lock=folder/'active.json'
    if lock.exists():raise RuntimeError('Existing transport owner must be checked before resuming')
    write_json(lock,dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],bucket=bucket))
    completed=False
    try:
        for index,start in enumerate(range(0,len(rr),128)):
            chunk=rr[start:start+128];receipt=folder/f'{index:04d}_complete.json'
            if receipt.exists():
                recorded=json.loads(receipt.read_text());assert recorded['paths']==[r['hf_path'] for r in chunk]
                assert all(sha(dest/k)==v for k,v in recorded['sha256'].items());continue
            base=[str(HF),'download',REPOSITORY,*[r['hf_path'] for r in chunk],'--repo-type','dataset','--revision',plan['revision'],
                  '--local-dir',str(dest),'--max-workers','2']
            dry=folder/f'{index:04d}_dry.json';dryerr=folder/f'{index:04d}_dry.stderr'
            # Failed/partial dry-run records are retained, not overwritten.
            if dry.exists():
                planned=json.loads(dry.read_text())
            else:
                with dry.open('x') as f,dryerr.open('x') as e:
                    result=subprocess.run(base+['--dry-run','--json'],stdout=f,stderr=e,env=env,cwd=ROOT)
                if result.returncode:raise RuntimeError(f'First HF dry-run failure: {dryerr}')
                planned=json.loads(dry.read_text())
            free=shutil.disk_usage(DATA).free
            assert isinstance(planned,list) and {x['file'] for x in planned}=={r['hf_path'] for r in chunk}
            estimated=0
            for item in planned:
                match=re.fullmatch(r'([0-9.]+)([KMGTP]?)',str(item['size']))
                if not match:raise ValueError('Unrecognized HF dry-run size; inspect before transport')
                estimated+=float(match[1])*(1000**('KMGTP'.index(match[2])+1) if match[2] else 1)
            if free<estimated*2+12_000_000_000:raise RuntimeError('Preserve free capacity and partial downloads; do not clean or delete automatically')
            write_json(folder/f'{index:04d}_capacity.json',dict(free_bytes=free,estimated_shard_bytes=int(estimated),temporary_overhead_factor=2,reserve_bytes=12_000_000_000,existing_environment_only=True,
                       repository=REPOSITORY,revision=plan['revision'],exact_paths=[r['hf_path'] for r in chunk],dry_run_parsed=True))
            log=folder/f'{index:04d}_download.log'
            with log.open('a') as f:
                result=subprocess.run(base,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            if result.returncode:raise RuntimeError(f'First HF transfer failure: {log}; inspect same owner/partials before resuming')
            digests={r['hf_path']:sha(dest/r['hf_path']) for r in chunk}
            write_json(receipt,dict(paths=[r['hf_path'] for r in chunk],sha256=digests,revision=plan['revision'],exit_code=0))
            print('FSD_AUDIO_SHARD_VERIFIED',bucket,index,len(chunk),sum((dest/r['hf_path']).stat().st_size for r in chunk),flush=True)
            time.sleep(2.)
        completed=True
        write_json(folder/'status.json',dict(complete=True,bucket=bucket,files=len(rr),revision=plan['revision']))
    finally:
        # Keep every owner receipt append-only. Removing this tiny ownership
        # marker after terminal execution is not deletion of source data.
        if lock.exists():lock.rename(folder/f'owner_{time.time_ns()}_terminal.json')
        if not completed:write_json(folder/f'failure_{time.time_ns()}.json',dict(complete=False,pid=os.getpid(),automatic_retry=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','targets','hard_negatives','background']);a=p.parse_args()
    plan() if a.mode=='plan' else transfer(a.mode)
