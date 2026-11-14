"""Recover shared ZIP member names without overwriting or duplicating audio.

Already extracted members are reused only after matching the official ZIP's
member SHA256. Colliding metadata versions are stored in split-scoped paths.
Original ZIPs, prior receipts and prior partially extracted files are retained.
"""
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path,PurePosixPath
from comparison.dataset import sha,write_csv,write_json
from .paths import ROOT,DATA,REPORT
from .realdesed_intake import download,DEST,META


def member_sha(z,info):
    h=hashlib.sha256()
    with z.open(info) as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    directory=REPORT/'realdesed_recovery';directory.mkdir(exist_ok=True);target=directory/'status.json'
    if target.exists():raise FileExistsError('Preserve completed recovery')
    marker=REPORT/'realdesed_transport/active.json'
    if marker.exists():marker.rename(marker.with_name('active_prior_terminal_'+str(time.time_ns())+'.json'))
    j=json.loads(META.read_text());items={x['key']:x for x in j['files']};all_records=[]
    for split in ['validation','train','test']:
        archive=download(items[split+'.zip']);archive_sha=sha(archive);records=[];reused=0;new=0
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():continue
                rel=PurePosixPath(info.filename)
                if rel.is_absolute() or '..' in rel.parts or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe ZIP member')
                if Path(rel.name).suffix.lower() not in {'.wav','.flac','.ogg','.mp3','.csv','.json','.txt','.md','.pdf'}:raise ValueError('Audio/metadata members only')
                expected=member_sha(z,info);legacy=DEST/info.filename
                if legacy.exists() and legacy.is_file() and sha(legacy)==expected:
                    actual=legacy;reused+=1
                else:
                    actual=DEST/'scoped'/split/info.filename;actual.parent.mkdir(parents=True,exist_ok=True)
                    if actual.exists():assert actual.is_file() and sha(actual)==expected,'Preserve conflicting partial scoped output'
                    else:
                        if shutil.disk_usage(DEST).free<info.file_size+12*2**30:raise OSError('Preserve source data and free-capacity reserve')
                        with z.open(info) as src,actual.open('xb') as dst:shutil.copyfileobj(src,dst)
                        assert sha(actual)==expected;new+=1
                records.append(dict(original_split=split,original_member=info.filename,filename=rel.name,relative_path=str(actual.relative_to(ROOT)),source_sha256=expected,
                   bytes=info.file_size,archive_sha256=archive_sha,source_record=20056072,source_native_unchanged=True))
        write_json(directory/(split+'.json'),dict(complete=True,members=len(records),records=records,reused_matching_prior_files=reused,new_scoped_files=new,original_archive_sha256=archive_sha))
        all_records+=records;print('OFFICIAL_REALDESED_SPLIT_RECOVERED',split,len(records),'reused',reused,'new_scoped',new,flush=True)
    write_csv(directory/'official_source_catalog.csv',all_records)
    native={(r['original_split'],r['filename']):r for r in all_records if r['filename'].endswith('.wav')};priority=json.loads((DATA/'realdesed_priority_plan.json').read_text());matched=[]
    for row in priority['rows']:
        key=(row['split'],row['metadata']['raw_metadata']['filename']);assert key in native
        official=native[key];assert official['source_sha256']==row['audio']['sha256'],'Mirror native bytes differ from original official release'
        matched.append(row['key'])
    write_json(target,dict(complete=True,official_members=len(all_records),native_audio_records=len(native),unique_native_audio_sha256=len({r['source_sha256'] for r in native.values()}),
        priority_mirror_native_files_verified_against_official_archive=len(matched),all_prior_files_and_archives_preserved=True,no_audio_copies_needed_when_prior_sha256_matched=True,
        metadata_collision_handling='Split-scoped files, never overwrite prior validation metadata',catalog_sha256=sha(directory/'official_source_catalog.csv'),
        actual_product_validation=False,model_training_data_not_automatically_changed=True))
    print('OFFICIAL_REALDESED_ARCHIVES_AND_PRIORITY_MEMBERS_VERIFIED',len(native),len(matched),flush=True)


if __name__=='__main__':main()
