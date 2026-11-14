"""Read-only audit of completed FSD transfers against pinned Hub metadata."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import time
from comparison.dataset import sha, write_json
from .paths import DATA, REPORT


def main():
    plan=json.loads((DATA/'fsd_collection_plan.json').read_text());root=DATA/'fsd_audio'
    sources={r['hf_path']:r for r in plan['rows']};records={};receipts={}
    for bucket in ['targets','hard_negatives','background']:
        for path in sorted((REPORT/'transport'/bucket).glob('*_complete.json')):
            j=json.loads(path.read_text());assert j['revision']==plan['revision'] and j['exit_code']==0
            receipts[str(path)]=sha(path)
            for k,v in j['sha256'].items():
                assert k not in records and sources[k]['bucket']==bucket
                records[k]=v
    def verify(item):
        name,expected=item;path=root/name;assert path.is_file() and not path.is_symlink()
        meta=root/'.cache/huggingface/download'/(name+'.metadata');lines=meta.read_text().splitlines()
        assert lines[0]==plan['revision'] and sha(path)==expected
        etag=lines[1].strip('"')
        if len(etag)==64:assert etag==expected
        elif len(etag)==40:
            b=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==etag
        else:raise ValueError('Unrecognized pinned Hub etag')
        return dict(path=name,sha256=expected,remote_etag=etag,bytes=path.stat().st_size,
            bucket=sources[name]['bucket'],label=sources[name]['label'],role=sources[name]['role'])
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,records.items()))
    counts=dict(Counter(r['bucket'] for r in rows))
    stamp=time.time_ns();out=REPORT/'collection_audits'/('fsd_'+str(stamp)+'.json')
    write_json(out,dict(passed=True,checked_at_unix_ns=stamp,repository=plan['repository'],revision=plan['revision'],
        verified_files=len(rows),verified_bytes=sum(r['bytes'] for r in rows),counts=counts,
        snapshot_completed_receipts_sha256=receipts,records=rows,
        partial_downloads_included=False,all_internet_sources_or_entire_queue_complete_claimed=False,
        raw_audio_or_cached_metadata_modified=False,hardware_or_environment_changed=False))
    print('COMPLETED_FSD_TRANSFERS_PINNED_REMOTE_ETAG_AUDIT_PASS',len(rows),counts,out,flush=True)


if __name__=='__main__':main()
