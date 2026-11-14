"""Prepare an exact, read-only cache scope. No deletion capability is included."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import wave
from comparison.dataset import sha, write_json
from .paths import ROOT, REPORT
from .data import CACHE as NATIVE
from .yamnet_frontend import CACHE, CONTRACT


def main():
    out=REPORT/'cache_release_review';out.mkdir(exist_ok=False)
    root=CACHE.absolute();assert root==root.resolve() and root.is_dir() and not root.is_symlink()
    protected=json.loads((REPORT/'protected_previous_comparison.json').read_text())
    source_gate=REPORT/'yamnet_frontend64_gate.json';assert json.loads(source_gate.read_text())['passed']
    candidates=sorted(root.glob('*/features.npy'));assert candidates
    def inspect(p):
        s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_dev==root.stat().st_dev
        assert not p.parent.is_symlink() and re.fullmatch('[0-9a-f]{64}',p.parent.name)
        assert str(p) not in protected
        meta=p.parent/'metadata.json';m=json.loads(meta.read_text())
        assert m['source_sha256']==p.parent.name and m['contract']==CONTRACT
        assert m['source_frontend_gate_sha256']==sha(source_gate) and sha(p)==m['feature_sha256']
        raw=NATIVE/p.parent.name/'audio.wav';native_meta=NATIVE/p.parent.name/'metadata.json'
        n=json.loads(native_meta.read_text());assert n['source_sha256']==m['source_sha256']
        assert n['decoded_pcm_sha256']==m['decoded_pcm_sha256']
        with wave.open(str(raw)) as w:
            assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
            assert w.getnframes()==m['sample_count']
            pcm=w.readframes(w.getnframes())
        assert hashlib.sha256(pcm).hexdigest()==m['decoded_pcm_sha256']
        return dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,sha256=m['feature_sha256'],
            device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,link_count=s.st_nlink,
            retained_feature_metadata_path=str(meta),retained_feature_metadata_sha256=sha(meta),
            retained_reconstruction_pcm=str(raw),retained_reconstruction_metadata_sha256=sha(native_meta),
            source_sha256=m['source_sha256'],decoded_pcm_sha256=m['decoded_pcm_sha256'])
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(inspect,candidates))
    def unescape(s):return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),s)
    nested=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        mount=unescape(line.split()[4])
        if mount.startswith(str(root)+'/'):nested.append(mount)
    assert not nested,nested
    files=[str(p) for p in candidates];manifest=out/'exact_feature_files.json'
    write_json(manifest,dict(root=str(root),records=records,cleanup_authorized=False,deletion_performed=False))
    summary=dict(created_at_unix=time.time(),audit_passed=True,root=str(root),files=len(records),
        apparent_bytes=sum(r['bytes'] for r in records),estimated_reclaimable_allocated_bytes=sum(r['allocated_bytes'] for r in records),
        exact_manifest_path=str(manifest),exact_manifest_sha256=sha(manifest),
        nested_mounts=[],symlinks_in_candidates=0,hardlinked_candidates=0,
        all_features_match_retained_metadata=True,all_required_pcm_verified_present=True,
        protected_previous_files_selected=0,source_original_audio_or_models_selected=False,
        metadata_json_or_representative_export_arrays_selected=False,
        recovery='Regenerate individual float32 96x64 arrays from retained canonical PCM and unchanged official frontend; retained metadata supplies exact expected hashes. Restoring all arrays simultaneously still needs sufficient disk space.',
        cleanup_authorized=False,deletion_performed=False,process_ownership_check_pending=True)
    write_json(out/'summary.json',summary)
    lines=['# 삭제 승인 전 캐시 검토','',
        f'대상: `{root}` 아래 정확한 manifest에 나열된 `features.npy` {len(records):,}개만입니다.',
        f'예상 회수 용량: {summary["estimated_reclaimable_allocated_bytes"]/2**30:.2f} GiB. 하드링크와 심볼릭 링크는 없습니다.','',
        '이 파일들은 이번 작업에서 계산한 96×64 float32 특징 캐시입니다. 원본 음원, 16 kHz 복원용 PCM, 특징 metadata.json, 모델·체크포인트·실험 결과·int8 대표 입력 배열은 삭제 대상이 아닙니다.',
        '모든 특징 파일의 해시와 복원용 PCM 해시가 저장된 metadata와 일치합니다. 원래 frontend 코드와 metadata를 사용해 개별 파일을 재생성할 수 있습니다. 전체를 한 번에 복원하려면 그만큼의 저장 공간이 다시 필요합니다.','',
        '실행 중 프로세스의 사용 여부를 별도 확인해야 하며, 이 보고서 생성으로 삭제를 승인하거나 실행하지 않습니다.','',
        f'정확한 파일 목록과 해시: `{manifest}`']
    (out/'REVIEW_KO.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('READ_ONLY_REPRODUCIBLE_FEATURE_CACHE_AUDIT',len(records),summary['estimated_reclaimable_allocated_bytes']/2**30,out,flush=True)


if __name__=='__main__':main()
