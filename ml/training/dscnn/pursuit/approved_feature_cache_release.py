"""One approved, immutable file list; preserve directories and reconstruction data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import wave

ROOT = Path('/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn')
CACHE = ROOT / '.cache/pursuit_yamnet_logmel64_v1'
NATIVE = ROOT / '.cache/pursuit_features_v1'
REPORT = ROOT / 'reports/pursuit_20261114/cache_release_review'
MANIFEST_SHA = 'e10ab1f787c43d17bb6768835bc19480a9149886e5f6b1613c7e83e135181f57'
SUMMARY_SHA = 'ab7a8c4d401ea379593f4516d8284f1d968edc421c682bf583bb2f9de09329ca'
RECONSTRUCTION_SHA = '0ca4c4f42462062f952a26bc6701e042b63a85a202a0e9a0b330d1e6f7808997'
RECEIPT = REPORT / 'approved_exact_release_receipt.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def check_ancestors(path):
    for p in [path, *path.parents]:
        assert not p.is_symlink(), str(p)
    assert path == path.resolve()


def mounts():
    found = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        p = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), line.split()[4])
        if p == str(CACHE) or p.startswith(str(CACHE) + '/'):
            found.append(p)
    assert not found, found
    return found


def process_owners(records):
    names = {r['path'] for r in records}
    inodes = {(r['device'], r['inode']) for r in records}
    matches, denied_own, active = [], {}, []
    own_seen = denied_other = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            same_uid = p.stat().st_uid == os.getuid()
            comm = (p / 'comm').read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if same_uid:
            own_seen += 1
            try:
                cmd = (p / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
                if 'pursuit.stage2_student_fit' in cmd and 'python' in cmd:
                    active.append({'pid': int(p.name), 'name': comm, 'kind': 'native40_in_memory_student_fit'})
            except (FileNotFoundError, ProcessLookupError):
                continue
        denied = False
        try:
            fds = list((p / 'fd').iterdir())
            for fd in fds:
                try:
                    target = os.readlink(fd)
                    st = fd.stat()
                    if target.removesuffix(' (deleted)') in names or (st.st_dev, st.st_ino) in inodes:
                        matches.append({'pid': int(p.name), 'name': comm, 'kind': 'fd', 'target': target})
                except (FileNotFoundError, ProcessLookupError):
                    continue
            maps = (p / 'maps').read_text()
            for line in maps.splitlines():
                fields = line.split(None, 5)
                if len(fields) < 5:
                    continue
                major, minor = (int(v, 16) for v in fields[3].split(':'))
                if (os.makedev(major, minor), int(fields[4])) in inodes:
                    matches.append({'pid': int(p.name), 'name': comm, 'kind': 'mmap', 'mapping': line})
        except PermissionError:
            denied = True
        except (FileNotFoundError, ProcessLookupError):
            continue
        if denied:
            if same_uid:
                denied_own[p.name] = comm
            else:
                denied_other += 1
    report = dict(checked_at_unix=time.time(), matching_open_files_or_mmaps=matches,
                  current_uid_processes_seen=own_seen, denied_current_uid_process_names=denied_own,
                  denied_other_uid_process_count=denied_other, active_student_processes=active,
                  coverage_limited=bool(denied_own or denied_other), processes_stopped=False)
    save(REPORT / 'approved_release_live_process_check.json', report)
    assert not matches, matches
    # These two pre-existing service processes are not training/download owners.
    assert set(denied_own.values()) <= {'(sd-pam)', 'fusermount3'}, denied_own
    return report


def inspect_record(r, cache_device):
    p = Path(r['path'])
    assert p.name == 'features.npy' and p.parent.parent == CACHE
    assert re.fullmatch('[0-9a-f]{64}', p.parent.name)
    assert p.parent.name == r['source_sha256'] and not p.parent.is_symlink()
    st = p.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1
    assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == (
        r['device'], r['inode'], r['bytes'], r['mtime_ns'])
    assert st.st_dev == cache_device and st.st_blocks * 512 == r['allocated_bytes']
    assert digest(p) == r['sha256']
    meta = p.parent / 'metadata.json'
    assert str(meta) == r['retained_feature_metadata_path'] and not meta.is_symlink()
    assert digest(meta) == r['retained_feature_metadata_sha256']
    raw = NATIVE / r['source_sha256'] / 'audio.wav'
    native_meta = raw.parent / 'metadata.json'
    assert str(raw) == r['retained_reconstruction_pcm']
    check_ancestors(raw)
    assert not native_meta.is_symlink()
    assert digest(native_meta) == r['retained_reconstruction_metadata_sha256']
    with wave.open(str(raw)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert hashlib.sha256(w.readframes(w.getnframes())).hexdigest() == r['decoded_pcm_sha256']
    parent_stat = p.parent.stat()
    return (parent_stat.st_dev, parent_stat.st_ino)


def main():
    assert not RECEIPT.exists(), 'An execution receipt already exists; inspect it instead of replaying.'
    manifest = REPORT / 'exact_feature_files.json'
    assert digest(manifest) == MANIFEST_SHA
    assert digest(REPORT / 'summary.json') == SUMMARY_SHA
    assert digest(REPORT / 'sample_reconstruction_gate.json') == RECONSTRUCTION_SHA
    assert json.loads((REPORT / 'sample_reconstruction_gate.json').read_text())['passed']
    m = json.loads(manifest.read_text())
    records = m['records']
    assert m['root'] == str(CACHE) and len(records) == 11581
    assert len({r['path'] for r in records}) == len(records)
    check_ancestors(CACHE)
    mounts()
    protected = json.loads((ROOT / 'reports/pursuit_20261114/protected_previous_comparison.json').read_text())
    assert not set(protected).intersection(r['path'] for r in records)
    for p, expected in protected.items():
        assert digest(Path(p)) == expected, p
    target_paths = {r['path'] for r in records}
    initial_out_of_scope = {
        str(p.relative_to(CACHE)): (p.lstat().st_dev, p.lstat().st_ino, p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in CACHE.rglob('*') if p.is_file() and str(p) not in target_paths
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        parent_ids = list(pool.map(lambda r: inspect_record(r, CACHE.stat().st_dev), records))
    print('APPROVED_SCOPE_REVALIDATED', len(records), 'feature_and_reconstruction_hashes', flush=True)
    owners = process_owners(records)
    mounts()
    v = os.statvfs(CACHE)
    receipt = dict(started_at_unix=time.time(), approved_user_answer='허용: 검토 목록의 특징 캐시만 삭제',
                   exact_manifest_sha256=MANIFEST_SHA, literal_root=str(CACHE),
                   requested_files=len(records), requested_apparent_bytes=sum(r['bytes'] for r in records),
                   requested_allocated_bytes=sum(r['allocated_bytes'] for r in records),
                   free_bytes_before=v.f_bavail * v.f_frsize, deleted_files=0, completed=False,
                   reconstruction_sample_gate_sha256=RECONSTRUCTION_SHA,
                   all_reconstruction_pcm_hashes_rechecked=True, processes_stopped=False,
                   process_coverage_limited=owners['coverage_limited'], collection_resume_authorized=False,
                   original_review_preserved=True, out_of_scope_files_observed=len(initial_out_of_scope))
    save(RECEIPT, receipt)
    root_fd = os.open(CACHE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for i, (r, parent_id) in enumerate(zip(records, parent_ids), 1):
            parent_fd = os.open(r['source_sha256'], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                st = os.fstat(parent_fd)
                assert (st.st_dev, st.st_ino) == parent_id
                st = os.stat('features.npy', dir_fd=parent_fd, follow_symlinks=False)
                assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1
                assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == (
                    r['device'], r['inode'], r['bytes'], r['mtime_ns'])
                os.unlink('features.npy', dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            receipt['deleted_files'] = i
            if i % 500 == 0:
                save(RECEIPT, receipt)
    finally:
        os.close(root_fd)
        save(RECEIPT, receipt)
    assert all(not Path(r['path']).exists() for r in records)
    for r in records:
        assert digest(Path(r['retained_feature_metadata_path'])) == r['retained_feature_metadata_sha256']
        raw = Path(r['retained_reconstruction_pcm'])
        assert raw.is_file()
        assert digest(raw.parent / 'metadata.json') == r['retained_reconstruction_metadata_sha256']
    for rel, expected in initial_out_of_scope.items():
        st = (CACHE / rel).lstat()
        assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == expected, rel
    for p, expected in protected.items():
        assert digest(Path(p)) == expected, p
    v = os.statvfs(CACHE)
    receipt.update(completed=True, finished_at_unix=time.time(),
                   free_bytes_after=v.f_bavail * v.f_frsize,
                   approved_files_all_absent=True, retained_feature_metadata_all_hashes_unchanged=True,
                   reconstruction_pcm_all_present=True, retained_native_metadata_all_hashes_unchanged=True,
                   protected_previous_files_checked=len(protected), protected_previous_files_unchanged=True,
                   all_observed_out_of_scope_files_stat_unchanged=True,
                   directories_deleted=0, files_deleted_outside_exact_manifest=0)
    receipt['observed_filesystem_free_byte_increase'] = receipt['free_bytes_after'] - receipt['free_bytes_before']
    save(RECEIPT, receipt)
    print('EXACT_APPROVED_FEATURE_RELEASE_COMPLETE', json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
