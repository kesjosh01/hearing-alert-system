"""Existing portable C++ frontend, compiled locally for bounded CPU intake."""
import ctypes
import json
import shutil
import subprocess
import time
import wave
import numpy as np
from dscnn_baseline.frontend import LogMel,windows,quantize,synthetic_pcm
from comparison.dataset import write_json,sha
from .paths import ROOT,REPO,REPORT

DIRECTORY=REPORT/'native_frontend'


def build():
    target=DIRECTORY/'libpursuit_frontend.so'
    if target.exists():return target
    DIRECTORY.mkdir(exist_ok=False);cpp=REPO/'firmware/transmitter_s3/dscnn'
    command=['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic','-shared','-fPIC',
      '-I',str(cpp/'include'),str(cpp/'log_mel_frontend.cc'),str(ROOT/'pursuit/native_frontend.cc'),'-o',str(target)]
    if not shutil.which('g++'):raise RuntimeError('Existing compiler missing; no host installation permitted')
    result=subprocess.run(command,text=True,capture_output=True);(DIRECTORY/'compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('First native compilation failure; inspect compile.log')
    write_json(DIRECTORY/'build.json',dict(command=command,sources={str(p):sha(p) for p in [cpp/'log_mel_frontend.cc',cpp/'include/log_mel_frontend.h',cpp/'include/frontend_tables.h',ROOT/'pursuit/native_frontend.cc']},binary_sha256=sha(target),host_only=True,installed_packages=False))
    return target


class NativeFrontend:
    def __init__(self):
        p=build();metadata=json.loads((DIRECTORY/'build.json').read_text());assert sha(p)==metadata['binary_sha256']
        self.library=ctypes.CDLL(str(p));self.call=self.library.pursuit_logmel
        self.call.argtypes=[np.ctypeslib.ndpointer(dtype=np.int16,ndim=1,flags='C_CONTIGUOUS'),ctypes.c_size_t,
                           np.ctypeslib.ndpointer(dtype=np.float32,ndim=3,flags='C_CONTIGUOUS'),ctypes.c_size_t]
        self.call.restype=ctypes.c_int
    def __call__(self,pcm):
        p=np.asarray(pcm)
        if p.dtype!=np.int16 or p.shape!=(16000,) or not p.flags.c_contiguous:raise ValueError('Expected contiguous one-second PCM16 mono')
        out=np.empty((98,40,1),np.float32)
        if self.call(p,len(p),out,out.size)!=1 or not np.isfinite(out).all():raise ValueError('Portable frontend rejected input')
        return out


def verify():
    out=DIRECTORY/'verification.json'
    if out.exists():raise FileExistsError('Preserve native verification')
    native=NativeFrontend();python=LogMel();signals=list(synthetic_pcm(24));signals+=[np.full(16000,-32768,np.int16),np.full(16000,32767,np.int16)]
    impulse=np.zeros(16000,np.int16);impulse[1234]=32767;signals.append(impulse)
    records=[];scale=.0922469049692154;zero=22
    for i,p in enumerate(signals):
        a=native(p);b=python(p);error=float(np.abs(a-b).max());q=int(np.abs(quantize(a,scale,zero).astype(np.int16)-quantize(b,scale,zero).astype(np.int16)).max())
        assert error<=.001 and q<=1
        records.append(dict(signal=i,max_float_abs=error,max_int8_lsb=q))
    start=time.monotonic()
    for _ in range(100):native(signals[4])
    write_json(out,dict(passed=True,signals=records,synthetic_signals_only=True,board_validated=False,host_ms_per_window=(time.monotonic()-start)*10,
                       float_atol=.001,int8_lsb_tolerance=1,actual_field_performance_measured=False))
    print('NATIVE_FRONTEND_PARITY_PASS',len(records),max(r['max_float_abs'] for r in records),flush=True)


if __name__=='__main__':verify()
