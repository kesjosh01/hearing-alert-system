"""Host C++ parity on actual quantized outputs and signed/tie edge cases."""
import ctypes
import json
import subprocess
import numpy as np
from comparison.dataset import sha,write_json
from .paths import ROOT,REPORT
from .operating import score


def main():
    out=REPORT/'native_integer_readout';out.mkdir(exist_ok=False);src=ROOT/'pursuit/integer_margin_readout.cc';library=out/'libinteger_margin.so'
    command=['g++','-std=c++17','-O2','-shared','-fPIC',str(src),'-o',str(library)];subprocess.run(command,check=True,capture_output=True)
    lib=ctypes.CDLL(str(library));fn=lib.hearing_integer_margin_trace
    fn.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int,ctypes.POINTER(ctypes.c_int16),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint8)];fn.restype=ctypes.c_int
    d=REPORT/'int8_readout_trials/pursuit_20261114_r02_freq24_aug/logits';m=json.loads((d/'model_metadata.json').read_text());scale=m['output']['scale'];zero=m['output']['zero_point'];a=d/'selection_outputs.npz'
    original=json.loads((d/'selection_analysis.json').read_text());assert sha(a)==original['outputs_sha256']
    with np.load(a,allow_pickle=False) as h:traces=[np.ascontiguousarray(np.rint(h[k]/scale+zero),dtype=np.int8) for k in h.files]
    rng=np.random.default_rng(42);traces.append(rng.integers(-128,128,size=(2048,3),dtype=np.int16).astype(np.int8))
    traces.append(np.array([[-128,-128,-128],[127,-128,-128],[-128,127,-128],[-128,-128,127],[127,127,-128],[-128,-128,127]],np.int8))
    checked=0
    for thresholds in [[1,1],[113,95],[255,256],[256,256]]:
        t=np.array(thresholds,np.int16)
        for policy in [((1,1),(1,1)),((1,1),(2,3)),((2,3),(1,1)),((2,3),(2,3))]:
            need=np.array([x[0] for x in policy],np.int32);width=np.array([x[1] for x in policy],np.int32)
            for q in traces:
                actual=np.zeros((len(q),2),np.uint8);code=fn(q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(q),t.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),need.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),width.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),actual.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
                expected=(score(q,'margin',1.,policy)>=t)&(score(q,'margin',1.,policy)>0)
                assert code==0 and np.array_equal(actual,expected);checked+=len(q)
    write_json(out/'verification.json',dict(passed=True,host_only=True,actual_model_and_edge_trace_windows_checked=checked,integer_margin_signed_range=[0,255],threshold_256_disables_class=True,
       tie_warmup_and_two_of_three_parity=True,build_command=command,binary_sha256=sha(library),source_sha256=sha(src),header_sha256=sha(src.with_suffix('.h')),
       actual_microcontroller_or_notifications_executed=False,product_target_rates_verified=False))
    print('INTEGER_READOUT_HOST_CPP_PARITY_PASS',checked,flush=True)


if __name__=='__main__':main()
