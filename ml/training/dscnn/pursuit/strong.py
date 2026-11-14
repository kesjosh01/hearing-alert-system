"""Use published temporal labels only for training; full recordings at validation."""
import json
import numpy as np


def indices(starts,intervals):
    starts=np.asarray(starts,dtype=np.float64)/16000;keep=np.zeros(len(starts),bool)
    for begin,end in intervals:
        assert np.isfinite([begin,end]).all() and 0<=begin<end
        overlap=np.maximum(0,np.minimum(starts+1,end)-np.maximum(starts,begin))
        keep |= overlap>=min(.5,(end-begin)*.5)-1e-8
    result=np.flatnonzero(keep)
    if not len(result):raise ValueError('No source window overlaps published positive events sufficiently')
    return result


class StrongView:
    def __init__(self,array,starts,intervals):
        self.array=array;self.indices=indices(starts,intervals);self.shape=(len(self.indices),*array.shape[1:])
    def __len__(self):return len(self.indices)
    def __getitem__(self,item):return self.array[self.indices[item]]


def verify():
    x=indices([0,8000,16000,24000,32000,40000],[[2.,2.4]])
    assert x.tolist()==[3,4]
    try:indices([0],[[2.,3.]])
    except ValueError:pass
    else:raise AssertionError('Out-of-range positive label was accepted')
    a=np.arange(6*98*40,dtype=np.float32).reshape(6,98,40,1);v=StrongView(a,[0,8000,16000,24000,32000,40000],[[2.,2.4]])
    assert np.array_equal(v[:],a[[3,4]]) and v.shape==(2,98,40,1)
    return dict(passed=True,overlap_indices=x.tolist(),unannotated_windows_excluded_from_positive_training=True,validation_crops_forbidden_by_loader=True)
