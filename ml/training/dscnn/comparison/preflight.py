"""Synthetic gradient/weak-pooling/integer-export gate before real training."""
import json
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.export import export_int8
from expansion.model import build_teacher_head
from .model import build,update_function
from .protocol import BASE,REPORT
from .dataset import write_json
from .metrics import pool


def main():
    p=REPORT/'gpu_preflight.json'
    if p.exists():raise FileExistsError('Preserve preflight receipt')
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();rng=np.random.default_rng(20261114);checks=[]
    keras.utils.set_random_seed(42)
    with tf.device('/GPU:0'):
        head=build_teacher_head(keras,42);opt=keras.optimizers.Adam(.001)
    up=update_function(tf,keras,head,opt,BASE,teacher=True)
    val=up([(rng.normal(size=(4,1024)).astype(np.float32),np.asarray([[.5,.5,0,0],[0,0,.5,.5]],np.float32),np.asarray([0,1],np.int64),np.zeros((4,3),np.float32))]).numpy()
    assert np.isfinite(val).all() and int(opt.iterations.numpy())==1
    checks.append(dict(teacher_head_gradient=True,optimizer_steps=1,losses=val.tolist()))
    for fraction in (1.,.5,.25):
        o=dict(BASE,pooling_fraction=fraction);keras.utils.set_random_seed(42)
        with tf.device('/GPU:0'):
            model=build(keras,o,42);optimizer=keras.optimizers.Adam(.001)
        update=update_function(tf,keras,model,optimizer,o)
        x=rng.normal(size=(4,98,40,1)).astype(np.float32)
        w=np.asarray([[.5,.5,0,0],[0,0,.5,.5]],np.float32);y=np.asarray([0,1],np.int64)
        t=np.asarray([[4,0,-3],[4,0,-3],[0,4,-3],[0,4,-3]],np.float32)
        v=update([(x,w,y,t)]).numpy()
        assert np.isfinite(v).all() and int(optimizer.iterations.numpy())==1
        assert all('GPU:0' in v.device for v in model.variables)
        pwin=model(x,training=False).numpy();bags=np.stack([pool(pwin[:2],fraction),pool(pwin[2:],fraction)])
        assert np.allclose(bags.sum(1),1.)
        checks.append(dict(pooling_fraction=fraction,losses=v.tolist(),optimizer_steps=1,variables_on_gpu=True))
        if fraction==1.:
            _,integer=export_int8(tf,model,x,REPORT/'synthetic_integer',[dict(split='synthetic',sample_id=f'synthetic-{i}') for i in range(4)],pipeline_only=True)
    write_json(REPORT/'gpu_preflight.json',dict(passed=True,synthetic_only=True,real_audio_files=0,checks=checks,integer=integer,apps_or_packages_installed=False))
    print('COMPARISON_SYNTHETIC_GPU_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
