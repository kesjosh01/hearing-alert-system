"""Synthetic finite-gradient gate for the new reference head, no real audio."""
import numpy as np
from dscnn_baseline.common import runtime
from comparison.dataset import write_json
from .head import make_update
from .paths import REPORT


def main():
    target=REPORT/'head_preflight.json'
    if target.exists():raise FileExistsError('Preserve completed gate')
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism()
    results=[]
    for fraction in (1.,.01):
        keras.backend.clear_session();keras.utils.set_random_seed(42)
        with tf.device('/GPU:0'):
            inputs=keras.Input((1024,));x=keras.layers.Dense(128,activation='relu')(inputs)
            x=keras.layers.Dropout(.2)(x);model=keras.Model(inputs,keras.layers.Softmax()(keras.layers.Dense(3)(x)))
            optimizer=keras.optimizers.Adam(.001)
        update=make_update(tf,keras,model,optimizer,fraction)
        rng=np.random.default_rng(42);before=[v.numpy().copy() for v in model.trainable_variables]
        x=rng.normal(size=(6,1024)).astype(np.float32)
        w=np.array([[.5,.5,0,0,0,0],[0,0,1/3,1/3,1/3,0],[0,0,0,0,0,1]],np.float32)
        loss=update([(x,w,np.array([0,1,2],np.int64),['synthetic-a','synthetic-b','synthetic-c'])])
        assert np.isfinite(loss) and int(optimizer.iterations.numpy())==1
        assert all(np.isfinite(v.numpy()).all() for v in model.trainable_variables)
        assert any(not np.array_equal(a,v.numpy()) for a,v in zip(before,model.trainable_variables))
        devices=sorted({v.device for v in model.variables});assert all('GPU:0' in d for d in devices)
        results.append(dict(fraction=fraction,loss=loss,optimizer_steps=1,devices=devices))
    write_json(target,dict(passed=True,synthetic_only=True,real_audio_files=0,checks=results))
    print('PURSUIT_HEAD_SYNTHETIC_GPU_GATE_PASS',flush=True)


if __name__=='__main__':main()
