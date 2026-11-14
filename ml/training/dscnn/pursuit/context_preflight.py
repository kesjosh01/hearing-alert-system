"""Validate causal context and the exact new GPU input/gradient contract."""
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.model import complexity
from comparison.dataset import write_json,sha
from comparison.protocol import BASE
from .context import Context,build_model,update_function
from .paths import ROOT,REPORT


def main():
    out=REPORT/'context_preflight.json'
    if out.exists():raise FileExistsError('Preserve completed gate')
    rng=np.random.default_rng(42);base=rng.normal(size=(8,98,40,1)).astype(np.float32);x=Context(base,5)
    before=x[:4].copy();base[4:]=rng.normal(size=base[4:].shape);assert np.array_equal(before,x[:4])
    assert np.array_equal(x[0],np.repeat(base[0],5,axis=-1))
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42)
    with tf.device('/GPU:0'):
        model=build_model(keras,dict(channels=24,depthwise_time_kernel=3,context_windows=5),42);optimizer=keras.optimizers.Adam(.001)
    checked=[]
    for weight in (0.,.25):
        o=dict(BASE,pooling_fraction=.5,kd_weight=weight);update=update_function(tf,keras,model,optimizer,o)
        w=np.array([[.5,.5,0,0],[0,0,.5,.5]],np.float32);y=np.array([0,1],np.int64)
        loss=update([(x[:4],w,y,rng.normal(size=(4,3)).astype(np.float32))]).numpy();assert np.isfinite(loss).all()
        checked.append(dict(kd_weight=weight,loss=loss.tolist()))
    assert optimizer.iterations.numpy()==2 and all('GPU:0' in v.device for v in model.variables)
    write_json(out,dict(passed=True,synthetic_only=True,no_future_window_leakage=True,context_windows=5,history_seconds=2,current_window_seconds=1,
       input_shape=model.input_shape,checks=checked,complexity=complexity(model,keras),real_board_or_microphone_execution=False,
       code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/context.py',ROOT/'pursuit/context_fit.py']}))
    print('PURSUIT_CAUSAL_CONTEXT_GPU_GATE_PASS',flush=True)


if __name__=='__main__':main()
