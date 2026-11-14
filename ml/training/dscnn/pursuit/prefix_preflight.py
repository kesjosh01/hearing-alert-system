"""GPU gradient and integer-operator gate for the new transferred encoder."""
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.export import export_int8
from comparison.protocol import BASE
from comparison.model import update_function
from comparison.dataset import write_json
from .yamnet_prefix import build_model
from .cost import complexity
from .paths import REPORT


def main():
    out=REPORT/'prefix_preflight.json'
    if out.exists():raise FileExistsError('Preserve completed gate')
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42);rng=np.random.default_rng(42);checks=[]
    x=rng.normal(-5,3,size=(6,98,40,1)).astype(np.float32);w=np.array([[.5,.5,0,0,0,0],[0,0,.5,.5,0,0],[0,0,0,0,.5,.5]],np.float32);y=np.array([0,1,2],np.int64)
    for pretrained in [False,True]:
        keras.backend.clear_session()
        with tf.device('/GPU:0'):
            model=build_model(keras,dict(prefix_layers=8,pretrained=pretrained),42);optimizer=keras.optimizers.Adam(.0003)
        update=update_function(tf,keras,model,optimizer,dict(BASE,pooling_fraction=.5,kd_weight=0.))
        loss=update([(x,w,y,np.zeros((6,3),np.float32))]).numpy();assert np.isfinite(loss).all() and optimizer.iterations.numpy()==1 and all('GPU:0' in v.device for v in model.variables)
        checks.append(dict(pretrained=pretrained,loss=loss.tolist(),complexity=complexity(model,keras)))
    _,integer=export_int8(tf,model,x,REPORT/'prefix_integer_synthetic_gate',[dict(split='synthetic',sample_id='synthetic-'+str(i)) for i in range(len(x))],pipeline_only=True)
    write_json(out,dict(passed=True,synthetic_only=True,real_audio_files=0,checks=checks,integer_contract=integer,board_performance_measured=False,source_yamnet_changed=False))
    print('PRETRAINED_PREFIX_GPU_AND_INTEGER_GATE_PASS',flush=True)


if __name__=='__main__':main()
