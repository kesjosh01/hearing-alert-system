"""Check the new loss on the real CNN graph with synthetic GPU inputs."""
import numpy as np
from dscnn_baseline.common import runtime
from expansion.model import build_model
from comparison.protocol import BASE
from comparison.dataset import write_json
from .presence_mil import update_function
from .paths import REPORT


def main():
    assert __import__('json').loads((REPORT/'presence_mil_cpu_gate.json').read_text())['passed']
    out=REPORT/'presence_mil_gpu_gate.json'
    if out.exists():raise FileExistsError('Preserve completed gate')
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42)
    with tf.device('/GPU:0'):
        model=build_model(keras,dict(channels=24,depthwise_time_kernel=3),42);optimizer=keras.optimizers.Adam(.001)
    update=update_function(tf,keras,model,optimizer,dict(BASE,kd_weight=0.,pooling_fraction=0.))
    rng=np.random.default_rng(42);x=rng.normal(-5,3,size=(6,98,40,1)).astype(np.float32);w=np.array([[.5,.5,0,0,0,0],[0,0,.5,.5,0,0],[0,0,0,0,.5,.5]],np.float32);y=np.array([0,1,2],np.int64)
    loss=update([(x,w,y,np.zeros((6,3),np.float32))]).numpy();assert np.isfinite(loss).all() and int(optimizer.iterations.numpy())==1 and all('GPU:0' in v.device for v in model.variables)
    write_json(out,dict(passed=True,synthetic_only=True,real_audio_files=0,actual_optimizer_steps=1,loss=loss.tolist(),input_shape=model.input_shape,
        inference_architecture_unchanged=True,actual_board_or_microphone=False))
    print('CLASS_PRESENCE_ABSENCE_MIL_GPU_GATE_PASS',flush=True)


if __name__=='__main__':main()
