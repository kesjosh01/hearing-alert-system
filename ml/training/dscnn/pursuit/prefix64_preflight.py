"""Synthetic GPU gradient and all-integer 96x64 contract verification."""
import json
import numpy as np
from comparison.dataset import write_json,sha
from comparison.protocol import BASE
from dscnn_baseline.common import runtime
from dscnn_baseline.export import TFLM_OPS
from .paths import REPORT
from .cost import complexity
from .generic_update import build_model,update_function
from .yamnet_frontend import CONTRACT


def main():
    target=REPORT/'prefix64_preflight.json'
    if target.exists():raise FileExistsError('Preserve completed gate')
    assert json.loads((REPORT/'yamnet_frontend64_gate.json').read_text())['passed']
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42)
    rng=np.random.default_rng(42);x=rng.normal(-5,2,size=(6,96,64,1)).astype(np.float32)
    w=np.array([[.5,.5,0,0,0,0],[0,0,.5,.5,0,0],[0,0,0,0,.5,.5]],np.float32);y=np.array([0,1,2],np.int64)
    with tf.device('/GPU:0'):
        model=build_model(keras,dict(prefix_layers=8,pretrained=True),42)
        optimizer=keras.optimizers.Adam(.0003)
    update=update_function(tf,keras,model,optimizer,dict(BASE,pooling_fraction=.5,kd_weight=0.))
    loss=update([(x,w,y,np.zeros((6,3),np.float32))]).numpy()
    assert np.isfinite(loss).all() and optimizer.iterations.numpy()==1 and all('GPU:0' in v.device for v in model.variables)
    out=REPORT/'prefix64_integer_synthetic_gate';out.mkdir(exist_ok=False)

    class Serving(tf.Module):
        def __init__(self,net):super().__init__();self.net=net
        @tf.function(input_signature=[tf.TensorSpec([1,96,64,1],tf.float32,name='official_log_mel64')])
        def serve(self,x):return {'probabilities':self.net(x,training=False)}
    serving=Serving(model);saved=out/'saved_model'
    tf.saved_model.save(serving,str(saved),signatures={'serving_default':serving.serve})
    c=tf.lite.TFLiteConverter.from_saved_model(str(saved));c.optimizations=[tf.lite.Optimize.DEFAULT]
    c.representative_dataset=lambda:([a[None]] for a in x)
    c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
    path=out/'pipeline_only_official64_int8.tflite';path.write_bytes(c.convert())
    i=tf.lite.Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF);i.allocate_tensors()
    a,b=i.get_input_details()[0],i.get_output_details()[0]
    assert list(a['shape'])==[1,96,64,1] and list(b['shape'])==[1,3]
    assert a['dtype']==b['dtype']==np.int8
    ops=sorted({o['op_name'] for o in i._get_ops_details()});assert not set(ops)-TFLM_OPS
    assert all(t['dtype'] in (np.int8,np.int32) for t in i.get_tensor_details())
    scale,zero=a['quantization'];v=np.clip(np.rint(x[:1]/scale+zero),-128,127).astype(np.int8)
    i.set_tensor(a['index'],v);i.invoke();assert i.get_tensor(b['index']).shape==(1,3)
    write_json(target,dict(passed=True,loss=loss.tolist(),optimizer_steps=1,real_audio_files=0,
        input_shape=[1,96,64,1],contract=CONTRACT,operator_set=ops,model_bytes=path.stat().st_size,
        model_sha256=sha(path),complexity=complexity(model,keras),pipeline_only=True,
        old_98x40_firmware_drop_in_compatible=False,board_validated=False,product_performance_measured=False))
    print('PREFIX64_GPU_GRADIENT_AND_FULL_INT8_GATE_PASS',flush=True)


if __name__=='__main__':main()
