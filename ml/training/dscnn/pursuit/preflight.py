"""New study's synthetic GPU gradient and causal-policy gate, no real audio."""
import numpy as np
from dscnn_baseline.common import runtime
from expansion.model import build_model
from comparison.protocol import BASE
from comparison.model import update_function
from comparison.dataset import write_json,sha
from .metrics import scores,calibrate
from .paths import ROOT,REPORT


def main():
    out=REPORT/'gpu_preflight.json'
    if out.exists():raise FileExistsError('Preserve synthetic gate')
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42);rng=np.random.default_rng(42);checks=[]
    for channels in [24,48]:
        keras.backend.clear_session()
        with tf.device('/GPU:0'):
            net=build_model(keras,dict(channels=channels,depthwise_time_kernel=3),42);opt=keras.optimizers.Adam(.001)
        o=dict(BASE,kd_weight=0.,pooling_fraction=.5);update=update_function(tf,keras,net,opt,o,False)
        x=rng.normal(size=(4,98,40,1)).astype(np.float32);w=np.array([[.5,.5,0,0],[0,0,.5,.5]],np.float32);y=np.array([0,1],np.int64)
        v=update([(x,w,y,np.zeros((4,3),np.float32))]).numpy();assert np.isfinite(v).all() and int(opt.iterations.numpy())==1 and all('GPU:0' in t.device for t in net.variables)
        checks.append(dict(channels=channels,optimizer_steps=1,loss=float(v[0]),all_variables_gpu=True))
    p=np.array([[.9,.05,.05],[.05,.05,.9],[.9,.05,.05]],np.float32);s=scores(p,[[2,3],[1,1]]);assert s[0,0]==s[1,0]==0 and s[2,0]>.89
    rr=[dict(sample_id=str(k),label=label,split_group_id=str(k)) for k,label in enumerate(['fire_alarm','doorbell','other'])]
    ww={'0':np.array([[.99,.005,.005]],np.float32),'1':np.array([[.4,.59,.01]],np.float32),'2':np.array([[.7,.2,.1]],np.float32)}
    t,m=calibrate(rr,ww);assert m['total_cross_class_or_negative_alert_files']==0 and m['fire_recall']==1 and m['doorbell_recall']==1
    write_json(out,dict(passed=True,synthetic_only=True,real_audio_files=0,checks=checks,class_specific_vote_causal=True,
         cross_target_alerts_included_in_negative_calibration=True,not_product_performance=True,
         code_identity={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'pursuit').glob('*.py')}))
    print('PURSUIT_SYNTHETIC_GPU_GATE_PASS',flush=True)


if __name__=='__main__':main()
