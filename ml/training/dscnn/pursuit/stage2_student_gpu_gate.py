"""Deterministic synthetic GPU backward gate for both student loss settings."""
import hashlib
import numpy as np
from comparison.dataset import write_json,sha
from comparison.protocol import BASE
from dscnn_baseline.common import runtime
from .paths import ROOT,REPORT
from .stage2_student_model import build
from .stage2_student_loss import make_update


def main():
    target=REPORT/'stage2_student_mil_kd_gpu_gate_v2.json';assert not target.exists()
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism()
    x=np.random.default_rng(842).normal(-10.,2.,(3,98,40,1)).astype(np.float32)
    target_logits=np.array([[3.,-1.,0.,1.],[-1.,3.,0.,1.],[-1.,0.,3.,1.]],np.float32)
    initial=None;checks=[]
    for alpha in (0.,.25):
        with tf.device('/GPU:0'):
            model=build(keras,42);before=[v.numpy().copy() for v in model.trainable_variables]
            fingerprint=hashlib.sha256(b''.join(a.tobytes() for a in model.get_weights())).hexdigest()
            if initial is None:initial=fingerprint
            else:assert fingerprint==initial,'Matched controls require identical initialization'
            optimizer=keras.optimizers.Adam(.0007)
            update=make_update(tf,keras,model,optimizer,dict(BASE,kd_weight=alpha,temperature=2.))
            losses=[]
            for _ in range(2):
                loss=update([(x,np.eye(3,dtype=np.float32),np.arange(3,dtype=np.int64),target_logits)]).numpy()
                assert np.isfinite(loss).all();losses.append(loss.tolist())
            assert all('GPU:0' in v.device for v in model.variables)
            changed=sum(not np.array_equal(a,v.numpy()) for a,v in zip(before,model.trainable_variables));assert changed>0
            assert int(optimizer.iterations.numpy())==2
            if alpha:assert all(loss[2]>0 for loss in losses)
            else:assert all(loss[2]==0 for loss in losses)
            checks.append(dict(kd_weight=alpha,synthetic_optimizer_steps=2,changed_trainable_arrays=changed,losses=losses))
        keras.backend.clear_session()
    write_json(target,dict(passed=True,synthetic_only=True,real_audio_files=0,device='GPU',deterministic_ops_enabled=True,
        matched_initialization_sha256=initial,checks=checks,feature_arrays_saved=False,source_checkpoints_modified=False,
        package_or_environment_changes=False,actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/stage2_student_gpu_gate.py',ROOT/'pursuit/stage2_student_model.py',ROOT/'pursuit/stage2_student_loss.py']}))
    print('NATIVE40_STUDENT_DETERMINISTIC_GPU_GATE_PASS',flush=True)


if __name__=='__main__':main()
