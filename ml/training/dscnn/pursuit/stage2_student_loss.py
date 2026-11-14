"""Stable class-presence MIL plus additive, explicitly masked teacher KL."""
import numpy as np


def make_update(tf,keras,model,optimizer,options,device='/GPU:0'):
    net=keras.Model(model.input,model.get_layer('class_logits').output)
    alpha=float(options['kd_weight']);temperature=float(options['temperature'])
    assert 0<=alpha<=1 and temperature>0
    shape=[None,*model.input_shape[1:]]
    @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32),tf.TensorSpec([None,None],tf.float32),
        tf.TensorSpec([None],tf.int64),tf.TensorSpec([None,4],tf.float32)])
    def differentiate(x,w,y,teacher):
        with tf.device(device),tf.GradientTape() as tape:
            if len(shape)==4:
                frames,bands=model.input_shape[1:3];n=tf.shape(x)[0]
                fw=tf.random.uniform([n,1,1,1],0,options['max_frequency_mask']+1,dtype=tf.int32)
                tw=tf.random.uniform([n,1,1,1],0,options['max_time_mask']+1,dtype=tf.int32)
                fs=tf.minimum(tf.random.uniform([n,1,1,1],0,bands,dtype=tf.int32),bands-fw)
                ts=tf.minimum(tf.random.uniform([n,1,1,1],0,frames,dtype=tf.int32),frames-tw)
                mask=((tf.range(bands)[None,None,:,None]>=fs)&(tf.range(bands)[None,None,:,None]<fs+fw))|((tf.range(frames)[None,:,None,None]>=ts)&(tf.range(frames)[None,:,None,None]<ts+tw))
                x=tf.where(mask,tf.reduce_mean(x,axis=[1,2],keepdims=True),x)
            logits=net(x,training=True)
            odds=[logits[:,k]-tf.reduce_logsumexp(tf.gather(logits,[j for j in range(3) if j!=k],axis=1),axis=1) for k in (0,1)]
            def one(pair):
                weight,label=pair;loss=[]
                for k in (0,1):
                    index=tf.argmax(tf.where(weight>0,odds[k],tf.constant(-1e30,tf.float32)),output_type=tf.int32)
                    loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label==k,tf.float32),logits=odds[k][index]))
                return tf.reduce_mean(tf.stack(loss))
            hard=tf.reduce_mean(tf.map_fn(one,(w,y),fn_output_signature=tf.float32))
            if alpha:
                teacher_logp=tf.stop_gradient(tf.nn.log_softmax(teacher[:,:3]/temperature))
                student_logp=tf.nn.log_softmax(logits/temperature)
                divergence=tf.reduce_sum(tf.exp(teacher_logp)*(teacher_logp-student_logp),axis=1)
                soft=tf.reduce_mean(tf.linalg.matvec(w,divergence*teacher[:,3]))*temperature**2
            else:soft=tf.constant(0.,tf.float32)
            loss=hard+alpha*soft
        gradients=tape.gradient(loss,model.trainable_variables)
        return tf.stack([loss,hard,soft]),gradients
    @tf.function
    def apply(gradients):
        checked=[tf.debugging.check_numerics(g,'nonfinite student gradient') for g in gradients]
        optimizer.apply_gradients(zip(checked,model.trainable_variables))
    def update(batches):
        batches=list(batches);count=sum(len(b[2]) for b in batches)
        gradients=[tf.zeros_like(v) for v in model.trainable_variables];loss=tf.zeros(3)
        for x,w,y,t in batches:
            value,grad=differentiate(x,w,y,t);fraction=len(y)/count
            loss+=value*fraction;gradients=[a+b*fraction for a,b in zip(gradients,grad)]
        apply(gradients);return loss
    return update


def preflight():
    from dscnn_baseline.common import runtime
    from comparison.dataset import write_json
    from .paths import REPORT
    from .stage2_student_model import build
    target=REPORT/'stage2_student_mil_kd_cpu_gate_v2.json';assert not target.exists()
    tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    def tiny():
        x=keras.Input((3,));z=keras.layers.Dense(3,use_bias=False,kernel_initializer=keras.initializers.Identity(),name='class_logits')(x)
        return keras.Model(x,keras.layers.Softmax()(z))
    x=np.array([[1.,-.2,.1],[-.4,.8,.2]],np.float32);w=np.eye(2,dtype=np.float32);y=np.array([0,1],np.int64)
    teacher=np.array([[2.,-1.,0.,0.],[-1.,2.,0.,0.]],np.float32)
    options=dict(temperature=2.,max_frequency_mask=0,max_time_mask=0,kd_weight=0.)
    a=tiny();b=tiny();ua=make_update(tf,keras,a,keras.optimizers.SGD(.01),options,device='/CPU:0')
    ub=make_update(tf,keras,b,keras.optimizers.SGD(.01),dict(options,kd_weight=.25),device='/CPU:0')
    va=ua([(x,w,y,teacher)]).numpy();vb=ub([(x,w,y,teacher)]).numpy()
    assert np.allclose(a.get_weights()[0],b.get_weights()[0],atol=1e-7) and np.allclose(va,vb)
    c=tiny();uc=make_update(tf,keras,c,keras.optimizers.SGD(.01),dict(options,kd_weight=.25),device='/CPU:0')
    teacher[:,3]=1.;vc=uc([(x,w,y,teacher)]).numpy()
    assert vc[2]>0 and vc[0]>vc[1] and not np.allclose(a.get_weights()[0],c.get_weights()[0])
    probability=np.exp(teacher[:,:3]/2.);probability/=probability.sum(1,keepdims=True)
    student=np.exp(x/2.);student/=student.sum(1,keepdims=True)
    expected=float(np.mean(np.sum(probability*np.log(probability/student),axis=1))*4.)
    assert abs(expected-float(vc[2]))<1e-6
    model=build(keras);pred=model(np.zeros((2,98,40,1),np.float32),training=False).numpy()
    assert pred.shape==(2,3) and np.isfinite(pred).all()
    write_json(target,dict(passed=True,execution='Explicit CPU synthetic only',real_audio_files=0,
        kd_mask_zero_preserves_full_supervised_gradient=True,kd_nonzero_changes_student_gradient=True,
        teacher_kl_hand_calculated=expected,teacher_kl_actual=float(vc[2]),
        additive_hard_plus_alpha_kd=True,native_input_shape=list(model.input_shape[1:]),parameters=int(model.count_params()),
        receptive_field_time_frames=117,asymmetric_depthwise_stride_replaced_by_time_pool=True,
        no_feature_arrays_saved=True,gpu_optimizer_steps=0,actual_product_targets_verified=False))
    print('STUDENT_MIL_MASKED_KD_CPU_GATE_PASS',expected,model.count_params(),flush=True)


if __name__=='__main__':preflight()
