"""Stable weak-label presence/absence loss for the two alert classes.

For each file and class, select its largest log odds. Presence rewards one
matching instance; absence penalizes the strongest false instance anywhere in
that file. No file label or published interval is needed at inference time.
"""
import numpy as np


def update_function(tf,keras,model,optimizer,o,teacher=False,device='/GPU:0'):
    assert not teacher and o['kd_weight']==0,'This experiment isolates supervised MIL, not distillation'
    net=keras.Model(model.input,model.get_layer('class_logits').output);shape=[None,*model.input_shape[1:]]
    @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32),tf.TensorSpec([None,None],tf.float32),tf.TensorSpec([None],tf.int64),tf.TensorSpec([None,3],tf.float32)])
    def differentiate(x,w,y,unused):
        with tf.device(device),tf.GradientTape() as tape:
            if len(shape)==4:
                n=tf.shape(x)[0];frames,bands=model.input_shape[1:3]
                fw=tf.random.uniform([n,1,1,1],0,o['max_frequency_mask']+1,dtype=tf.int32);tw=tf.random.uniform([n,1,1,1],0,o['max_time_mask']+1,dtype=tf.int32)
                fs=tf.minimum(tf.random.uniform([n,1,1,1],0,bands,dtype=tf.int32),bands-fw);ts=tf.minimum(tf.random.uniform([n,1,1,1],0,frames,dtype=tf.int32),frames-tw)
                mask=((tf.range(bands)[None,None,:,None]>=fs)&(tf.range(bands)[None,None,:,None]<fs+fw))|((tf.range(frames)[None,:,None,None]>=ts)&(tf.range(frames)[None,:,None,None]<ts+tw))
                x=tf.where(mask,tf.reduce_mean(x,axis=[1,2],keepdims=True),x)
            z=net(x,training=True)
            odds=[z[:,k]-tf.reduce_logsumexp(tf.gather(z,[j for j in range(3) if j!=k],axis=1),axis=1) for k in (0,1)]
            def one(pair):
                weight,label=pair;losses=[]
                for k in (0,1):
                    masked=tf.where(weight>0,odds[k],tf.constant(-1e30,tf.float32));index=tf.argmax(masked,output_type=tf.int32)
                    target=tf.cast(label==k,tf.float32)
                    losses.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=odds[k][index]))
                return tf.reduce_mean(tf.stack(losses))
            hard=tf.reduce_mean(tf.map_fn(one,(w,y),fn_output_signature=tf.float32))
        return tf.stack([hard,hard,tf.constant(0.)]),tape.gradient(hard,model.trainable_variables)
    @tf.function
    def apply(gradients):optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    def update(batches):
        batches=list(batches);n=sum(len(b[2]) for b in batches);gradients=[tf.zeros_like(v) for v in model.trainable_variables];total=tf.zeros(3)
        for x,w,y,targets in batches:
            loss,grad=differentiate(x,w,y,targets);fraction=len(y)/n;gradients=[a+b*fraction for a,b in zip(gradients,grad)];total+=loss*fraction
        apply(gradients);return total
    return update


def preflight():
    from dscnn_baseline.common import runtime
    from comparison.protocol import BASE
    from comparison.dataset import write_json
    from .paths import REPORT
    target=REPORT/'presence_mil_cpu_gate.json'
    if target.exists():raise FileExistsError('Preserve completed gate')
    tf,keras=runtime('cpu');keras.utils.set_random_seed(42)
    inputs=keras.Input((3,));logits=keras.layers.Dense(3,kernel_initializer=keras.initializers.Identity(),bias_initializer='zeros',name='class_logits')(inputs);model=keras.Model(inputs,keras.layers.Softmax()(logits))
    optimizer=keras.optimizers.SGD(.00001);o=dict(BASE,kd_weight=0.);update=update_function(tf,keras,model,optimizer,o,device='/CPU:0')
    probabilities=np.array([[.1,.8,.1],[.6,.3,.1],[.1,.8,.1],[.6,.3,.1]],np.float32)
    x=np.log(probabilities);w=np.array([[.5,.5,0,0],[0,0,.5,.5]],np.float32);y=np.array([0,2],np.int64)
    expected=float((-np.log(.6)-np.log(.2)-np.log(.4)-np.log(.2))/4)
    actual=update([(x,w,y,np.zeros((4,3),np.float32))]).numpy();assert abs(float(actual[0])-expected)<1e-6
    # Reset the identity model. Even a grossly wrong saturated prediction must
    # retain finite, nonzero loss/gradients instead of clipped-softmax silence.
    model.get_layer('class_logits').set_weights([np.eye(3,dtype=np.float32),np.zeros(3,np.float32)])
    before=[v.numpy().copy() for v in model.trainable_variables]
    extreme=np.array([[1000,-1000,-1000],[-1000,1000,-1000]],np.float32)
    saturated=update([(extreme,np.array([[.5,.5]],np.float32),np.array([2],np.int64),np.zeros((2,3),np.float32))]).numpy()
    assert np.isfinite(saturated).all() and saturated[0]>1000 and any(not np.array_equal(a,v.numpy()) for a,v in zip(before,model.trainable_variables))
    write_json(target,dict(passed=True,synthetic_only=True,real_audio_files=0,expected_hand_calculated_loss=expected,actual_loss=float(actual[0]),extreme_wrong_prediction_loss=float(saturated[0]),
        optimizer_steps=int(optimizer.iterations.numpy()),strongest_absent_class_penalized=True,labels_used_for_training_only=True,gpu_or_board_execution=False))
    print('STABLE_CLASS_PRESENCE_ABSENCE_MIL_CPU_GATE_PASS',flush=True)


if __name__=='__main__':preflight()
