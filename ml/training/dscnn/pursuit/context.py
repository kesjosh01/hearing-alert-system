"""Causal 3-second context from five overlapping portable one-second windows."""
import numpy as np
from expansion.model import build_model as original_build


class Context:
    def __init__(self,base,count):
        assert count>=1 and base.ndim==4 and base.shape[-1]==1
        self.base=base;self.count=count;self.shape=(*base.shape[:-1],count);self.ndim=4
    def __len__(self):return len(self.base)
    def __getitem__(self,item):
        indices=np.arange(len(self.base))[item];scalar=np.ndim(indices)==0;indices=np.atleast_1d(indices)
        # Repeat the first available past window during stream warm-up only.
        # No future sample or future window enters an earlier prediction.
        out=np.concatenate([self.base[np.maximum(indices-delay,0)] for delay in range(self.count-1,-1,-1)],axis=-1)
        return out[0] if scalar else out


def build_model(keras,options,seed):
    if options.get('context_windows',1)==1:return original_build(keras,options,seed)
    keras.utils.set_random_seed(seed);channels=options['channels'];kernel=options['depthwise_time_kernel'];count=options['context_windows']
    inputs=keras.Input((98,40,count),name='causal_log_mel_context')
    x=keras.layers.Conv2D(channels,3,strides=2,padding='same',use_bias=False,name='stem_conv')(inputs)
    x=keras.layers.BatchNormalization(name='stem_bn')(x);x=keras.layers.ReLU(name='stem_relu')(x)
    for i in range(3):
        x=keras.layers.DepthwiseConv2D((kernel,3),padding='same',use_bias=False,name=f'dw_{i}')(x)
        x=keras.layers.BatchNormalization(name=f'dw_bn_{i}')(x);x=keras.layers.ReLU(name=f'dw_relu_{i}')(x)
        x=keras.layers.Conv2D(channels,1,use_bias=False,name=f'pw_{i}')(x)
        x=keras.layers.BatchNormalization(name=f'pw_bn_{i}')(x);x=keras.layers.ReLU(name=f'pw_relu_{i}')(x)
    x=keras.layers.AveragePooling2D((49,1),name='global_average')(x);x=keras.layers.Flatten(name='flatten')(x)
    logits=keras.layers.Dense(3,name='class_logits')(x)
    return keras.Model(inputs,keras.layers.Softmax(name='class_probability')(logits),name='causal_context_dscnn')


def update_function(tf,keras,model,optimizer,o,teacher=False):
    logits_model=model if teacher else keras.Model(model.input,model.get_layer('class_logits').output)
    shape=[None,1024] if teacher else [None,*model.input_shape[1:]]

    @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32),tf.TensorSpec([None,None],tf.float32),tf.TensorSpec([None],tf.int64),tf.TensorSpec([None,3],tf.float32)])
    def differentiate(x,weights,y,targets):
        with tf.device('/GPU:0'),tf.GradientTape() as tape:
            if not teacher:
                n=tf.shape(x)[0]
                fw=tf.random.uniform([n,1,1,1],0,o['max_frequency_mask']+1,dtype=tf.int32)
                tw=tf.random.uniform([n,1,1,1],0,o['max_time_mask']+1,dtype=tf.int32)
                fs=tf.minimum(tf.random.uniform([n,1,1,1],0,40,dtype=tf.int32),40-fw)
                ts=tf.minimum(tf.random.uniform([n,1,1,1],0,98,dtype=tf.int32),98-tw)
                mask=((tf.range(40)[None,None,:,None]>=fs)&(tf.range(40)[None,None,:,None]<fs+fw))|((tf.range(98)[None,:,None,None]>=ts)&(tf.range(98)[None,:,None,None]<ts+tw))
                x=tf.where(mask,tf.reduce_mean(x,axis=[1,2],keepdims=True),x)
            logits=logits_model(x,training=True);p=tf.nn.softmax(logits)
            bagweights=weights
            if not teacher and o['pooling_fraction']<1:
                def one(w):
                    valid=w>0;count=tf.reduce_sum(tf.cast(valid,tf.int32))
                    k=tf.maximum(1,tf.cast(tf.math.ceil(tf.cast(count,tf.float32)*o['pooling_fraction']),tf.int32))
                    score=tf.where(valid,1-p[:,2],tf.constant(-1.,tf.float32))
                    inds=tf.math.top_k(score,k=k).indices
                    return tf.scatter_nd(inds[:,None],tf.ones([k],tf.float32)/tf.cast(k,tf.float32),tf.shape(w))
                bagweights=tf.map_fn(one,weights,fn_output_signature=tf.TensorSpec([None],tf.float32))
            hard=tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y,tf.matmul(bagweights,p)))
            if teacher or o['kd_weight']==0:soft=tf.constant(0.,tf.float32);loss=hard
            else:
                t=o['temperature'];tl=tf.stop_gradient(tf.nn.log_softmax(targets/t));sl=tf.nn.log_softmax(logits/t)
                kl=tf.reduce_sum(tf.exp(tl)*(tl-sl),axis=1)
                soft=tf.reduce_mean(tf.linalg.matvec(weights,kl))*t*t
                loss=(1-o['kd_weight'])*hard+o['kd_weight']*soft
        return tf.stack([loss,hard,soft]),tape.gradient(loss,model.trainable_variables)

    @tf.function
    def apply(gradients):optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    def update(batches):
        batches=list(batches);n=sum(len(b[2]) for b in batches)
        gradients=[tf.zeros_like(v) for v in model.trainable_variables];losses=tf.zeros(3)
        for x,w,y,t in batches:
            loss,grad=differentiate(x,w,y,t);fraction=len(y)/n
            gradients=[a+b*fraction for a,b in zip(gradients,grad)];losses+=loss*fraction
        apply(gradients);return losses
    return update
