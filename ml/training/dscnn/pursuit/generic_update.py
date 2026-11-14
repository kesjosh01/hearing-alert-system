"""Shape-aware weak-bag CE, preserving the original objective and masking."""


def update_function(tf,keras,model,optimizer,o,teacher=False):
    assert not teacher and o['kd_weight']==0,'This version is the CE-only encoder experiment'
    logits_model=keras.Model(model.input,model.get_layer('class_logits').output)
    shape=list(model.input_shape[1:]);time_bins,freq_bins=shape[:2]

    @tf.function(input_signature=[tf.TensorSpec([None,*shape],tf.float32),
        tf.TensorSpec([None,None],tf.float32),tf.TensorSpec([None],tf.int64)])
    def differentiate(x,weights,y):
        with tf.device('/GPU:0'),tf.GradientTape() as tape:
            n=tf.shape(x)[0]
            fw=tf.random.uniform([n,1,1,1],0,o['max_frequency_mask']+1,dtype=tf.int32)
            tw=tf.random.uniform([n,1,1,1],0,o['max_time_mask']+1,dtype=tf.int32)
            fs=tf.minimum(tf.random.uniform([n,1,1,1],0,freq_bins,dtype=tf.int32),freq_bins-fw)
            ts=tf.minimum(tf.random.uniform([n,1,1,1],0,time_bins,dtype=tf.int32),time_bins-tw)
            mask=((tf.range(freq_bins)[None,None,:,None]>=fs)&(tf.range(freq_bins)[None,None,:,None]<fs+fw))|((tf.range(time_bins)[None,:,None,None]>=ts)&(tf.range(time_bins)[None,:,None,None]<ts+tw))
            x=tf.where(mask,tf.reduce_mean(x,axis=[1,2],keepdims=True),x)
            probability=tf.nn.softmax(logits_model(x,training=True));bagweights=weights
            if o['pooling_fraction']<1:
                def one(w):
                    valid=w>0;count=tf.reduce_sum(tf.cast(valid,tf.int32))
                    k=tf.maximum(1,tf.cast(tf.math.ceil(tf.cast(count,tf.float32)*o['pooling_fraction']),tf.int32))
                    scores=tf.where(valid,1-probability[:,2],tf.constant(-1.,tf.float32))
                    indices=tf.math.top_k(scores,k=k).indices
                    return tf.scatter_nd(indices[:,None],tf.ones([k],tf.float32)/tf.cast(k,tf.float32),tf.shape(w))
                bagweights=tf.map_fn(one,weights,fn_output_signature=tf.TensorSpec([None],tf.float32))
            hard=tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y,tf.matmul(bagweights,probability)))
        return tf.stack([hard,hard,tf.constant(0.,tf.float32)]),tape.gradient(hard,model.trainable_variables)

    @tf.function
    def apply(g):optimizer.apply_gradients(zip(g,model.trainable_variables))

    def update(batches):
        batches=list(batches);count=sum(len(b[2]) for b in batches)
        gradients=[tf.zeros_like(v) for v in model.trainable_variables];losses=tf.zeros(3)
        for x,w,y,_ in batches:
            loss,g=differentiate(x,w,y);fraction=len(y)/count
            gradients=[a+b*fraction for a,b in zip(gradients,g)];losses+=loss*fraction
        apply(gradients);return losses
    return update


def build_model(keras,options,seed):
    import json
    import numpy as np
    from comparison.dataset import sha
    from .yamnet_prefix import network,assign,DIRECTORY
    model=network(keras,(96,64,1),options['prefix_layers'],seed=seed)
    model._name='yamnet_prefix'+str(options['prefix_layers'])+'_official64'
    if options['pretrained']:
        j=json.loads((DIRECTORY/'verification.json').read_text());p=DIRECTORY/'frozen_variables.npz'
        assert j['passed'] and sha(p)==j['archive_sha256']
        with np.load(p,allow_pickle=False) as h:values={k:h[k] for k in h.files}
        copied=assign(model,values);assert len(copied)==4+8*(options['prefix_layers']-1)
    return model
