"""Larger integer-compatible CNN with a full-window receptive field."""
def build_model(keras,options,seed):
    keras.utils.set_random_seed(seed);inputs=keras.Input((98,40,1),name='log_mel')
    x=keras.layers.Conv2D(48,3,strides=2,padding='same',use_bias=False,name='stem_conv')(inputs)
    x=keras.layers.BatchNormalization(name='stem_bn')(x);x=keras.layers.ReLU(max_value=6,name='stem_relu')(x)
    for i,(width,stride) in enumerate([(48,1),(64,2),(64,1),(96,2),(96,1),(96,1)]):
        x=keras.layers.DepthwiseConv2D((9,5),strides=stride,padding='same',use_bias=False,name=f'dw_{i}')(x)
        x=keras.layers.BatchNormalization(name=f'dw_bn_{i}')(x);x=keras.layers.ReLU(max_value=6,name=f'dw_relu_{i}')(x)
        x=keras.layers.Conv2D(width,1,use_bias=False,name=f'pw_{i}')(x)
        x=keras.layers.BatchNormalization(name=f'pw_bn_{i}')(x);x=keras.layers.ReLU(max_value=6,name=f'pw_relu_{i}')(x)
    x=keras.layers.AveragePooling2D((int(x.shape[1]),1),name='global_average')(x);x=keras.layers.Flatten(name='flatten')(x)
    logits=keras.layers.Dense(3,name='class_logits')(x)
    return keras.Model(inputs,keras.layers.Softmax(name='class_probability')(logits),name='deep_dscnn_48_64_96')
