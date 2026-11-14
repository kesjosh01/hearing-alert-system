"""Native40 six-block DS-CNN with a full-window temporal receptive field."""


def build(keras, seed=42):
    keras.utils.set_random_seed(seed)
    inputs=keras.Input((98,40,1),name='log_mel')
    x=keras.layers.Conv2D(32,3,strides=2,padding='same',use_bias=False,name='stem_conv')(inputs)
    x=keras.layers.BatchNormalization(name='stem_bn')(x);x=keras.layers.ReLU(name='stem_relu')(x)
    widths=[32,48,64,96,128,128]
    strides=[(1,1),(2,1),(1,1),(2,2),(1,1),(1,1)]
    for i,(width,stride) in enumerate(zip(widths,strides)):
        # This GPU depthwise implementation requires equal spatial strides.
        # A time-only average pool retains the intended frequency resolution.
        depthwise_stride=(1,1) if stride==(2,1) else stride
        x=keras.layers.DepthwiseConv2D((5,3),strides=depthwise_stride,padding='same',use_bias=False,name=f'dw_{i}')(x)
        x=keras.layers.BatchNormalization(name=f'dw_bn_{i}')(x);x=keras.layers.ReLU(name=f'dw_relu_{i}')(x)
        if stride==(2,1):x=keras.layers.AveragePooling2D((2,1),strides=(2,1),padding='same',name=f'time_downsample_{i}')(x)
        x=keras.layers.Conv2D(width,1,use_bias=False,name=f'pw_{i}')(x)
        x=keras.layers.BatchNormalization(name=f'pw_bn_{i}')(x);x=keras.layers.ReLU(name=f'pw_relu_{i}')(x)
    x=keras.layers.AveragePooling2D((int(x.shape[1]),1),name='global_average')(x)
    x=keras.layers.Flatten(name='flatten')(x)
    logits=keras.layers.Dense(3,name='class_logits')(x)
    return keras.Model(inputs,keras.layers.Softmax(name='class_probability')(logits),name='native40_long_receptive_dscnn')
