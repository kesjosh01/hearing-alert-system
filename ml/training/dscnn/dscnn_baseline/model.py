"""Small depthwise-separable CNN; only TFLM-friendly inference operators."""
from __future__ import annotations

from .common import config
from .frontend import LogMel


def build_model(keras, seed=42):
    options = config()[0]["model"]
    keras.utils.set_random_seed(seed)
    channels = options["channels"]
    inputs = keras.layers.Input(shape=LogMel().shape, name="log_mel")
    x = keras.layers.Conv2D(channels, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)
    for i in range(options["depthwise_blocks"]):
        x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"dw_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"dw_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"dw_relu_{i}")(x)
        x = keras.layers.Conv2D(channels, 1, use_bias=False, name=f"pw_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"pw_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"pw_relu_{i}")(x)
    x = keras.layers.AveragePooling2D(pool_size=(int(x.shape[1]), int(x.shape[2])), name="global_average")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(3, name="class_logits")(x)
    outputs = keras.layers.Softmax(name="class_probability")(x)
    return keras.Model(inputs, outputs, name="dscnn_1s_logmel_v1")


def complexity(model, keras):
    macs = 0
    max_elements = 0
    for layer in model.layers:
        shape = layer.output_shape
        elements = 1
        for n in shape[1:]:
            elements *= int(n)
        max_elements = max(max_elements, elements)
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            macs += elements * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, keras.layers.Conv2D):
            macs += elements * layer.kernel_size[0] * layer.kernel_size[1] * int(layer.input_shape[-1])
        elif isinstance(layer, keras.layers.Dense):
            macs += int(layer.input_shape[-1]) * int(layer.output_shape[-1])
    return {"parameters_including_bn_statistics": model.count_params(),
            "convolution_and_dense_macs_per_window": macs,
            "largest_single_int8_activation_bytes": max_elements,
            "input_int8_bytes": 98 * 40,
            "board_tensor_arena_bytes": None, "board_processing_ms": None,
            "notice": "Single tensor size is not peak RAM; excludes arena scratch/overlap, frontend, audio buffers, stack and firmware."}
