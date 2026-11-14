"""Two controlled receptive-field candidates with the same integer operators."""
from __future__ import annotations

from improvement.model import make_update


def build_model(keras, options, seed):
    keras.utils.set_random_seed(seed)
    channels, kernel = options["channels"], options["depthwise_time_kernel"]
    inputs = keras.layers.Input((98, 40, 1), name="log_mel")
    x = keras.layers.Conv2D(channels, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)
    for i in range(3):
        x = keras.layers.DepthwiseConv2D((kernel, 3), padding="same", use_bias=False, name=f"dw_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"dw_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"dw_relu_{i}")(x)
        x = keras.layers.Conv2D(channels, 1, use_bias=False, name=f"pw_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"pw_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"pw_relu_{i}")(x)
    x = keras.layers.AveragePooling2D((49, 1), name="global_average")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    logits = keras.layers.Dense(3, name="class_logits")(x)
    outputs = keras.layers.Softmax(name="class_probability")(logits)
    return keras.Model(inputs, outputs, name="expanded_temporal_dscnn")


def build_teacher_head(keras, seed):
    keras.utils.set_random_seed(seed)
    inputs = keras.layers.Input((1024,), name="yamnet_embedding")
    x = keras.layers.Dense(128, activation="relu", name="embedding_classifier")(inputs)
    x = keras.layers.Dropout(.2, name="head_dropout")(x)
    return keras.Model(inputs, keras.layers.Dense(3, name="class_logits")(x), name="expanded_yamnet_head")


def teacher_update(tf, keras, model, optimizer):
    @tf.function(input_signature=[tf.TensorSpec([None, 1024], tf.float32),
                                  tf.TensorSpec([None, None], tf.float32), tf.TensorSpec([None], tf.int64)])
    def differentiate(x, weights, y):
        with tf.device("/GPU:0"), tf.GradientTape() as tape:
            probability = tf.nn.softmax(model(x, training=True))
            loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, tf.matmul(weights, probability)))
        return loss, tape.gradient(loss, model.trainable_variables)

    @tf.function
    def apply(gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def update(batches):
        batches = list(batches)
        count = sum(len(b[2]) for b in batches)
        with tf.device("/GPU:0"):
            gradients, loss = [tf.zeros_like(v) for v in model.trainable_variables], tf.zeros(())
            for x, weights, y, _ in batches:
                value, current = differentiate(x, weights, y)
                fraction = len(y) / count
                loss += value * fraction
                gradients = [a + b * fraction for a, b in zip(gradients, current)]
            apply(gradients)
        return tf.stack([loss, loss, tf.zeros(())])
    return update
