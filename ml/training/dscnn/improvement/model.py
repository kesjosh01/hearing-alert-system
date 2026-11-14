from __future__ import annotations


def build_model(keras, options, seed):
    keras.utils.set_random_seed(seed)
    channels = options["channels"]
    inputs = keras.layers.Input((98, 40, 1), name="log_mel")
    x = keras.layers.Conv2D(channels, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)
    for i in range(3):
        x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"dw_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"dw_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"dw_relu_{i}")(x)
        x = keras.layers.Conv2D(channels, 1, use_bias=False, name=f"pw_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"pw_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"pw_relu_{i}")(x)
    pool = (49, 1 if options["keep_frequency"] else 20)
    x = keras.layers.AveragePooling2D(pool_size=pool, name="global_average")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    logits = keras.layers.Dense(3, name="class_logits")(x)
    outputs = keras.layers.Softmax(name="class_probability")(logits)
    return keras.Model(inputs, outputs, name="distilled_dscnn_1s")


def make_update(tf, keras, model, optimizer, options, augmentation):
    logits_model = keras.Model(model.input, model.get_layer("class_logits").output)
    temperature, alpha = options["temperature"], options["hard_label_weight"]

    @tf.function(input_signature=[tf.TensorSpec([None, 98, 40, 1], tf.float32),
                                  tf.TensorSpec([None, None], tf.float32),
                                  tf.TensorSpec([None], tf.int64),
                                  tf.TensorSpec([None, 3], tf.float32)])
    def differentiate(x, weights, y, teacher_logits):
        with tf.device("/GPU:0"), tf.GradientTape() as tape:
            if augmentation:
                n = tf.shape(x)[0]
                fw = tf.random.uniform([n, 1, 1, 1], 0, options["max_frequency_mask"] + 1, dtype=tf.int32)
                tw = tf.random.uniform([n, 1, 1, 1], 0, options["max_time_mask"] + 1, dtype=tf.int32)
                fs = tf.minimum(tf.random.uniform([n, 1, 1, 1], 0, 40, dtype=tf.int32), 40 - fw)
                ts = tf.minimum(tf.random.uniform([n, 1, 1, 1], 0, 98, dtype=tf.int32), 98 - tw)
                fi, ti = tf.range(40)[None, None, :, None], tf.range(98)[None, :, None, None]
                masked = ((fi >= fs) & (fi < fs + fw)) | ((ti >= ts) & (ti < ts + tw))
                x = tf.where(masked, tf.reduce_mean(x, axis=[1, 2], keepdims=True), x)
            logits = logits_model(x, training=True)
            probability = tf.nn.softmax(logits)
            hard = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, tf.matmul(weights, probability)))
            teacher_log = tf.stop_gradient(tf.nn.log_softmax(teacher_logits / temperature))
            student_log = tf.nn.log_softmax(logits / temperature)
            per_window_kl = tf.reduce_sum(tf.exp(teacher_log) * (teacher_log - student_log), axis=1)
            soft = tf.reduce_mean(tf.linalg.matvec(weights, per_window_kl)) * temperature ** 2
            loss = alpha * hard + (1 - alpha) * soft
        return tf.stack([loss, hard, soft]), tape.gradient(loss, model.trainable_variables)

    @tf.function
    def apply(gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def update(batches):
        batches = list(batches)
        files = sum(len(b[2]) for b in batches)
        with tf.device("/GPU:0"):
            gradients = [tf.zeros_like(v) for v in model.trainable_variables]
            losses = tf.zeros(3)
            for x, weights, y, targets in batches:
                value, current = differentiate(x, weights, y, targets)
                fraction = len(y) / files
                gradients = [a + b * fraction for a, b in zip(gradients, current)]
                losses += value * fraction
            apply(gradients)
        return losses
    return update
