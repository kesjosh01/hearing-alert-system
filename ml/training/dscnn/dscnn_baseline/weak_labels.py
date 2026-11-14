"""File-label supervision: one label per recording, never fabricated window labels."""
from __future__ import annotations

import numpy as np

from .common import config, source
from .data import development_identity


def require_file_label_approval(path):
    value = source.read_json(path)
    if value.get("kind") != "dscnn_file_label_supervision_approval" or value.get("file_label_training_selected_by_user") is not True:
        raise ValueError("Explicit file-label supervision selection is required")
    if value.get("event_intervals_reviewed") is not False:
        raise ValueError("File labels must not be represented as reviewed event intervals")
    if value["labels"] != config()[0]["labels"] or value["development_identity"] != development_identity():
        raise ValueError("Approved file-label dataset or class map changed")
    return {"mode": "file_labels", "approval_sha256": source.sha256(path),
            "window_labels_assigned": False, "event_intervals_used": False,
            "training_loss": "cross_entropy_of_all_window_mean_softmax_per_file",
            "within_group_sampling": "uniform_file_then_all_valid_windows",
            "microbatch_files": config()[0]["training"]["file_label_microbatch_files"]}


def bag_arrays(arrays):
    """Make a differentiable equal-window mean for each file, without padding."""
    sizes = [len(array) for array in arrays]
    if not sizes or min(sizes) <= 0:
        raise ValueError("Every file bag must contain at least one valid window")
    features = np.concatenate([np.asarray(array[:], dtype=np.float32) for array in arrays])
    weights = np.zeros((len(arrays), sum(sizes)), dtype=np.float32)
    offset = 0
    for i, size in enumerate(sizes):
        weights[i, offset:offset + size] = 1 / size
        offset += size
    return features, weights


class FileBagSampler(source.GroupSampler):
    """Keep original class/pool/group probabilities; draw complete recording bags."""
    def draw_bags(self, batch_size, microbatch_files):
        indices = self.rng.choice(len(self.ids), size=batch_size, p=self.probabilities)
        selected = []
        for index in indices:
            members = self.groups[self.ids[index]]
            selected.append(members[self.rng.integers(len(members))]["sample_id"])
        for begin in range(0, batch_size, microbatch_files):
            ids = selected[begin:begin + microbatch_files]
            x, weights = bag_arrays([self.features[sample] for sample in ids])
            y = np.asarray([int(self.rows[sample]["label_id"]) for sample in ids], dtype=np.int64)
            yield x, weights, y, ids


def file_label_update(tf, keras, model, optimizer, placement):
    """Accumulate file-balanced gradients with bounded file microbatches."""
    @tf.function(input_signature=[tf.TensorSpec([None, 98, 40, 1], tf.float32),
                                  tf.TensorSpec([None, None], tf.float32),
                                  tf.TensorSpec([None], tf.int64)])
    def differentiate(x, weights, y):
        with tf.device(placement), tf.GradientTape() as tape:
            probabilities = model(x, training=True)
            clip_probabilities = tf.matmul(weights, probabilities)
            loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, clip_probabilities))
        return loss, tape.gradient(loss, model.trainable_variables)

    @tf.function
    def apply(gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def update(microbatches):
        batches = list(microbatches)
        count = sum(len(y) for _, _, y, _ in batches)
        if not count:
            raise ValueError("No file bags in training batch")
        with tf.device(placement):
            accumulated = [tf.zeros_like(v) for v in model.trainable_variables]
            total_loss = tf.constant(0, tf.float32)
            for x, weights, y, _ in batches:
                loss, gradients = differentiate(x, weights, y)
                fraction = len(y) / count
                accumulated = [a + g * fraction for a, g in zip(accumulated, gradients)]
                total_loss += loss * fraction
            apply(accumulated)
        return total_loss
    return update
