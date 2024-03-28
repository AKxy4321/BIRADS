import tensorflow as tf


def f1_score(y_true, y_pred):
    y_pred = tf.cast(tf.round(y_pred), dtype=tf.float32)  # Ensure y_pred is float32

    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, dtype=tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), dtype=tf.float32), axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    # Weighted average of F1 scores
    weights = tf.reduce_sum(y_true, axis=0) / tf.reduce_sum(y_true)
    weights = tf.cast(weights, dtype=tf.float32)
    weighted_f1 = tf.reduce_sum(f1 * weights)

    return weighted_f1
