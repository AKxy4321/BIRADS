import tensorflow as tf


def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)  # convert probabilities to binary values

    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    # Weighted average of F1 scores
    weights = tf.reduce_sum(y_true, axis=0) / tf.reduce_sum(y_true)
    weighted_f1 = tf.reduce_sum(f1 * weights)

    return weighted_f1
