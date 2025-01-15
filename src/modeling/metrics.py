import tensorflow as tf

def subset_accuracy(y_true, y_pred):
    """Calculate subset accuracy (all labels must match)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(tf.round(y_true), tf.round(y_pred)), axis=1), tf.float32))

def hamming_loss(y_true, y_pred):
    """Calculate Hamming loss."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.cast(tf.not_equal(y_true, tf.round(y_pred)), tf.float32))

def matthews_correlation(y_true, y_pred):
    """Calculate Matthews Correlation Coefficient."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + tf.keras.backend.epsilon())

def macro_f1_score(y_true, y_pred):
    """Calculate macro F1 score."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    return tf.reduce_mean(f1)
