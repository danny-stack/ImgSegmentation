import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=3)
        
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weights = tf.pow(1 - y_pred, gamma) * y_true
        weights = weights * alpha
        fl = weights * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return loss
