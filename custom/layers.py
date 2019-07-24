import tensorflow as tf


# CONDITIONAL BATCH NORMALIZATION
# - REFERENCE: https://github.com/MingtaoGuo/sngan_projection_TensorFlow
def conditional_batch_normalization(inputs, training, y1=None, y2=None, alpha=1.0, name='conditional_batch_normalization'):
    x = inputs
    with tf.variable_scope(name):
        if y1 == None:
            beta = tf.get_variable(name=name + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=name + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            beta = tf.get_variable(name=name + 'beta', shape=[y1.shape[-1], x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=name + 'gamma', shape=[y1.shape[-1], x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta1 = tf.matmul(y1[:1, :], beta)
            gamma1 = tf.matmul(y1[:1, :], gamma)
            beta2 = tf.matmul(y2[:1, :], beta)
            gamma2 = tf.matmul(y2[:1, :], gamma)
            beta = beta1 * alpha + beta2 * (1 - alpha)
            gamma = gamma1 * alpha + gamma2 * (1 - alpha)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
