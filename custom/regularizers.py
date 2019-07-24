import tensorflow as tf


# SPECTRAL NORMALIZATION
# - REFERENCE: https://github.com/taki0112/Spectral_Normalization-Tensorflow
def spectral_normalization(kernel, scope, collection=tf.GraphKeys.UPDATE_OPS, iteration=1):
    with tf.name_scope("regularizer"):
        with tf.variable_scope(scope, reuse=False):
            w = kernel
            w_shape = w.shape.as_list()
            w = tf.reshape(w, [-1, w_shape[-1]])

            u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

            u_hat = u
            v_hat = None
            for i in range(iteration):
                """
                power iteration
                Usually iteration = 1 will be enough
                """
                v_ = tf.matmul(u_hat, tf.transpose(w))
                v_hat = tf.nn.l2_normalize(v_)

                u_ = tf.matmul(v_hat, w)
                u_hat = tf.nn.l2_normalize(u_)

            u_hat = tf.stop_gradient(u_hat)
            v_hat = tf.stop_gradient(v_hat)

            sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = w / sigma
                w_norm = tf.reshape(w_norm, w_shape)
                apply_sn = kernel.assign(w_norm)
                tf.add_to_collection(collection, apply_sn)
