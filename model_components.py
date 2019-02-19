import tensorflow as tf


class DiscriminatorComponentDCGAN:
    def __init__ (self, device, scope, training):
        self.device = device
        self.scope = scope
        self.initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.regularizer = None
        self.constraint = None
        self.training = training


    def build_model(self, image_input, model_scope, reuse=tf.AUTO_REUSE):
        x = image_input

        with tf.device(self.device):
            with tf.name_scope(model_scope):
                with tf.variable_scope(self.scope, reuse=reuse):
                    _x = x # first layer initialized

                    _x = tf.layers.conv2d(inputs=_x, filters=64, kernel_size=[5,5], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='conv0')
                    _x = tf.nn.leaky_relu(_x, name='leakyrelu0')

                    _x = tf.layers.conv2d(inputs=_x, filters=128, kernel_size=[5,5], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='conv1')
                    _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm0')
                    _x = tf.nn.leaky_relu(_x, name='leakyrelu1')

                    _x = tf.layers.conv2d(inputs=_x, filters=256, kernel_size=[5,5], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='conv2')
                    _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm_1')
                    _x = tf.nn.leaky_relu(_x, name='leakyrelu2')

                    _x = tf.layers.conv2d(inputs=_x, filters=512, kernel_size=[5,5], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='conv3')
                    _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm_2')
                    _x = tf.nn.leaky_relu(_x, name='leakyrelu3')

                    _x = tf.layers.flatten(inputs=_x, name='flatten')
                    _x = tf.layers.dense(inputs=_x, units=1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, name='fullyconnected0')

                    _xf = _x
                    _x = tf.nn.sigmoid(_x, name='sigmoid0')

                    self.model_output = _x, _xf
                    return self.model_output


class GeneratorComponentDCGAN:
    def __init__ (self, device, scope, training):
        self.device = device
        self.scope = scope
        self.initializer = tf.initializers.variance_scaling()
        self.regularizer = None
        self.constraint = None
        self.training = training


    def build_model(self, noise_input, image_input, model_scope, reuse=tf.AUTO_REUSE):
        x = image_input # only used for checking the shape
        z = noise_input

        with tf.device(self.device):
            with tf.name_scope(model_scope):
                with tf.variable_scope(self.scope, reuse=reuse):
                    _z = z # first layer initialized

                    _z = tf.layers.dense(inputs=_z, units=(x.shape.as_list()[1]//16)*(x.shape.as_list()[2]//16)*512, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, name='fullyconnected0')
                    _z = tf.reshape(_z, shape=[-1, x.shape.as_list()[1]//16, x.shape.as_list()[2]//16, 512], name='reshape_0')
                    _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm0')
                    _z = tf.nn.relu(_z, name='relu0')

                    _z = tf.layers.conv2d_transpose(inputs=_z, filters=256, kernel_size=[5,5], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='deconv1')
                    _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm1')
                    _z = tf.nn.relu(_z, name='relu1')

                    _z = tf.layers.conv2d_transpose(inputs=_z, filters=128, kernel_size=[3,3], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='deconv2')
                    _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm2')
                    _z = tf.nn.relu(_z, name='relu2')

                    _z = tf.layers.conv2d_transpose(inputs=_z, filters=64, kernel_size=[3,3], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='deconv3')
                    _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm3')
                    _z = tf.nn.relu(_z, name='relu3')

                    _z = tf.layers.conv2d_transpose(inputs=_z, filters=3, kernel_size=[3,3], kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, kernel_constraint=self.constraint, strides=2, padding='SAME', name='deconv4')
                    _z = tf.nn.tanh(_z, name='tanh0')

                    self.model_output = _z
                    return self.model_output
