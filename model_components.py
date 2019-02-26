import tensorflow as tf
from utils import image


class DiscriminatorComponentGAN:
    def __init__ (self, device, scope, type, training):
        self.device = device
        self.scope = scope
        self.type = type
        self.training = training


    def build(self, image_input, label_input, model_scope, reuse=tf.AUTO_REUSE):
        x = image_input
        y = label_input

        with tf.device(self.device):
            with tf.name_scope(model_scope):
                with tf.variable_scope(self.scope, reuse=reuse):
                    _x = x # first layer initialized - this is for preventing unwanted modification of the raw input data and/or for preventing unwanted passing of raw input to intermediate layers
                    _y = y # first layer initialized
                    _xs = x.shape.as_list()[1:] # shape of x initialized [height, width, channel]

                    # VANILLA GAN
                    if self.type=='gan':
                        _x = tf.layers.conv2d(inputs=_x, filters=64, kernel_size=[4,4], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv0')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu0')
                        _x = tf.layers.conv2d(inputs=_x, filters=128, kernel_size=[4,4], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv1')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm0')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu1')
                        _x = tf.layers.flatten(inputs=_x, name='flatten')
                        _x = tf.layers.dense(inputs=_x, units=1024, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected0')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm1')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu2')
                        _x = tf.layers.dense(inputs=_x, units=1, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected1')
                        _xl = _x
                        _x = tf.nn.sigmoid(_x, name='sigmoid')

                    # DEEP CONVOLUTIONAL GAN
                    elif self.type=='dcgan':
                        _x = tf.layers.conv2d(inputs=_x, filters=64, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv0')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu0')
                        _x = tf.layers.conv2d(inputs=_x, filters=128, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv1')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm0')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu1')
                        _x = tf.layers.conv2d(inputs=_x, filters=256, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv2')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm_1')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu2')
                        _x = tf.layers.conv2d(inputs=_x, filters=512, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv3')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm_2')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu3')
                        _x = tf.layers.flatten(inputs=_x, name='flatten')
                        _x = tf.layers.dense(inputs=_x, units=1, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected0')
                        _xl = _x
                        _x = tf.nn.sigmoid(_x, name='sigmoid')

                    # CONDITIONAL GAN
                    elif self.type=='cgan':
                        _x = tf.concat([_x, tf.tile(_y[:, tf.newaxis, tf.newaxis, :], multiples=[1, _xs[0], _xs[1], 1], name=self.scope+'_convconcat_tile0')], axis=3, name=self.scope+'_convconcat_concat0')
                        _x = tf.layers.conv2d(inputs=_x, filters=64, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv0')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu0')
                        _x = tf.concat([_x, tf.tile(_y[:, tf.newaxis, tf.newaxis, :], multiples=[1, _xs[0]//2, _xs[1]//2, 1], name=self.scope+'_convconcat_tile1')], axis=3, name=self.scope+'_convconcat_concat1')
                        _x = tf.layers.conv2d(inputs=_x, filters=128, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='conv1')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm1')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu1')
                        _x = tf.reshape(_x, shape=[-1, (_xs[0]//4)*(_xs[1]//4)*128], name='reshape0')
                        _x = tf.concat([_x, _y], axis=1, name='image_label_concat0')
                        _x = tf.layers.dense(inputs=_x, units=256, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected0')
                        _x = tf.layers.batch_normalization(inputs=_x, training=self.training, name='batchnorm_2')
                        _x = tf.nn.leaky_relu(_x, name='leakyrelu2')
                        _x = tf.concat([_x, _y], axis=1, name='image_label_concat1')
                        _x = tf.layers.dense(inputs=_x, units=1, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected1')
                        _xl = _x
                        _x = tf.nn.sigmoid(_x, name='sigmoid')

                    # SELF-ATTENTION GAN
                    elif self.type=='sagan':
                        raise NotImplementedError('{} is to be updated'.format(type))

                    # AUXILIARY CLASSIFIER GAN
                    elif self.type=='acgan':
                        raise NotImplementedError('{} is to be updated'.format(type))

                    else:
                        raise ValueError('unknown gan model type: {}'.format(type))

                    self.model_output = _x, _xl
                    return self.model_output


class GeneratorComponentGAN:
    def __init__ (self, device, scope, type, training):
        self.device = device
        self.scope = scope
        self.type = type
        self.training = training


    def build(self, noise_input, image_input, label_input, model_scope, reuse=tf.AUTO_REUSE):
        x = image_input # only used for checking the shape
        y = label_input
        z = noise_input

        with tf.device(self.device):
            with tf.name_scope(model_scope):
                with tf.variable_scope(self.scope, reuse=reuse):
                    _z = z # first layer initialized
                    _y = y # first layer initialized
                    _x = x # first layer initialized
                    _xs = x.shape.as_list()[1:] # shape of x initialized [height, width, channel]

                    # VANILLA GAN
                    if self.type=='gan':
                        _z = tf.layers.dense(inputs=_z, units=1024, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected0')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm0')
                        _z = tf.nn.relu(_z, name='relu0')
                        _z = tf.layers.dense(inputs=_z, units=(_xs[0]//4)*(_xs[1]//4)*128, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected1')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm1')
                        _z = tf.nn.relu(_z, name='relu1')
                        _z = tf.reshape(_z, shape=[-1, _xs[0]//4, _xs[1]//4, 128], name='reshape_0')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=64, kernel_size=[4,4], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv0')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm2')
                        _z = tf.nn.relu(_z, name='relu2')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=_xs[2], kernel_size=[4,4], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv1')
                        _z = tf.nn.sigmoid(_z, name='sigmoid')
                        _z = image.scale_out(_z)

                    # DEEP CONVOLUTIONAL GAN
                    elif self.type=='dcgan':
                        _z = tf.layers.dense(inputs=_z, units=(_xs[0]//16)*(_xs[1]//16)*512, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected0')
                        _z = tf.reshape(_z, shape=[-1, _xs[0]//16, _xs[1]//16, 512], name='reshape_0')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm0')
                        _z = tf.nn.relu(_z, name='relu0')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=256, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv0')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm1')
                        _z = tf.nn.relu(_z, name='relu1')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=128, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv1')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm2')
                        _z = tf.nn.relu(_z, name='relu2')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=64, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv2')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm3')
                        _z = tf.nn.relu(_z, name='relu3')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=_xs[2], kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv3')
                        _z = tf.nn.tanh(_z, name='tanh0')

                    # CONDITIONAL GAN
                    elif self.type=='cgan':
                        _z = tf.concat([_z, _y], axis=1, name='noise_label_concat0')
                        _z = tf.layers.dense(inputs=_z, units=1024, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected0')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm0')
                        _z = tf.nn.relu(_z, name='relu0')
                        _z = tf.concat([_z, _y], axis=1, name='noise_label_concat1')
                        _z = tf.layers.dense(inputs=_z, units=(_xs[0]//4)*(_xs[1]//4)*128, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='fullyconnected1')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm1')
                        _z = tf.nn.relu(_z, name='relu1')
                        _z = tf.reshape(_z, shape=[-1, _xs[0]//4, _xs[1]//4, 128], name='reshape0')
                        _z = tf.concat([_z, tf.tile(_y[:, tf.newaxis, tf.newaxis, :], multiples=[1, _xs[0]//4, _xs[1]//4, 1], name=self.scope+'_convconcat_tile0')], axis=3, name=self.scope+'_convconcat_concat0')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=128, kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv0')
                        _z = tf.layers.batch_normalization(inputs=_z, training=self.training, name='batchnorm2')
                        _z = tf.nn.relu(_z, name='relu2')
                        _z = tf.concat([_z, tf.tile(_y[:, tf.newaxis, tf.newaxis, :], multiples=[1, _xs[0]//2, _xs[1]//2, 1], name=self.scope+'_convconcat_tile1')], axis=3, name=self.scope+'_convconcat_concat1')
                        _z = tf.layers.conv2d_transpose(inputs=_z, filters=_xs[2], kernel_size=[5,5], kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), strides=2, padding='SAME', name='deconv1')
                        _z = tf.nn.sigmoid(_z, name='sigmoid')
                        _z = image.scale_out(_z)

                    # SELF-ATTENTION GAN
                    elif self.type=='sagan':
                        raise NotImplementedError('{} is to be updated'.format(type))

                    # AUXILIARY CLASSIFIER GAN
                    elif self.type=='acgan':
                        raise NotImplementedError('{} is to be updated'.format(type))

                    else:
                        raise ValueError('unknown gan model type: {}'.format(type))

                    self.model_output = _z
                    return self.model_output


class ClassifierComponentGAN:
    def __init__ (self, device, scope, type, training):
        self.device = device
        self.scope = scope
        self.type = type
        self.training = training


    def build(self, image_input, label_input, model_scope, reuse=tf.AUTO_REUSE):
        x = image_input
        y = label_input

        with tf.device(self.device):
            with tf.name_scope(model_scope):
                with tf.variable_scope(self.scope, reuse=reuse):
                    _x = x # first layer initialized
                    _y = y # first layer initialized

                    # AUXILIARY CLASSIFIER GAN
                    if self.type=='acgan':
                        raise NotImplementedError('{} is to be updated'.format(type))
