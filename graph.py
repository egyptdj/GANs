from utils import image
import tensorflow as tf


class GraphGAN:
    def __init__(self, device, scope):
        self.device = device
        self.scope = scope


    def define_nodes(self, image_shape, noise_shape):
        self.image_shape = image_shape
        self.noise_shape = noise_shape

        with tf.device(self.device):
            with tf.name_scope(self.scope+"_placeholders"):
                self.image_input = tf.placeholder(tf.float32, shape=[None, image_shape[0], self.image_shape[1], self.image_shape[2]], name='image_input')
                self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_shape], name='noise_input')
                self.discriminator_learning_rate = tf.placeholder(tf.float32, shape=[], name='discriminator_learning_rate')
                self.generator_learning_rate = tf.placeholder(tf.float32, shape=[], name='generator_learning_rate')
                self.training = tf.placeholder(tf.bool, shape=[], name='training')


    def build_model(self, model, type):
        self.model = model
        self.model.build_model(image_input=self.image_input, noise_input=self.noise_input, type=type, training=self.training)


    def build_graph(self, type):
        # LOSSES
        if type=='gan':
            with tf.device(self.device):
                with tf.name_scope(self.scope+"_loss"):
                    with tf.name_scope("generator_loss"):
                        self.generator_loss = tf.losses.sigmoid_cross_entropy(\
                            multi_class_labels=tf.ones_like(self.model.discriminator_fake_model_output), \
                            logits=self.model.discriminator_fake_model_feature, \
                            weights=1.0, scope='generator_loss')

                    with tf.name_scope("discriminator_loss"):
                        self.discriminator_real_loss = tf.losses.sigmoid_cross_entropy(\
                            multi_class_labels=tf.ones_like(self.model.discriminator_real_model_output), \
                            logits=self.model.discriminator_real_model_feature, \
                            weights=1.0, scope='discriminator_real_loss')

                        self.discriminator_fake_loss = tf.losses.sigmoid_cross_entropy(\
                            multi_class_labels=tf.zeros_like(self.model.discriminator_fake_model_output), \
                            logits=self.model.discriminator_fake_model_feature, \
                            weights=1.0, scope='discriminator_fake_loss')

                        self.discriminator_loss = tf.add_n([self.discriminator_real_loss, self.discriminator_fake_loss], name='discriminator_loss')
        elif type=='lsgan':
            pass

        else:
            raise ValueError('unknown gan graph type: {}'.format(type))


        with tf.device(self.device):
            # IMAGES
            with tf.name_scope(self.scope+'_image'):
                self.generated_image = self.model.generator_model_output
                generated_image_summary = tf.summary.image(name='generated_image', tensor=(image.scale_in(self.generated_image)), max_outputs=self.image_shape[0], family='generated_image', collections=["GENERATOR_SUMMARY"])
                target_image_summary = tf.summary.image(name='target_image', tensor=(image.scale_in(self.image_input)), max_outputs=self.image_shape[0], family='target_image', collections=["GENERATOR_SUMMARY"])
                self.image_summary = tf.summary.merge([generated_image_summary, target_image_summary])

        with tf.device(self.device):
            # SUMMARIES
            with tf.name_scope(self.scope+'_summary'+'_op'):
                generator_loss_mean, generator_loss_mean_op = tf.metrics.mean(self.generator_loss, name='generator_loss', updates_collections=["GENERATOR_OPS"])
                discriminator_real_loss_mean, discriminator_real_loss_mean_op = tf.metrics.mean(self.discriminator_real_loss, name='discriminator_real_loss', updates_collections=["DISCRIMINATOR_OPS"])
                discriminator_fake_loss_mean, discriminator_fake_loss_mean_op = tf.metrics.mean(self.discriminator_fake_loss, name='discriminator_fake_loss', updates_collections=["DISCRIMINATOR_OPS"])
                discriminator_loss_mean, discriminator_loss_mean_op = tf.metrics.mean(self.discriminator_loss, name='discriminator_loss', updates_collections=["DISCRIMINATOR_OPS"])

            with tf.name_scope(self.scope+'_summary'):
                _ = tf.summary.scalar(name='generator_loss', tensor=generator_loss_mean, collections=["GENERATOR_SUMMARY"], family='01_losses')
                _ = tf.summary.scalar(name='discriminator_real_loss', tensor=discriminator_real_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='01_losses')
                _ = tf.summary.scalar(name='discriminator_fake_loss', tensor=discriminator_fake_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='01_losses')
                _ = tf.summary.scalar(name='discriminator_loss', tensor=discriminator_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='01_losses')

                _ = tf.summary.scalar(name='generator_learning_rate', tensor=self.generator_learning_rate, collections=["GENERATOR_SUMMARY"], family='02_learning_rate')
                _ = tf.summary.scalar(name='discriminator_learning_rate', tensor=self.discriminator_learning_rate, collections=["DISCRIMINATOR_SUMMARY"], family='02_learning_rate')

            with tf.name_scope(self.scope+'_summary'+'_merge'):
                self.generator_summary = tf.summary.merge(tf.get_collection("GENERATOR_SUMMARY"))
                self.discriminator_summary = tf.summary.merge(tf.get_collection("DISCRIMINATOR_SUMMARY"))
                self.merged_summary = tf.summary.merge_all()

        with tf.device(self.device):
            # OPTIMIZATIONS
            with tf.name_scope(self.scope+'_optimize'):
                with tf.name_scope("train_variables"):
                    generator_variables = tf.trainable_variables(scope='generator')
                    discriminator_variables = tf.trainable_variables(scope='discriminator')

                with tf.name_scope("generator_optimize"):
                    generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.generator_learning_rate, name='generator_optimizer')
                    generator_gradients_and_variables = generator_optimizer.compute_gradients(loss=self.generator_loss, var_list=generator_variables)
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)+tf.get_collection("GENERATOR_OPS")):
                        self.generator_optimize = generator_optimizer.apply_gradients(generator_gradients_and_variables, name='generator_train')

                with tf.name_scope("discriminator_optimize"):
                    discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.discriminator_learning_rate, name='discriminator_optimizer')
                    discriminator_gradients_and_variables = discriminator_optimizer.compute_gradients(loss=self.discriminator_loss, var_list=discriminator_variables)
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)+tf.get_collection("DISCRIMINATOR_OPS")):
                        self.discriminator_optimize = discriminator_optimizer.apply_gradients(discriminator_gradients_and_variables, name='discriminator_train')

                self.generator_train = [self.generator_summary, self.generator_optimize]
                self.discriminator_train = [self.discriminator_summary, self.discriminator_optimize]


        with tf.device("/CPU:0"):
            # SAVERS
            with tf.name_scope(self.scope+'_saver'):
                self.generator_saver = tf.train.Saver(var_list=tf.global_variables(scope='generator'), name='generator_saver')

        # CREATE COLLECTION FOR IMAGE GENERATION
        tf.add_to_collection("TEST_GENERATION_OPS", self.generated_image)
        tf.add_to_collection("TEST_GENERATION_OPS", self.noise_input)
        tf.add_to_collection("TEST_GENERATION_OPS", self.training)
