import tensorflow as tf
from utils import image


class GraphGAN:
    def __init__(self, device, scope):
        self.device = device
        self.scope = scope


    def define_nodes(self, image_shape, label_shape, noise_shape):
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.noise_shape = noise_shape

        with tf.device(self.device):
            with tf.name_scope(self.scope+"_placeholders"):
                self.image_input = tf.placeholder(tf.float32, shape=[None, image_shape[0], self.image_shape[1], self.image_shape[2]], name='image_input')
                self.label_input = tf.placeholder(tf.float32, shape=[None, label_shape], name='label_input')
                self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_shape], name='noise_input')
                self.discriminator_learning_rate = tf.placeholder(tf.float32, shape=[], name='discriminator_learning_rate')
                self.generator_learning_rate = tf.placeholder(tf.float32, shape=[], name='generator_learning_rate')
                self.training = tf.placeholder(tf.bool, shape=[], name='training')


    def build_model(self, model, type):
        self.model = model
        self.model.build_model(image_input=self.image_input, noise_input=self.noise_input, label_input=self.label_input, type=type, training=self.training)
        self.generated_image = self.model.generator_model_output


    def build_graph(self, type, regularizer):
        self.type = type
        self.regularizer = regularizer

        # LOSSES
        with tf.device(self.device):
            with tf.name_scope(self.scope+"_loss"):
                # VANILLA GAN
                if type=='gan':
                    with tf.name_scope("generator_loss"):
                        self.generator_fake_loss = tf.losses.sigmoid_cross_entropy(\
                            multi_class_labels=tf.ones_like(self.model.discriminator_fake_model_output), \
                            logits=self.model.discriminator_fake_model_logit, \
                            weights=1.0, loss_collection="GENERATOR_LOSS", scope='generator_loss')

                    with tf.name_scope("discriminator_loss"):
                        self.discriminator_real_loss = tf.losses.sigmoid_cross_entropy(\
                            multi_class_labels=tf.ones_like(self.model.discriminator_real_model_output), \
                            logits=self.model.discriminator_real_model_logit, \
                            weights=1.0, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_real_loss')

                        self.discriminator_fake_loss = tf.losses.sigmoid_cross_entropy(\
                            multi_class_labels=tf.zeros_like(self.model.discriminator_fake_model_output), \
                            logits=self.model.discriminator_fake_model_logit, \
                            weights=1.0, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_fake_loss')

                # LEAST-SQUARES GAN
                elif type=='lsgan':
                    with tf.name_scope("generator_loss"):
                        self.generator_fake_loss = tf.losses.mean_squared_error(\
                            labels=tf.ones_like(self.model.discriminator_fake_model_output), \
                            predictions=self.model.discriminator_fake_model_logit, \
                            weights=0.5, loss_collection="GENERATOR_LOSS", scope='generator_loss')

                    with tf.name_scope("discriminator_loss"):
                        self.discriminator_real_loss = tf.losses.mean_squared_error(\
                            labels=tf.ones_like(self.model.discriminator_real_model_output), \
                            predictions=self.model.discriminator_real_model_logit, \
                            weights=0.5, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_real_loss')

                        self.discriminator_fake_loss = tf.losses.mean_squared_error(\
                            labels=tf.zeros_like(self.model.discriminator_fake_model_output), \
                            predictions=self.model.discriminator_fake_model_logit, \
                            weights=0.5, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_fake_loss')

                # WASSERSTEIN GAN
                elif type=='wgan' or type=='wgan-gp':
                    with tf.name_scope("generator_loss"):
                        self.generator_fake_loss = tf.losses.compute_weighted_loss(\
                            losses=self.model.discriminator_fake_model_logit, \
                            weights=-1.0, loss_collection="GENERATOR_LOSS", scope='generator_loss')

                    with tf.name_scope("discriminator_loss"):
                        self.discriminator_real_loss = tf.losses.compute_weighted_loss(\
                            losses=self.model.discriminator_real_model_logit, \
                            weights=-1.0, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_real_loss')

                        self.discriminator_fake_loss = tf.losses.compute_weighted_loss(\
                            losses=self.model.discriminator_fake_model_logit, \
                            weights=1.0, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_fake_loss')

                        # WASSERSTEIN GAN - GRADIENT PENALTY
                        if 'gp' in type:
                            with tf.name_scope('gradient_penalty'):
                                epsilon = tf.random.uniform([], name='epsilon')
                                gradient_image = tf.identity((epsilon * self.image_input + (1-epsilon) * self.generated_image), name='gradient_image')
                                discriminator_gradient_model_output, discriminator_gradient_model_logit = self.model.discriminator.build(image_input=gradient_image, model_scope='discriminator_gradient_image', reuse=True)
                                gradients = tf.gradients(discriminator_gradient_model_logit, gradient_image, name='gradients')
                                gradient_norm = tf.norm(gradients[0], ord=2, name='gradient_norm')
                                self.gradient_penalty = tf.square(gradient_norm - 1)
                                tf.add_to_collection("DISCRIMINATOR_LOSS", self.gradient_penalty)

                # GEOMETRIC GAN
                elif type=='geogan':
                    with tf.name_scope("generator_loss"):
                        self.generator_fake_loss = tf.losses.compute_weighted_loss(\
                            losses=self.model.discriminator_fake_model_logit, \
                            weights=-1.0, loss_collection="GENERATOR_LOSS", scope='generator_loss')

                    with tf.name_scope("discriminator_loss"):
                        self.discriminator_real_loss = tf.losses.hinge_loss(\
                            labels=tf.ones_like(self.model.discriminator_real_model_output), \
                            logits=self.model.discriminator_real_model_logit, \
                            weights=1.0, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_real_loss')

                        self.discriminator_fake_loss = tf.losses.hinge_loss(\
                            labels=tf.zeros_like(self.model.discriminator_fake_model_output), \
                            logits=self.model.discriminator_fake_model_logit, \
                            weights=1.0, loss_collection="DISCRIMINATOR_LOSS", scope='discriminator_fake_loss')

                else:
                    raise ValueError('unknown gan graph type: {}'.format(type))

        # REGULARIZERS
        with tf.device(self.device):
            with tf.name_scope(self.scope+'_regularizer'):
                if regularizer=='modeseek':
                    with tf.name_scope(self.regularizer):
                        _img1, _img2 = tf.split(self.generated_image, 2, axis=0, name='image_split')
                        _noise1, _noise2 = tf.split(self.generated_image, 2, axis=0, name='noise_split')
                        _modeseek_loss = tf.reduce_mean(tf.abs(_img1-_img2)) / tf.reduce_mean(tf.abs(_noise1-_noise2))
                        self.regularizer_loss = 1 / (_modeseek_loss + 1e-8)
                        tf.add_to_collection("GENERATOR_LOSS", self.regularizer_loss)

                elif regularizer=='spectralnorm':
                    raise NotImplementedError('{} is to be updated'.format(regularizer))

                else:
                    pass

        # GET FINAL LOSS
        self.generator_loss = tf.add_n(tf.get_collection("GENERATOR_LOSS"), name='generator_loss')
        self.discriminator_loss = tf.add_n(tf.get_collection("DISCRIMINATOR_LOSS"), name='discriminator_loss')

        # IMAGES
        with tf.device(self.device):
            with tf.name_scope(self.scope+'_image'):
                generated_image_summary = tf.summary.image(name='generated_image', tensor=(image.scale_in(self.generated_image)), max_outputs=64, family='generated_image', collections=["GENERATOR_SUMMARY"])
                target_image_summary = tf.summary.image(name='target_image', tensor=(image.scale_in(self.image_input)), max_outputs=64, family='target_image', collections=["GENERATOR_SUMMARY"])
                self.image_summary = tf.summary.merge([generated_image_summary, target_image_summary])

        # SUMMARIES
        with tf.device(self.device):
            with tf.name_scope(self.scope+'_summary'+'_op'):
                generator_loss_mean, generator_loss_mean_op = tf.metrics.mean(self.generator_loss, name='generator_loss', updates_collections=["GENERATOR_OPS"])
                generator_fake_loss_mean, generator_fake_loss_mean_op = tf.metrics.mean(self.generator_fake_loss, name='generator_fake_loss', updates_collections=["GENERATOR_OPS"])
                discriminator_loss_mean, discriminator_loss_mean_op = tf.metrics.mean(self.discriminator_loss, name='discriminator_loss', updates_collections=["DISCRIMINATOR_OPS"])
                discriminator_real_loss_mean, discriminator_real_loss_mean_op = tf.metrics.mean(self.discriminator_real_loss, name='discriminator_real_loss', updates_collections=["DISCRIMINATOR_OPS"])
                discriminator_fake_loss_mean, discriminator_fake_loss_mean_op = tf.metrics.mean(self.discriminator_fake_loss, name='discriminator_fake_loss', updates_collections=["DISCRIMINATOR_OPS"])
                if 'gp' in type: gradient_penalty_mean, gradient_penalty_mean_op = tf.metrics.mean(self.gradient_penalty, name='gradient_penalty', updates_collections=["DISCRIMINATOR_OPS"])
                if regularizer: regularizer_loss_mean, regularizer_loss_mean_op = tf.metrics.mean(self.regularizer_loss, name='regularizer_loss', updates_collections=["DISCRIMINATOR_OPS"])

            with tf.name_scope(self.scope+'_summary'):
                _ = tf.summary.scalar(name='generator_loss', tensor=generator_loss_mean, collections=["GENERATOR_SUMMARY"], family='01_loss_total')
                _ = tf.summary.scalar(name='discriminator_loss', tensor=discriminator_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='01_loss_total')
                _ = tf.summary.scalar(name='discriminator_real_loss', tensor=discriminator_real_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='02_loss_discriminator')
                _ = tf.summary.scalar(name='discriminator_fake_loss', tensor=discriminator_fake_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='02_loss_discriminator')
                if 'gp' in type: _ = tf.summary.scalar(name='gradient_penalty', tensor=gradient_penalty_mean, collections=["DISCRIMINATOR_SUMMARY"], family='02_loss_discriminator')
                _ = tf.summary.scalar(name='generator_learning_rate', tensor=self.generator_learning_rate, collections=["GENERATOR_SUMMARY"], family='03_hyperparameter')
                _ = tf.summary.scalar(name='discriminator_learning_rate', tensor=self.discriminator_learning_rate, collections=["DISCRIMINATOR_SUMMARY"], family='03_hyperparameter')
                if regularizer: _ = tf.summary.scalar(name='regularizer_loss', tensor=regularizer_loss_mean, collections=["DISCRIMINATOR_SUMMARY"], family='04_regularizer')

            with tf.name_scope(self.scope+'_summary'+'_merge'):
                self.generator_summary = tf.summary.merge(tf.get_collection("GENERATOR_SUMMARY"))
                self.discriminator_summary = tf.summary.merge(tf.get_collection("DISCRIMINATOR_SUMMARY"))
                self.merged_summary = tf.summary.merge_all()

        # OPTIMIZATIONS
        with tf.device(self.device):
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
                    if type=='wgan':
                        with tf.name_scope("clip_weight"):
                            for weight in discriminator_variables:
                                tf.add_to_collection("DISCRIMINATOR_OPS", weight.assign(tf.clip_by_value(weight, -0.01, 0.01, name='clip'), name='apply_clipping'))

                    discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.discriminator_learning_rate, name='discriminator_optimizer')
                    discriminator_gradients_and_variables = discriminator_optimizer.compute_gradients(loss=self.discriminator_loss, var_list=discriminator_variables)
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)+tf.get_collection("DISCRIMINATOR_OPS")):
                        self.discriminator_optimize = discriminator_optimizer.apply_gradients(discriminator_gradients_and_variables, name='discriminator_train')

                self.generator_train = [self.generator_summary, self.generator_optimize]
                self.discriminator_train = [self.discriminator_summary, self.discriminator_optimize]

        # SAVERS
        with tf.device("/CPU:0"):
            with tf.name_scope(self.scope+'_saver'):
                self.generator_saver = tf.train.Saver(var_list=tf.global_variables(scope='generator'), name='generator_saver')

        # CREATE COLLECTION FOR IMAGE GENERATION
        tf.add_to_collection("TEST_GENERATION_OPS", self.generated_image)
        tf.add_to_collection("TEST_GENERATION_OPS", self.label_input)
        tf.add_to_collection("TEST_GENERATION_OPS", self.noise_input)
        tf.add_to_collection("TEST_GENERATION_OPS", self.training)
