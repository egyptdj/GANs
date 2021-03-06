import model_components
import tensorflow as tf


class ModelGAN:
    def __init__(self, device, scope):
        self.device = device
        self.scope = scope


    def build_model(self, image_input, noise_input, label_input, type, training):
        self.type = type

        # INITIALIZE GENERATOR AND DISCRIMINATOR
        self.generator = model_components.GeneratorComponentGAN(device=self.device, scope="generator", type=type, training=training)
        self.discriminator = model_components.DiscriminatorComponentGAN(device=self.device, scope="discriminator", type=type, training=training)
        if type=='acgan':
            self.classifier = model_components.ClassifierComponentGAN(device=self.device, scope="classifier", type=type, training=training)

        # BUILD GENERATORS AND DISCRIMINATORS
        if type=='gan' or type=='dcgan' or type=='cgan' or type=='acgan' or type=='sngan':
            with tf.name_scope(self.scope):
                with tf.name_scope("model_inputs"):
                    noise_input = tf.identity(noise_input, name="noise_input")
                    image_input = tf.identity(image_input, name="image_input")
                    label_input = tf.identity(label_input, name="label_input")

                with tf.name_scope("generator_models"):
                    self.generator_model_output = self.generator.build(noise_input=noise_input, image_input=image_input, label_input=label_input, model_scope='generator_output', reuse=False)
                    self.discriminator_fake_model_output, self.discriminator_fake_model_logit = self.discriminator.build(image_input=self.generator_model_output, label_input=label_input, model_scope='discriminator_fake', reuse=False)

                with tf.name_scope("discriminator_models"):
                    self.discriminator_real_model_output, self.discriminator_real_model_logit = self.discriminator.build(image_input=image_input, label_input=label_input, model_scope='discriminator_real', reuse=True)

                if type=='acgan':
                    with tf.name_scope("classifier_models"):
                        self.classifier_real_model_output = self.classifier.build(image_input=image_input, label_input=label_input, model_scope='classifier_real', reuse=False)
                        self.classifier_fake_model_output = self.classifier.build(image_input=self.generator_model_output, label_input=label_input, model_scope='classifier_fake', reuse=True)
        else:
            raise ValueError('unknown gan model type: {}'.format(type))
