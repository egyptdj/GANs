import tensorflow as tf
import model_components


class ModelDCGAN:
    def __init__(self, device, scope):
        self.device = device
        self.scope = scope


    def build_model(self, noise_input, image_input, training):
        self.generator = model_components.GeneratorComponentDCGAN(device=self.device, scope="generator", training=training)
        self.discriminator = model_components.DiscriminatorComponentDCGAN(device=self.device, scope="discriminator", training=training)

        with tf.name_scope(self.scope):
            with tf.name_scope("model_inputs"):
                noise_input = tf.identity(noise_input, name="noise_input")
                image_input = tf.identity(image_input, name="image_input")

            with tf.name_scope("generator_models"):
                self.generator_model_output = self.generator.build_model(noise_input=noise_input, image_input=image_input, model_scope='generator_output', reuse=False)
                self.discriminator_fake_model_output, self.discriminator_fake_model_feature = self.discriminator.build_model(image_input=self.generator_model_output, model_scope='discriminator_fake', reuse=False)

            with tf.name_scope("discriminator_models"):
                self.discriminator_real_model_output, self.discriminator_real_model_feature = self.discriminator.build_model(image_input=image_input, model_scope='discriminator_real', reuse=True)
