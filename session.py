import numpy as np
import tensorflow as tf
from os import path
from utils import image
from warnings import warn


class SessionGAN:
    def __init__(self, config):
        self.config = config
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()


    def build_session(self, dataset, graph, num_epoch, discriminator_learning_rate, generator_learning_rate):
        self.dataset = dataset
        self.graph = graph
        self.num_epoch = num_epoch
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate


    def train_graph(self, savedir, save_epoch):
        with tf.Session(config=self.config) as train_sess:
            global_init = tf.initializers.global_variables()
            local_init = tf.initializers.local_variables()
            train_sess.run(global_init)
            self.summary_writer_train = tf.summary.FileWriter(path.join(path.normpath(savedir),"summary","train"), graph=train_sess.graph)
            train_sess.graph.finalize()

            # RUN EPOCHS
            for epoch in range(self.num_epoch):
                train_idx = self.dataset.get_idx(shuffle=True)
                train_sess.run(local_init)

                # RUN STEPS IN EPOCH
                for step, idx in enumerate(train_idx):
                    train_image_input = self.dataset.get_image_batch(idx=idx)
                    train_label_input = self.dataset.get_label_batch(idx=idx)
                    train_noise_input = self.dataset.get_noise_batch()

                    # UPDATE DISCRIMINATOR
                    train_discriminator_summary, _ = train_sess.run(self.graph.discriminator_train, options=self.run_options, run_metadata=self.run_metadata, \
                        feed_dict={self.graph.image_input: train_image_input, self.graph.noise_input: train_noise_input, self.graph.label_input: train_label_input, self.graph.training: True, self.graph.discriminator_learning_rate: self.discriminator_learning_rate})
                    if epoch==0 and step==0: self.summary_writer_train.add_run_metadata(self.run_metadata, 'discriminator')

                    # UPDATE GENERATOR
                    if ('wgan' in self.graph.type) and ((step+1)%5==0): continue
                    train_generator_summary, _ = train_sess.run(self.graph.generator_train, options=self.run_options, run_metadata=self.run_metadata, \
                        feed_dict={self.graph.image_input: train_image_input, self.graph.noise_input: train_noise_input, self.graph.label_input: train_label_input, self.graph.training: True, self.graph.generator_learning_rate: self.generator_learning_rate})
                    if epoch==0 and step==0: self.summary_writer_train.add_run_metadata(self.run_metadata, 'generator')

                # ADD SUMMARY
                self.summary_writer_train.add_summary(train_discriminator_summary, epoch)
                self.summary_writer_train.add_summary(train_generator_summary, epoch)

                # SAVE MODEL
                if (epoch+1)%save_epoch==0: self.graph.generator_saver.save(train_sess, path.join(path.normpath(savedir),"model","generator_model.ckpt"))
            self.graph.generator_saver.save(train_sess, path.join(path.normpath(savedir),"model","generator_model.ckpt"))


    def test_graph(self, savedir, loaddir):
        generated_image = self._generate_image(loaddir=loaddir, num_image=100)
        image.save(savedir, generated_image, row=10, column=10)


    def _generate_image(self, loaddir, num_image):
        tf.reset_default_graph()
        with tf.Session(config=self.config) as generate_sess:
            # RESTORE MODEL
            try:
                latest_checkpoint = tf.train.latest_checkpoint(path.join(loaddir, "model"))
                meta_graph = tf.train.import_meta_graph(".".join([latest_checkpoint, "meta"]))
                meta_graph.restore(sess=generate_sess, save_path=latest_checkpoint)
            except:
                warn("meta graph not found")
                return None

            # SET VARIABLES
            generate_image_op, label_input, noise_input, training = tf.get_collection("TEST_GENERATION_OPS")
            label_shape = label_input.shape.as_list()[1]
            noise_shape = noise_input.shape.as_list()[1]

            # DEFINE LABEL AND NOISE
            label = np.zeros(shape=(num_image, label_shape))
            for i in range(num_image): label[i, i%label_shape] = 1 # condition one sample per one label
            noise = np.random.uniform(-1, 1, size=(num_image, noise_shape))
            feed_dict = {label_input: label, noise_input: noise, training: False}

            # GENERATE IMAGE
            generated_image = generate_sess.run(generate_image_op, feed_dict=feed_dict)
            generated_image = image.scale_in(generated_image)
            return generated_image
