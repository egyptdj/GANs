from os import path
from warnings import warn
import numpy as np
import tensorflow as tf

class SessionDCGAN:
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

            for epoch in range(self.num_epoch):
                train_idx = self.dataset.get_idx(shuffle=True)
                train_sess.run(local_init)
                for step, idx in enumerate(train_idx):
                    train_image_input = self.dataset.get_image_batch(idx=idx)
                    train_noise_input = self.dataset.get_noise_batch()

                    # UPDATE DISCRIMINATOR
                    train_discriminator_summary, _ = train_sess.run(self.graph.discriminator_train, options=self.run_options, run_metadata=self.run_metadata, \
                        feed_dict={self.graph.image_input: train_image_input, self.graph.noise_input: train_noise_input, self.graph.training: True, self.graph.discriminator_learning_rate: self.discriminator_learning_rate})
                    if epoch==0 and step==0: self.summary_writer_train.add_run_metadata(self.run_metadata, 'discriminator')

                    # UPDATE GENERATOR
                    train_generator_summary, _ = train_sess.run(self.graph.generator_train, options=self.run_options, run_metadata=self.run_metadata, \
                        feed_dict={self.graph.image_input: train_image_input, self.graph.noise_input: train_noise_input, self.graph.training: True, self.graph.generator_learning_rate: self.generator_learning_rate})
                    if epoch==0 and step==0: self.summary_writer_train.add_run_metadata(self.run_metadata, 'generator')

                self.summary_writer_train.add_summary(train_discriminator_summary, epoch)
                self.summary_writer_train.add_summary(train_generator_summary, epoch)

                if (epoch+1)%save_epoch==0: self.graph.generator_saver.save(train_sess, path.join(path.normpath(savedir),"model","generator_model.ckpt"))

            self.graph.generator_saver.save(train_sess, path.join(path.normpath(savedir),"model","generator_model.ckpt"))


    def test_graph(self, savedir, loaddir):
        from matplotlib.pyplot import imsave
        generated_image = self._generate_image(loaddir=loaddir, num_image=64)

        # MAKE 64 IMAGES INTO 64x64 ONE IMAGE
        plot_column_list = []
        _plot_row = np.split(generated_image, 8)
        for _plot in _plot_row:
            _plot_column = np.split(np.squeeze(_plot), 8)
            plot_column = np.squeeze(np.concatenate(_plot_column, axis=1))
            plot_column_list.append(plot_column)
        plot = np.concatenate(plot_column_list, axis=1)

        imsave(path.join(savedir, 'generated_image.jpg'), plot)


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

            generated_image, noise_input, training = tf.get_collection("TEST_GENERATION_OPS")
            noise = np.random.uniform(-1, 1, size=(num_image, noise_input.shape.as_list()[1]))
            feed_dict = {noise_input: noise, training: False}

            image = generate_sess.run(generated_image, feed_dict=feed_dict)
            image = (image+1.0)/2
            return image
