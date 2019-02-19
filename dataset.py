import numpy as np
from os import path


class Cifar10:
    def __init__(self, batch_size, noise_shape):
        from keras.datasets import cifar10

        self.batch_size = batch_size
        self.noise_shape = noise_shape
        (train_image, train_label), (test_image, test_label) = cifar10.load_data()
        _image_batch = np.concatenate([train_image, test_image], axis=0)
        self.label_batch = np.concatenate([train_label, test_label], axis=0)
        self.image_batch = (_image_batch/127.5)-1.0
        self.train_index = np.arange(start=0, stop=self.image_batch.shape[0], step=self.batch_size, dtype=int)

    def get_shape(self):
        return self.image_batch.shape[1:], self.noise_shape

    def get_idx(self, shuffle=True):
        if shuffle:
            return np.random.permutation(self.train_index)
        else:
            return self.train_index

    def get_image_batch(self, idx):
        return self.image_batch[idx:idx+self.batch_size]

    def get_label_batch(self, idx):
        return self.label_batch[idx:idx+self.batch_size]

    def get_noise_batch(self):
        return np.random.uniform(-1, 1, size=(self.batch_size, self.noise_shape))
