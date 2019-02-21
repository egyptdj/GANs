import numpy as np
from os import path
from utils import image


class DatasetGAN:
    def __init__(self, batch_size, noise_shape):
        self.batch_size = batch_size
        self.noise_shape = noise_shape


    def build_dataset(self, type):
        if type=='cifar10':
            from keras.datasets import cifar10
            (train_image, train_label), (test_image, test_label) = cifar10.load_data()
            _image_batch = np.concatenate([train_image, test_image], axis=0)
            self.label_batch = np.concatenate([train_label, test_label], axis=0)
            self.image_batch = image.scale_out(_image_batch/255.0)
            self.train_index = np.arange(start=0, stop=self.image_batch.shape[0]-self.batch_size, step=self.batch_size, dtype=int)

        elif type=='cifar100':
            from keras.datasets import cifar100
            (train_image, train_label), (test_image, test_label) = cifar100.load_data()
            _image_batch = np.concatenate([train_image, test_image], axis=0)
            self.label_batch = np.concatenate([train_label, test_label], axis=0)
            self.image_batch = image.scale_out(_image_batch/255.0)
            self.train_index = np.arange(start=0, stop=self.image_batch.shape[0]-self.batch_size, step=self.batch_size, dtype=int)

        elif type=='mnist':
            from keras.datasets import mnist
            (train_image, train_label), (test_image, test_label) = mnist.load_data()
            train_image, test_image = np.pad(train_image, ((0,0), (2,2), (2,2)), 'edge'), np.pad(test_image, ((0,0), (2,2), (2,2)), 'edge')
            train_image, test_image = train_image[...,np.newaxis], test_image[...,np.newaxis]
            _image_batch = np.concatenate([train_image, test_image], axis=0)
            self.label_batch = np.concatenate([train_label, test_label], axis=0)
            self.image_batch = image.scale_out(_image_batch/255.0)
            self.train_index = np.arange(start=0, stop=self.image_batch.shape[0]-self.batch_size, step=self.batch_size, dtype=int)

        else:
            raise ValueError('unknown dataset type: {}'.format(type))


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
