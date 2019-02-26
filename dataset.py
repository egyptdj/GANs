import numpy as np
from os import path
from utils import image


class DatasetGAN:
    def __init__(self, batch_size, noise_shape):
        self.batch_size = batch_size
        self.noise_shape = noise_shape


    def build_dataset(self, type):
        if type=='cifar10':
            # LOAD DATA
            from keras.datasets import cifar10
            (train_image, train_label), (test_image, test_label) = cifar10.load_data()

            # PREPARE IMAGE BATCH
            _image_batch = np.concatenate([train_image, test_image], axis=0)
            self.image_batch = image.scale_out(_image_batch/255.0)

            # PREPARE LABEL BATCH
            _label_batch = np.concatenate([train_label, test_label], axis=0)
            self.label_batch = np.zeros(shape=[self.batch_size, self.noise_shape])
            for i, label in enumerate(_label_batch): self.label_batch[i, label] = 1

            # PREPARE TRAIN INDEX
            self.train_index = np.arange(start=0, stop=self.image_batch.shape[0]-self.batch_size, step=self.batch_size, dtype=int)

        elif type=='cifar100':
            # LOAD DATA
            from keras.datasets import cifar100
            (train_image, train_label), (test_image, test_label) = cifar100.load_data()

            # PREPARE IMAGE BATCH
            _image_batch = np.concatenate([train_image, test_image], axis=0)
            self.image_batch = image.scale_out(_image_batch/255.0)

            # PREPARE LABEL BATCH
            _label_batch = np.concatenate([train_label, test_label], axis=0)
            self.label_batch = np.zeros(shape=[self.batch_size, self.noise_shape])
            for i, label in enumerate(_label_batch): self.label_batch[i, label] = 1

            # PREPARE TRAIN INDEX
            self.train_index = np.arange(start=0, stop=self.image_batch.shape[0]-self.batch_size, step=self.batch_size, dtype=int)

        elif type=='mnist':
            # LOAD DATA
            from keras.datasets import mnist
            (train_image, train_label), (test_image, test_label) = mnist.load_data()

            # PREPARE IMAGE BATCH
            _train_image, _test_image = np.pad(train_image, ((0,0), (2,2), (2,2)), 'edge'), np.pad(test_image, ((0,0), (2,2), (2,2)), 'edge')
            _train_image, _test_image = _train_image[...,np.newaxis], _test_image[...,np.newaxis]
            _image_batch = np.concatenate([_train_image, _test_image], axis=0)
            self.image_batch = image.scale_out(_image_batch/255.0)

            # PREPARE LABEL BATCH
            _label_batch = np.concatenate([train_label, test_label], axis=0)
            self.label_batch = np.zeros(shape=[self.batch_size, self.noise_shape])
            for i, label in enumerate(_label_batch): self.label_batch[i, label] = 1

            # PREPARE TRAIN INDEX
            self.train_index = np.arange(start=0, stop=self.image_batch.shape[0]-self.batch_size, step=self.batch_size, dtype=int)

        else:
            raise ValueError('unknown dataset type: {}'.format(type))


    def get_shape(self):
        return self.image_batch.shape[1:], self.noise_shape, self.label_batch.max()+1


    def get_idx(self, shuffle=True):
        if shuffle:
            return np.random.permutation(self.train_index)

        else:
            return self.train_index


    def get_image_batch(self, idx):
        return self.image_batch[idx:idx+self.batch_size]


    def get_noise_batch(self):
        return np.random.uniform(-1, 1, size=(self.batch_size, self.noise_shape))


    def get_label_batch(self, idx):
        return self.label_batch[idx:idx+self.batch_size]
