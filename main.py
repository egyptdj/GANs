from network import *
from utils import option


class GAN:
    def __init__(self):
        self.base_option = option.parse()
        self.config, self.device = option.gpu(device_id=self.base_option['gpu_device'])

    def initialize(self):
        self.network = NetworkGAN(\
            device=self.device, \
            config=self.config)

    def build(self):
        self.network.build_network(\
            batch_size=self.base_option['batch_size'], \
            noise_shape=self.base_option['noise_shape'], \
            num_epoch=self.base_option['num_epoch'], \
            discriminator_learning_rate=self.base_option['discriminator_learning_rate'], \
            generator_learning_rate=self.base_option['generator_learning_rate'], \
            scope=self.base_option['scope'])

    def train(self):
        self.network.train(\
            savedir=self.base_option['savedir'], \
            save_epoch=self.base_option['save_epoch'])

    def test(self):
        self.network.test(\
            savedir=self.base_option['savedir'], \
            loaddir=self.base_option['savedir'])


if __name__ == "__main__":
    gan = GAN()
    gan.initialize()
    gan.build()
    gan.train()
    gan.test()
