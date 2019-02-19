from network import *
from utils import option


class GAN:
    def __init__(self):
        self.base_option = option.parse()
        self.config, self.device = option.gpu(device_id=self.base_option['gpu_device'])

    def initialize(self):
        self.network = NetworkDCGAN(\
            batch_size=self.base_option['batch_size'], \
            num_epoch=self.base_option['num_epoch'], \
            discriminator_learning_rate=self.base_option['discriminator_learning_rate'], \
            generator_learning_rate=self.base_option['generator_learning_rate'], \
            scope=self.base_option['scope'])

    def build(self):
        self.network.build_network(\
            device=self.device, \
            config=self.config)

    def train(self):
        self.network.train(\
            savedir=self.base_option['savedir'], \
            save_epoch=self.base_option['save_epoch'])

    def test(self):
        self.network.test(\
            config=self.config, \
            savedir=self.base_option['savedir'], \
            loaddir=self.base_option['savedir'])


if __name__ == "__main__":
    gan = GAN()
    gan.initialize()
    gan.build()
    gan.train()
    gan.test()
