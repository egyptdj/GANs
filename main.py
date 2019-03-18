from network import *
from utils import option


class GAN:
    def __init__(self):
        self.base_option = option.parse()
        self.config, self.device = option.gpu(device_id=self.base_option['gpu_device'])


    def run(self):
        self._initialize()

        if self.base_option['loaddir']:
            self._test()

        else:
            self.base_option['loaddir'] = self.base_option['savedir']
            self._build()
            self._train()
            self._test()


    def _initialize(self):
        self.network = NetworkGAN(\
            device=self.device, \
            config=self.config)


    def _build(self):
        self.network.build_network(\
            batch_size=self.base_option['batch_size'], \
            noise_shape=self.base_option['noise_shape'], \
            num_epoch=self.base_option['num_epoch'], \
            discriminator_learning_rate=self.base_option['discriminator_learning_rate'], \
            generator_learning_rate=self.base_option['generator_learning_rate'], \
            dataset_type=self.base_option['dataset_type'], \
            model_type=self.base_option['model_type'], \
            graph_type=self.base_option['graph_type'], \
            regularizer_type=self.base_option['regularizer_type'], \
            scope=self.base_option['scope'])


    def _train(self):
        self.network.train(\
            savedir=self.base_option['savedir'], \
            save_epoch=self.base_option['save_epoch'])


    def _test(self):
        self.network.test(\
            savedir=self.base_option['savedir'], \
            loaddir=self.base_option['loaddir'])


if __name__ == "__main__":
    gan = GAN()
    gan.run()
