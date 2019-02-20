from dataset import *
from graph import *
from model import *
from session import *


class NetworkGAN:
    def __init__(self, device, config):
        self.device = device
        self.config = config


    def build_network(self, batch_size, noise_shape, num_epoch, discriminator_learning_rate, generator_learning_rate, scope):
        # 1. Define dataset object
        self.dataset = DatasetCifar10(batch_size=batch_size, noise_shape=noise_shape)
        self.image_shape, self.noise_shape = self.dataset.get_shape()

        # 2. Define graph and model object
        self.model = ModelDCGAN(device=self.device, scope=scope+"_model")
        self.graph = GraphGAN(device=self.device, scope=scope+"_graph")
        self.graph.define_nodes(image_shape=self.image_shape, noise_shape=self.noise_shape)
        self.graph.build_model(model=self.model)
        self.graph.build_graph()

        # 3. Define session object
        self.session = SessionGAN(config=self.config)
        self.session.build_session(dataset=self.dataset, graph=self.graph, num_epoch=num_epoch, discriminator_learning_rate=discriminator_learning_rate, generator_learning_rate=generator_learning_rate)


    def train(self, savedir, save_epoch):
        self.session.train_graph(savedir=savedir, save_epoch=save_epoch)


    def test(self, savedir, loaddir):
        self.session = SessionGAN(config=self.config)
        self.session.test_graph(savedir=savedir, loaddir=loaddir)
