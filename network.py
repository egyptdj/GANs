from dataset import *
from graph import *
from model import *
from session import *


class NetworkGAN:
    def __init__(self, device, config):
        self.device = device
        self.config = config


    def build_network(self, batch_size, noise_shape, num_epoch, discriminator_learning_rate, generator_learning_rate, dataset_type, model_type, graph_type, scope):
        # 1. BUILD DATASET OBJECT
        self.dataset = DatasetGAN(batch_size=batch_size, noise_shape=noise_shape)
        self.dataset.build_dataset(type=dataset_type)
        self.image_shape, self.noise_shape = self.dataset.get_shape()

        # 2. BUILD MODEL AND GRAPH OBJECT
        self.model = ModelGAN(device=self.device, scope=scope+"_model")
        self.graph = GraphGAN(device=self.device, scope=scope+"_graph")
        self.graph.define_nodes(image_shape=self.image_shape, noise_shape=self.noise_shape)
        self.graph.build_model(model=self.model, type=model_type)
        self.graph.build_graph(type=graph_type)

        # 3. BUILD SESSION OBJECT
        self.session = SessionGAN(config=self.config)
        self.session.build_session(dataset=self.dataset, graph=self.graph, num_epoch=num_epoch, discriminator_learning_rate=discriminator_learning_rate, generator_learning_rate=generator_learning_rate)


    def train(self, savedir, save_epoch):
        self.session.train_graph(savedir=savedir, save_epoch=save_epoch)


    def test(self, savedir, loaddir):
        self.session = SessionGAN(config=self.config)
        self.session.test_graph(savedir=savedir, loaddir=loaddir)
