from dataset import *
from graph import *
from model import *
from session import *


class NetworkDCGAN:
    def __init__(self, batch_size, num_epoch, discriminator_learning_rate, generator_learning_rate, scope):
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.scope = scope


    def build_network(self, device, config):
        # 1. Define dataset object
        self.dataset = Cifar10(batch_size=self.batch_size, noise_shape=100)
        self.image_shape, self.noise_shape = self.dataset.get_shape()
        self.device = device
        self.config = config

        # 2. Define graph and model object
        self.model = ModelDCGAN(device=self.device, scope=self.scope+"_model")
        self.graph = GraphDCGAN(device=self.device, scope=self.scope+"_graph")
        self.graph.define_nodes(image_shape=self.image_shape, noise_shape=self.noise_shape)
        self.graph.build_model(model=self.model)
        self.graph.build_graph()

        # 3. Define session object
        self.session = SessionDCGAN(config=self.config)
        self.session.build_session(dataset=self.dataset, graph=self.graph, num_epoch=self.num_epoch, discriminator_learning_rate=self.discriminator_learning_rate, generator_learning_rate=self.generator_learning_rate)


    def train(self, savedir, save_epoch):
        self.session.train_graph(savedir=savedir, save_epoch=save_epoch)


    def test(self, config, savedir, loaddir):
        self.session = SessionDCGAN(config=config)
        self.session.test_graph(savedir=savedir, loaddir=loaddir)
