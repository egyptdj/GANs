# TensorFlow Generative Adversarial Networks

Repository for TensorFlow implementation of Generative Adversarial Networks (GANs).

## Background
TensorFlow implementation of GAN and its variants can be easily found from GitHub.
Despite the fact that most GANs share the minimax-game philosophy with some modification in its **model** (**model** refers to the multi-layer perceptron structure in this document. i.e. layers of D and G) or **graph** (**graph** refers to the operations applied to the **model** in this document. e.g. optimizers, losses and regularizers of the network), there has been no implementation that can separately switch between these modifications.
Say that you want to try a network that has a **model** of the DCGAN, combined with a **graph** of the GEOGAN with hinge loss.
It will take a substantial amount of effort even when not writing the whole script from scratch, since the implementation of DCGAN and GEOGAN may be coded in a different style, with many customized operations.

To address this issue, I built the GANs with
1. clear demarcation of the **model** and the **graph** structure
2. using only TensorFlow module functions without custom wrapping of the functions.

This implementation approach offers appealing advantages including
1. flexibility of separately applying various **model** and **graph** structures,
2. and ease of reading/modifying the script to make a new GAN yourself.

I also considered maximizing the compatibility with the TensorBoard, so that clean summary of the network can be visualized.

## Requirements
Python3 with following packages
- `tensorflow >= 1.12.0`
- `keras >= 2.2.4` for importing datasets
- `matplotlib >= 3.0.2` for saving images

## Usage
Run the main script `python3 main.py`. You can check the arguments by calling `python3 main.py -h`, and change hyperparameters like `python3 main.py -g 0 -e 100 -b 128 -n 100 -tM dcgan -lD 1e-7`. It will run the DCGAN **model** combined with the vanilla GAN **graph**, with batch size, noise vector length, and discriminator learning rate set to 128, 100, and 1e-7 respectively, for 100 epochs using the GPU with ID 0.

## Options
Datasets
- MNIST
- Cifar-10
- Cifar-100

Models
- GAN
- DCGAN

Graphs
- GAN
- LSGAN
- WGAN
- WGAN-GP
- GEOGAN

Other model options are being updated.

## Implementation
- `main.py`

  Main function that runs the script. Hyperparameters can be passed as arguments.

- `network.py`

  Defines the class `NetworkGAN`, which is the full network that wraps the **dataset**, **model**, **graph**, and **session**.

- `dataset.py`

  Defines the class `DatasetGAN`, which can import datasets and return image, label, and noise.

- `model.py`

  Defines the class `ModelGAN`, which can build the **model** by creating `GeneratorComponentGAN` and `DiscriminatorComponentGAN` objects from the `model_components` module.
  - `model_components.py`

    Defines the class `GeneratorComponentGAN` and the class `DiscriminatorComponentGAN`, which holds multi-layer structure of the generator and the discriminator.

- `graph.py`

  Defines the class `GraphGAN`, which can build the **graph** by defining needed graph operations such as input nodes, losses, and optimizers upon the ModelGAN **model** object.

- `session.py`

  Defines the class `SessionGAN`, which can run the dynamic **session** of the built **graph**.

## Reference
- GAN: [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)
- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- LSGAN: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- WGAN: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- WGAN-GP: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- GEOGAN: [Geometric GAN](https://arxiv.org/abs/1705.02894)
