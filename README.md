# TensorFlow Generative Adversarial Networks

Repository for TensorFlow implementation of Generative Adversarial Networks (GANs).

## Background
TensorFlow implementation of GAN and its variants can be easily found from GitHub.
Despite the fact that most GANs share the minimax-game philosophy with some modification in its model structure (e.g. DCGAN) or graph operation (e.g., LSGAN, WGAN), there has not been a single module that can flexibly change these modifications.

it takes some effort to understand and further modify scripts using custom operations. Considerin

## Requirements
Python3 with following packages
- `tensorflow >= 1.12.0`
- `keras >= 2.2.4` - for importing datasets
- `matplotlib >= 3.0.2`- for saving image

## Usage
Run the main script `python3 main.py`. You can check the arguments by calling `python3 main.py -h`, and change hyperparameters like `python3 main.py -e 50 -lD 1e-7 -lG 2e-6`, which will run the script with discriminator learning rate 1e-7 and generator learning rate 2e-6 for 50 epochs.

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

Other model options are to be updated.

## Reference
- GAN: [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)
- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- LSGAN: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- WGAN: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- WGAN-GP: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- GEOGAN: [Geometric GAN](https://arxiv.org/abs/1705.02894)
