# TensorFlow Generative Adversarial Networks

Repository for TensorFlow implementation of Generative Adversarial Networks (GANs).
Currently available model is the [Deep Convolutional Generative Adversarial Network (DCGAN)](https://arxiv.org/abs/1511.06434), trained on the Cifar-10 dataset.
Other models are to be updated.

## Requirements
Python packages
* `tensorflow >= 1.12.0`
* `keras >= 2.2.4` - For importing Cifar-10 dataset
* `matplotlib >= 3.0.2`- For saving image to file

## Usage
Run the main script `python3 main.py`. You can check the arguments by calling `python3 main.py -h`, and change hyperparameters like `python3 main.py -e 50 -lD 1e-7 -lG 2e-6`, which will run the script with discriminator learning rate 1e-7 and generator learning rate 2e-6 for 50 epochs.
