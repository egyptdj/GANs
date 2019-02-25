import os
import csv
import warnings
import argparse
import tensorflow as tf


def parse():
    parser = argparse.ArgumentParser(description='- GANs - implemented by github @egyptdj')
    parser.add_argument('-g', '--gpu_device', type=int, default=None, help='ID of gpu device to build the graph with. CPU is used if not specified')
    parser.add_argument('-s', '--scope', type=str, default='GANs', help='outermost scope name of the network')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of data to place in a minibatch')
    parser.add_argument('-n', '--noise_shape', type=int, default=100, help='length of the noise vector')
    parser.add_argument('-tD', '--dataset_type', type=str, default='mnist', help='type of the dataset [mnist/cifar10/cifar100]')
    parser.add_argument('-tM', '--model_type', type=str, default='gan', help='type of the GAN model [gan/cgan/dcgan]')
    parser.add_argument('-tG', '--graph_type', type=str, default='gan', help='type of the GAN graph [gan/lsgan/wgan/wgan-gp/geogan]')
    parser.add_argument('-lD', '--discriminator_learning_rate', type=float, default=1e-3, help='learning rate of the discriminator training')
    parser.add_argument('-lG', '--generator_learning_rate', type=float, default=1e-4, help='learning rate of the generator training')
    parser.add_argument('-dS', '--savedir', type=str, default='./GANs', help='directory path to save the trained generator model and/or the resulting image')
    parser.add_argument('-dL', '--loaddir', type=str, default=None, help='directory path to load a saved generator model (optional).')
    parser.add_argument('-eN', '--num_epoch', type=int, default=1000, help='total number of epochs to train')
    parser.add_argument('-eS', '--save_epoch', type=int, default=50, help='save model at every specified epochs')
    parser.add_argument('--allow_soft_placement', type=bool, default=True, help='TensorFlow device option: allow soft placement')
    parser.add_argument('--allow_growth', type=bool, default=False, help='TensorFlow device option: allow growth')
    parser.add_argument('--per_process_gpu_memory_fraction', type=float, default=None, help='TensorFlow device option: per process memory fraction')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='TensorFlow device option: log device placement')
    opt_dict = vars(parser.parse_args())

    if not os.path.exists(opt_dict['savedir']): os.makedirs(opt_dict['savedir'])
    with open(os.path.join(opt_dict['savedir'],"argv.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(opt_dict.items())

    return opt_dict


def gpu(device_id, log_device_placement=False, allow_soft_placement=True, per_process_gpu_memory_fraction=False, allow_growth=False):
    if not tf.test.is_gpu_available or device_id==None:
        gpu_config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        return gpu_config, "/CPU:0"

    else:
        if per_process_gpu_memory_fraction:
            gpu_config = tf.ConfigProto(log_device_placement=log_device_placement, per_process_gpu_memory_fraction=per_process_gpu_memory_fraction, allow_soft_placement=allow_soft_placement)
        else:
            gpu_config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        gpu_config.gpu_options.allow_growth = allow_growth

        return gpu_config, "/device:GPU:{}".format(device_id)
