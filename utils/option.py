import os
import csv
import warnings
import argparse
import tensorflow as tf

def parse():
    parser = argparse.ArgumentParser(description='GANs, implemented by github @egyptdj')
    parser.add_argument('-g', '--gpu_device', type=int, default=0, help='ID of gpu device to build graph')
    parser.add_argument('-e', '--num_epoch', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of data to place in a minibatch')
    parser.add_argument('-s', '--scope', type=str, default='GAN', help='outermost scope of the network')
    parser.add_argument('-lD', '--discriminator_learning_rate', type=float, default=1e-3, help='learning rate of the discriminator training')
    parser.add_argument('-lG', '--generator_learning_rate', type=float, default=1e-4, help='learning rate of the generator training')
    parser.add_argument('-dS', '--savedir', type=str, default='./GAN', help='directory path to save full model')
    parser.add_argument('-dL', '--loaddir', type=str, default='./GAN', help='directory path to save full model')
    parser.add_argument('--save_epoch', type=int, default=100, help='save model at every defined epochs')

    opt_dict = vars(parser.parse_args())

    if not os.path.exists(opt_dict['savedir']): os.makedirs(opt_dict['savedir'])
    with open(os.path.join(opt_dict['savedir'],"argv.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(opt_dict.items())

    return opt_dict

def gpu(device_id=0, log_device_placement=False, allow_soft_placement=True, per_process_gpu_memory_fraction=False, allow_growth=False):
    if not tf.test.is_built_with_cuda:
        if tf.test.is_gpu_available: warnings.warn("GPU is available but is not built with CUDA. Using CPU for computation.")
        gpu_config=tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        return gpu_config, "/CPU:0"

    else:
        if per_process_gpu_memory_fraction: gpu_config = tf.ConfigProto(log_device_placement=log_device_placement, per_process_gpu_memory_fraction=per_process_gpu_memory_fraction, allow_soft_placement=allow_soft_placement)
        else: gpu_config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        gpu_config.gpu_options.allow_growth = allow_growth

        return gpu_config, "/device:GPU:{}".format(device_id)
