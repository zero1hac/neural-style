import numpy as np
import scipy.io
import tensorflow as tf


def load_net(data_path, image):
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')
    data = scipy.io.loadmat(data)
    avg = data['normalization'][0][0][0]
    avg_pixel = np.mean(avg, axis=(0, 1))
    wts = data['layers'][0]

    curr = image
    net = {}

    for i, name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels, bias = wts[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            curr = conv_layer(curr, kernels, bias)
        elif layer_type == 'pool':
            curr = pool_layer(curr)
        elif layer_type == 'relu':
            curr = tf.nn.relu(curr)
        net[name] = curr

    return net, avg_pixel


def conv_layer(in_, wts, bias):
    conv = tf.nn.conv2d(in_, tf.constant(wts), strides=(1, 1, 1, 1), padding='SAME')

    return tf.nn.bias_add(conv, bias)


def pool_layer(in_):
    return tf.nn.max_pool(in_, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def preprocess(im, avg_pixel):
    return im - avg_pixel


def deprocess(im, avg_pixel):
    return im + avg_pixel
