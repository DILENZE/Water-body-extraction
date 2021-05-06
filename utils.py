# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:29:51 2020

@author: DILENZE
"""

"""
This is the utils for deep learning, implemented with TensorFlow.
@ Author: Yu Sun. vxallset@outlook.com
@ Date created: Jun 04, 2019
@ Last modified: Jun 06, 2019
"""
import tensorflow as tf


def leaky_Relu(input, name=''):
    return tf.nn.leaky_relu(input, alpha=0.1, name=name + '_relu')


def conv_2d(inputs, channels, kernel_size=3, strides=1, batch_normalization=True, activation=None,
            name='', padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), is_training=True):

    output = tf.layers.conv2d(inputs=inputs, filters=channels, kernel_size=kernel_size, strides=strides,
                              padding=padding, name=name + '_conv', kernel_initializer=kernel_initializer)
    name = name + '_conv'

    if batch_normalization:
        output = tf.layers.batch_normalization(output, axis=-1, momentum=0.9, name=name+'_bn', training=is_training)
        name = name + '_bn'

    if activation:
        output = activation(output, name=name)

    return output


def down_sampling(input, method='strided_convolution', rate=2, name='', activation=leaky_Relu, is_training=True):
    assert method == 'max_pooling' or method == 'strided_convolution', \
        'Unknown type of down_sample method! "strided_convolution" and "' \
        'max_pooling" are expected, but "' + method + '" is provided!'
    output = input

    if method == 'strided_convolution':
        _, _, _, channels = input.get_shape()
        channels = channels.value
        output = input
        loop_index = 1
        new_rate = rate
        while new_rate > 1:
            assert new_rate % 2 == 0, 'The rate of down_sampling (using "strided_convolution") must be the power of ' \
                                      '2, but "{}" is provided!'.format(rate)
            output = conv_2d(output, channels=channels * (2 ** loop_index), strides=2, activation=activation,
                             name=name + 'down_sampling' + '_x' + str(loop_index * 2), is_training=is_training)
            loop_index += 1
            new_rate = int(new_rate / 2)

    elif method == 'max_pooling':
        output = tf.layers.max_pooling2d(input, pool_size=rate, strides=rate, name=name+'_max_pooling')

    return output


def up_sampling(input, channels, method='nearest_neighbor', rate=2, name='', activation=leaky_Relu, is_training=True):
    assert method == 'nearest_neighbor', 'Only "nearest_neighbor" method is supported now! ' \
                                         'However, "' + method + '" is provided.'
    output = input
    if method == 'nearest_neighbor':
        _, x, y, _= input.get_shape()
        x = x.value
        y = y.value

        output = tf.image.resize_nearest_neighbor(input, size=(x*rate, y*rate), name=name + '_upsampling')
        name += '_upsampling'
        output = conv_2d(output, channels=channels, kernel_size=1, activation=activation,
                         name=name + '_align_channels', is_training=is_training)

    return output


# Repeated multi-scale fusion (namely the exchange block) within a stage (the input and the output has the same number
# of sub-networks)
def exchange_within_stage(inputs, name='exchange_within_stage', is_training=True):
    with tf.variable_scope(name):
        subnetworks_number = len(inputs)
        outputs = []

        # suppose i is the index of the input sub-network, o is the index of the output sub-network
        for o in range(subnetworks_number):
            one_subnetwork = 0
            for i in range(subnetworks_number):
                if i == o:
                    # if in the same resolution
                    temp_subnetwork = inputs[i]
                elif i - o < 0:
                    # if the input resolution is greater the output resolution, down-sampling with rate
                    # of 2 ** (o - i)
                    temp_subnetwork = down_sampling(inputs[i], rate=2 ** (o - i), name='i_{}_o_{}'.format(i, o),
                                                    is_training=is_training)
                else:
                    # if the input resolution is smaller the output resolution, up-sampling with rate of
                    # 2 ** (o - i)
                    _, _, _, c = inputs[o].get_shape()
                    temp_subnetwork = up_sampling(inputs[i], channels=c, rate=2 ** (i - o),
                                                  name='i_{}_o_{}'.format(i, o), is_training=is_training)
                one_subnetwork = tf.add(temp_subnetwork, one_subnetwork, name='add_i_{}_o_{}'.format(i, o))
            outputs.append(one_subnetwork)
    return outputs


# Repeated multi-scale fusion (namely the exchange block) between two stages (the input and the output has the same
# number of sub-networks)
def exchange_between_stage(inputs, name='exchange_between_stage', is_training=True):
    subnetworks_number = len(inputs)
    outputs = []

    # suppose i is the index of the input sub-network, o is the index of the output sub-network
    for o in range(subnetworks_number):
        one_subnetwork = 0
        for i in range(subnetworks_number):
            if i == o:
                # if in the same resolution
                temp_subnetwork = inputs[i]
            elif i - o < 0:
                # if the input resolution is greater the output resolution, down-sampling with rate
                # of 2 ** (o - i)
                temp_subnetwork = down_sampling(inputs[i], rate=2 ** (o - i), name='i_{}_o_{}'.format(i, o),
                                                is_training=is_training)
            else:
                # if the input resolution is smaller the output resolution, up-sampling with rate of
                # 2 ** (o - i)
                _, _, _, c = inputs[o].get_shape()
                temp_subnetwork = up_sampling(inputs[i], channels=c, rate=2 ** (i - o),
                                              name='i_{}_o_{}'.format(i, o), is_training=is_training)
            one_subnetwork = tf.add(temp_subnetwork, one_subnetwork, name='add_i_{}_o_{}'.format(i, o))
        outputs.append(one_subnetwork)
    one_subnetwork = down_sampling(inputs[-1], rate=2, name='new_resolution', is_training=is_training)
    outputs.append(one_subnetwork)
    return outputs


def residual_unit_bottleneck(input, name='RU_bottleneck', channels=64, is_training=True,c=1):
    """
    Residual unit with bottleneck design, default width is 64.
    :param input:
    :param name:
    :return:
    """
    #_, _, _, c = input.get_shape()
    conv_1x1_1 = conv_2d(input, channels=channels, kernel_size=1, activation=leaky_Relu, name=name + '_conv1x1_1',
                         is_training = is_training)
    conv_3x3 = conv_2d(conv_1x1_1, channels=channels, activation=leaky_Relu, name=name + '_conv3x3',
                       is_training=is_training)
    conv_1x1_2 = conv_2d(conv_3x3, channels=c, kernel_size=1, name=name + '_conv1x1_2', is_training=is_training)
    _output = tf.add(input, conv_1x1_2, name=name + '_add')
    output = leaky_Relu(_output, name=name + '_out')
    return output


def residual_unit(input, name='RU', is_training=True):
    """
    Residual unit with two 3 x 3 convolution layers.
    :param input:
    :param name:
    :return:
    """
    _, _, _, channels = input.get_shape()
    conv3x3_1 = conv_2d(inputs=input, channels=channels, activation=leaky_Relu, name=name + '_conv3x3_1',
                        is_training=is_training)
    conv3x3_2 = conv_2d(inputs=conv3x3_1, channels=channels, name=name + '_conv3x3_2', is_training=is_training)
    _output = tf.add(input, conv3x3_2, name=name + '_add')
    output = leaky_Relu(_output, name=name + '_out')
    return output


def exchange_block(inputs, name='exchange_block', is_training=True):
    with tf.variable_scope(name):
        output = []
        level = 0
        for input in inputs:
            sub_network = residual_unit(input, name='level{}RU1'.format(level), is_training=is_training)
            sub_network = residual_unit(sub_network, name='level{}RU2'.format(level), is_training=is_training)
            sub_network = residual_unit(sub_network, name='level{}RU3'.format(level), is_training=is_training)
            sub_network = residual_unit(sub_network, name='level{}RU4'.format(level), is_training=is_training)
            output.append(sub_network)
            level += 1
        outputs = exchange_within_stage(output, is_training=is_training)
    return outputs