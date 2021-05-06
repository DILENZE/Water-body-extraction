"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]X
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import layers
from keras.models import Model

from keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization,Input,DepthwiseConv2D,Add,Dropout,AveragePooling2D,Concatenate

from keras.layers import ZeroPadding2D

from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K

from keras.utils import conv_utils
import multiprocessing
import os
import datetime
import re
import keras
from keras import backend as K
from Config import Config
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.utils as keras_utils
from hrnet import *
BASE_WEIGTHS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.8/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


####################################################################################
class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x



#####################################################################################

def deeplab(weights='None', input_tensor=None, input_shape=(224, 224, 4), classes=2, backbone='xception', OS=16, alpha=1.):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

#############################################################################################################


    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation('relu')(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = _xception_block(x, [16, 16, 16], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [32, 32, 32], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [64, 64, 64], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [64, 64, 64], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [64, 128, 128], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [128, 128, 256], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                            int(np.ceil(input_shape[1] / 4))))(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if classes == 1:
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', activation="sigmoid",name=last_layer_name)(x)
    
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input

    # model = Model(inputs, x, name='deeplab')

    # # load weights

    # # if weights == 'pascal_voc':
    # #     if backbone == 'xception':
    # #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
    # #                                 WEIGHTS_PATH_X,
    # #                                 cache_subdir='models')
    # #     else:
    # #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
    # #                                 WEIGHTS_PATH_MOBILE,
    # #                                 cache_subdir='models')
    # #     model.load_weights(weights_path, by_name=True)
    # return model


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = KE.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='deeplab')
    return model
   
############################################################################################################

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #  if input_tensor is not None:
    #      inputs = KE.get_source_inputs(input_tensor)
    #  else:
    #      inputs = img_input

    # # # Create model.
    #  if blocks == [6, 12, 24, 16]:
    #      model = KE.Model(inputs, x, name='densenet121')
    #  elif blocks == [6, 12, 32, 32]:
    #      model = KM.Model(inputs, x, name='densenet169')
    #  elif blocks == [6, 12, 48, 32]:
    #      model = KM.Model(inputs, x, name='densenet201')
    #  else:
    #      model = KM.Model(inputs, x, name='densenet')
    #  return model
##################################################################
#                       loss function
##################################################################
def mask_loss_graph(target_masks,pred_mask):
    loss=K.binary_crossentropy(target=target_masks,output=pred_mask)
    loss=K.mean(loss)*10
    return loss
##################################################################
#def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    #def focal_loss_fixed(y_true, y_pred):
        #pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        #pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        #return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    #return focal_loss_fixed
#def get_weight(shape,lamda):
   # var = tf.Variable(tf.random_normal(shape=shape),dtype=tf.float32)

   # tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamda)(var))
  #  return varl2
#def mask_loss_graph(target_masks,pred_mask,alpha=0.75,gamma=2):
  #  pred_mask = tf.nn.softmax(pred_mask)
   # loss=K.binary_crossentropy(target=target_masks,output=pred_mask)
   # loss=K.mean(loss)
   # tf.add_to_collection("losses",loss)
   # loss = tf.add_n(tf.get_collection("losses"))
   # return loss
##################################################################
#                        data generator
##################################################################
def load_image_mask(dataset,image_id,augment=False):
    """
    load image and label according to the image_id
    """
    image = dataset.load_image(image_id)
    mask = dataset.load_mask(image_id)
    return image,mask


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def data_generator(dataset,config,shuffle=True, augment=False):
    """A generator that returns images and corresponding target mask.

    dataset: The Dataset object to pick data from
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns one list, inputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - mask: [batch,H,W].
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.

            image_index = (image_index + 1) % len(image_ids)

            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]
            #image_meta:image_id,image_shape,windows.active_class_ids
            image,img_mask=load_image_mask(dataset,image_id)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (config.BATCH_SIZE ,)+ image.shape, dtype=np.float32)

                batch_masks = np.zeros((config.BATCH_SIZE,)+img_mask.shape,dtype=np.float32)

            batch_images[b] = image

            batch_masks[b] = img_mask
            b += 1

            # Batch full?
            # input_image,input_labels
            if b >= config.BATCH_SIZE:
                batch_masks=np.reshape(batch_masks,[config.BATCH_SIZE,224,224,1])
                inputs = (batch_images,batch_masks)
                yield inputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise
##################################################################
#                           models
##################################################################

# def preprocess_input(x, data_format=None):
#     """Preprocesses a numpy array encoding a batch of images.
#
#     # Arguments
#         x: a 3D or 4D numpy array consists of RGB values within [0, 255].
#         data_format: data format of the image tensor.
#
#     # Returns
#         Preprocessed array.
#     """
#     return imagenet_utils.preprocess_input(x, data_format, mode='torch')

##############################################################
#                       DeFCN Class
##############################################################
class DeFCN():
    """Encapsulates the DeFCN model functionality.
    the actual Keras model is in the keras_model properity.
    """
    def __init__(self,mode,config,model_dir):
        """
        :param mode: Either "training" or "inference"
        :param config:  A Sub-class of the Config class
        :param model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training','inference']
        self.mode=mode
        self.config=config
        self.model_dir=model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)



    def build(self,mode,config,
                input_tensor=None):
        """Build DeFCN architecture."""
        #print(input_tensor.shape)
        #model = HRNet(input_tensor=None,input_shape=[4,224,224,1],c=1)
        model = deeplab(weights='None', input_tensor=None, input_shape=(224, 224, 1), classes=1, backbone='mobilenetv2', OS=16, alpha=1.)
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
               model directory.
        Returns:
            he path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]

        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        if self.mode=='training':
            dir_name = os.path.join(self.model_dir, dir_names[-2])
            os.rmdir(os.path.join(self.model_dir, dir_names[-1]))
        else:
            dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("DeFCN"), checkpoints)
        checkpoints = sorted(checkpoints)
        print(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self,filepath,by_name=False,exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weight(self):
        """Downloads ImageNet  trained weights form Keras.
        Return path to weights file."""
        #return weights_path
    def compile(self,learning_rate,momentum):
        """Gets the model ready for training.Adds losses,regulatization,and 
        metrics.Then call the Kerass compile() function"""
        optimizer = keras.optimizers.SGD(lr=learning_rate,
                                         momentum=momentum,nesterov=True)
        self.keras_model.compile(optimizer=optimizer,
                                 loss=mask_loss_graph,metrics=['accuracy'])

        

    def set_trainable(self,layer_regex,keras_modle=None,indent=0,verbose=1):
        """Sets model layres as trainable if their names match  the given regualr expression"""
    def set_log_dir(self,model_path=None):
        """Set the model log directory and epoch counter.
        model_path:If None ,or a format different form what this code uses then set a new 
        log directory and start epochs from 0. Otherwise,extract  the log directory and 
        the epoch counter form the file name.
        """
        if self.mode=='training':
            self.epoch=0
            now=datetime.datetime.now()
            #if we hanbe a model path with date and epochs use them
            if model_path:
                # Continue form we left of .Get epoch and date form the file name
                # A sample model path might look like:
                #/path/to/logs/coco2017.../DeFCN_0001.h5
                regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/DeFCN\_[\w-]+(\d{4})\.h5"
                m = re.match(regex,model_path)
                if m:
                    now=datetime.datetime(int(m.group(1)),int(m.group(2)),int(m.group(3)),
                                          int(m.group(4)),int(m.group(5)))
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    self.epoch = int(m.group(6)) - 1 + 1
                    print('Re-starting from epoch %d' % self.epoch)

                    # Directory for training logs
            self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                    self.config.NAME.lower(), now))
                # Create log_dir if not exists
            if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)

                # Path to save after each epoch. Include placeholders that get filled by Keras.
            self.checkpoint_path = os.path.join(self.log_dir, "DeFCN_{}_*epoch*.h5".format(
                    self.config.NAME.lower()))
            self.checkpoint_path = self.checkpoint_path.replace(
                    "*epoch*", "{epoch:04d}")
    def train(self,train_dataset,val_datset,learning_rate,epochs,augmentation=None):
        """Train the model.
                train_dataset, val_dataset: Training and validation Dataset objects.
                learning_rate: The learning rate to train with
                epochs: Number of training epochs. Note that previous training epochs
                        are considered to be done alreay, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
                layers: Allows selecting wich layers to train. It can be:
                    - A regular expression to match layer names to train
                    - One of these predefined values:
                      heads: The RPN, classifier and mask heads of the network
                      all: All the layers
                      3+: Train Resnet stage 3 and up
                      4+: Train Resnet stage 4 and up
                      5+: Train Resnet stage 5 and up
                augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
                    augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
                    flips images right/left 50% of the time. You can pass complex
                    augmentations as well. This augmentation applies 50% of the
                    time, and when it does it flips images right/left half the time
                    and adds a Gausssian blur with a random sigma in range 0 to 5.
                        augmentation = imgaug.augmenters.Sometimes(0.5, [
                            imgaug.augmenters.Fliplr(0.5),
                            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                        ])
        """
        assert self.mode == "training", "Create model in training mode."
        train_generator = data_generator(train_dataset, self.config)
        val_generator = data_generator(val_datset, self.config)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        # TODO:set trainable layrers
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )

        self.epoch = max(self.epoch, epochs)
    def detect(self,images,verbose=0):
        """Runs the detection pipeline.
                images: List of images, potentially of different sizes.
                Returns  a mask of image.
        """
        assert self.mode == "inference", "Create model in inference mode."
        images=np.reshape(images,[-1,224,224,1])
        result=self.keras_model.predict(images,batch_size=2)
        return np.round(result)
