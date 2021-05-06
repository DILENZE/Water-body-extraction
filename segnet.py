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

import os
import datetime
import re
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.utils as keras_utils
from keras import backend as K
from Config import Config
import numpy as np
import non_local
# import .imagenet_utils
# from imagenet_utils import decode_predictions
# from imagenet_utils import _obtain_input_shape
import logging
import tensorflow as tf
import multiprocessing
import argparse


from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


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



def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = KL.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = KL.Activation('relu', name=name + '_relu')(x)
    x = KL.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = KL.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = KL.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = KL.Activation('relu', name=name + '_0_relu')(x1)
    x1 = KL.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = KL.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = KL.Activation('relu', name=name + '_1_relu')(x1)
    x1 = KL.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = KL.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def segnet(blocks,
           input_tensor=None,
           input_shape=None,
           pool_size=(2, 2)):
    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # encoder
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  #  inputs = Input(shape=input_shape)
    #x = KL.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='block1/bn1')(img_input)
    x = img_input
    conv_1 = KL.Conv2D(64, 3, activation='relu', padding = 'same', name = 'block1/conv1')(x)
   # conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(img_input)
    conv_1 = BatchNormalization()(conv_1)
   # conv_1 = Activation("relu")(conv_1)
    conv_2 = KL.Conv2D(64, 3, activation='relu', padding='same', name='block1/conv2')(conv_1)
   # conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
  #  conv_2 = Activation("relu")(conv_2)
    #pool_1 = KL.MaxPooling2D(2, strides=2, name='block1/pool')(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    conv_3 = KL.Conv2D(128, 3, activation='relu', padding='same', name='block2/conv1')(pool_1)
  #  conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    #conv_3 = Activation("relu")(conv_3)
    conv_4 = KL.Conv2D(128, 3, activation='relu', padding='same', name='block2/conv2')(conv_3)
   # conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
   # conv_4 = Activation("relu")(conv_4)
    #pool_2 = KL.MaxPooling2D(2, strides=2, name='block2/pool')(x)
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 =  KL.Conv2D(256, 3, activation='relu', padding='same', name='block3/conv1')(pool_2)
    #conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    #conv_5 = Activation("relu")(conv_5)

    conv_6 =  KL.Conv2D(256, 3, activation='relu', padding='same', name='block3/conv2')(conv_5)
    #conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    #conv_6 = Activation("relu")(conv_6)
    conv_7 = KL.Conv2D(256, 3, activation='relu', padding='same', name='block3/conv3')(conv_6)
    #conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    #conv_7 = Activation("relu")(conv_7)
    #pool_3 = KL.MaxPooling2D(2, strides=2, name='block3/pool')(conv_7) #output 28*28*256
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = KL.Conv2D(512, 3, activation='relu', padding='same', name='block4/conv1')(pool_3)
    #conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    #conv_8 = Activation("relu")(conv_8)
    conv_9 = KL.Conv2D(512, 3, activation='relu', padding='same', name='block4/conv2')(conv_8)
    #conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    #conv_9 = Activation("relu")(conv_9)

    conv_10 = KL.Conv2D(512, 3, activation='relu', padding='same', name='block4/conv3')(conv_9)
    #conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    #conv_10 = Activation("relu")(conv_10)
    #pool_4 = KL.MaxPooling2D(2, strides=2, name='block4/pool')(conv_10) #output 14*14*512
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)


    conv_11 = KL.Conv2D(512, 3, activation='relu', padding='same', name='block5/conv1')(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_12 = KL.Conv2D(512, 3, activation='relu', padding='same', name='block5/conv2')(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_13 = KL.Conv2D(512, 3, activation='relu', padding='same', name='block5/conv3')(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    #pool_5 = KL.MaxPooling2D(2, strides=2, name='block5/pool')(conv_13) #output 7*7*512

    #conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    
    #conv_11 = Activation("relu")(conv_11)
    #conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    
    #conv_12 = Activation("relu")(conv_12)
    #conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
   
    #conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build encoder done..")

    # decoder
    #unpool_1 = KL.MaxUnpooling2D(2, strides=2, name='mask_unpool1')(pool_5)
    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = KL.Conv2D(512, 3, activation='relu', padding='same', name='mask_conv1')(unpool_1)
    #conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    #conv_14 = Activation("relu")(conv_14)
    conv_15 = KL.Conv2D(512, 3, activation='relu', padding='same', name='mask_conv2')(conv_14)

    conv_15 = BatchNormalization()(conv_15)
    conv_16 = KL.Conv2D(512, 3, activation='relu', padding='same', name='mask_conv3')(conv_14)

    # conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    
    # conv_15 = Activation("relu")(conv_15)

    # conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    # conv_16 = Activation("relu")(conv_16)

    #unpool_2 = KL.MaxUnpooling2D(2, strides=2, name='mask_unpool1')(conv_16)
    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = KL.Conv2D(512, 3, activation='relu', padding='same', name='mask_conv4')(unpool_2)

    # conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    # conv_17 = Activation("relu")(conv_17)

    conv_18 = KL.Conv2D(512, 3, activation='relu', padding='same', name='mask_conv5')(conv_17)

    # conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    # conv_18 = Activation("relu")(conv_18)

    conv_19 = KL.Conv2D(256, 3, activation='relu', padding='same', name='mask_conv6')(conv_18)

    # conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    # conv_19 = Activation("relu")(conv_19)

    #unpool_3 = KL.MaxUnpooling2D(2, strides=2, name='mask_unpool1')(conv_19)
    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = KL.Conv2D(256, 3, activation='relu', padding='same', name='mask_conv7')(unpool_3)

    # conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    # conv_20 = Activation("relu")(conv_20)

    conv_21 = KL.Conv2D(256, 3, activation='relu', padding='same', name='mask_conv8')(conv_20)

    # conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    # conv_21 = Activation("relu")(conv_21)

    conv_22 = KL.Conv2D(128, 3, activation='relu', padding='same', name='mask_conv9')(conv_21)

    # conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    # conv_22 = Activation("relu")(conv_22)

    #unpool_4 = KL.MaxUnpooling2D(2, strides=2, name='mask_unpool1')(conv_22)
    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = KL.Conv2D(128, 3, activation='relu', padding='same', name='mask_conv10')(unpool_4)

    # conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    # conv_23 = Activation("relu")(conv_23)

    conv_24 = KL.Conv2D(64, 3, activation='relu', padding='same', name='mask_conv11')(conv_23)

    # conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    # conv_24 = Activation("relu")(conv_24)

    #unpool_5 = KL.MaxUnpooling2D(2, strides=2, name='mask_unpool1')(conv_24)
    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = KL.Conv2D(64, 3, activation='relu', padding='same', name='mask_conv12')(unpool_5)

    # conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    # conv_25 = Activation("relu")(conv_25)

    #conv_26 = KL.Conv2D(64, 3, activation='relu', padding='same', name='mask_conv2')(conv_25)
    #conv_26 = KL.Conv2D(2,(1,1),strides= 1,activation = 'relu')(conv_25)
    outputs = KL.Conv2D(1,(1,1),strides=1,activation='sigmoid')(conv_25)

    #conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    #conv_26 = BatchNormalization()(conv_26)
    #outputs = Activation("sigmoid")(conv_26)

    #conv_26 = Reshape(
      #      (input_shape[0]*input_shape[1], n_labels),
      #      input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

   #outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

  #  model = Model(inputs=inputs, outputs=outputs, name="SegNet")


# Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = KE.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    #model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    model = KM.Model(inputs, outputs, name='segnet')
    return model


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



    def  build(self,mode,config,
                input_tensor=None):
        """Build DeFCN architecture."""

        model = segnet(config.DenseNet,
                            input_tensor, config.INPUT_SHAPE)

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
