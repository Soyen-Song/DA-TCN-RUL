# -*- coding: utf-8 -*-
from keras.layers import SeparableConv2D, Activation, Dropout, Flatten, Dense, Input, add, GlobalAveragePooling1D, \
    Conv2D, LeakyReLU
from keras.layers import SeparableConv1D, Conv1D, MaxPooling2D, GlobalMaxPooling1D, AveragePooling2D, Cropping2D, \
    BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.regularizers import l2
from keras.models import Model, Sequential
import os
from PIL import Image
import numpy as np
import keras
import time
import scipy.io as sio
import random
import h5py
import time
from scipy.misc import imresize
from numpy import *
from keras.layers import Activation, multiply, LeakyReLU
from keras.models import *
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model
import tensorflow as tf
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras_radam import RAdam
from keras_lookahead import Lookahead
# from lookahead import Lookahead
import keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def gated_activation(x):
    # Used in PixelCNN and WaveNet
    tanh = Activation('tanh')(x)
    sigmoid = Activation('sigmoid')(x)
    return multiply([tanh, sigmoid])  # 逐元素积


def myScore(Target, Pred):
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp


def scheduler(epochs):
    learning_rate_init = 0.001
    if epochs <= 50:
        learning_rate_init = 0.001
    elif epochs <= 80:
        learning_rate_init = 0.0005
    elif epochs <= 100:
        learning_rate_init = 0.0001
    return learning_rate_init


n_filters = 32


def squeeze_excite_block(tensor, filters, ratio):
    init = tensor
    Size = int(tensor.shape[1])
    se_shape = (Size, 1, filters)

    se = TimeDistributed(GlobalAveragePooling1D())(init)
    se = Reshape(se_shape)(se)
    # 'relu'
    se = TimeDistributed(Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False))(se)
    # 'sigmoid'
    se = TimeDistributed(Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False))(se)
    x = multiply([init, se])
    return x


def conv_block(inputs, filters, ratio, dilation):
    c1 = TimeDistributed(Conv1D(filters, kernel_size=5, strides=1, padding='same', dilation_rate=dilation))(inputs)
    # c1 = squeeze_excite_block(c1, filters, ratio)
    a1 = Activation(gated_activation)(c1)
    n1 = BatchNormalization(momentum=0.6)(a1)

    c2 = TimeDistributed(Conv1D(filters, kernel_size=5, strides=1, padding='same', dilation_rate=dilation))(n1)
    # c2 = squeeze_excite_block(c2, filters, ratio)
    a2 = Activation(gated_activation)(c2)
    n2 = BatchNormalization(momentum=0.6)(a2)
    # n2 = squeeze_excite_block(n2, filters, ratio)

    n = keras.layers.concatenate([n1, n2])
    n = squeeze_excite_block(n, 2 * filters, 2 * ratio)
    return n


def residual_block(inputs, factor, filters, ratio):
    dilation = 2 ** factor
    # Residual block
    n1 = conv_block(inputs, filters, ratio, dilation)
    # n1 = conv_block(n1, 2 * filters, 2 * ratio, 2)
    # Residual connection
    residual = TimeDistributed(SeparableConv1D(1, kernel_size=1, padding='same'))(n1)
    # print residual.shape
    outputs = keras.layers.add([inputs, residual])
    return outputs
    # return Model(inputs=inputs, outputs=outputs, name='residual_block_{}'.format(factor))


