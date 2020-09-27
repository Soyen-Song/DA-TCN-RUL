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
from keras.layers import Activation, multiply
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
from tcnet import rmse
from sklearn import linear_model
from sklearn import preprocessing

'''import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'''

regr = linear_model.LinearRegression()  # feature of linear coefficient


def fea_extract1(data):  # feature extraction of two features
    fea = []
    # print(data.shape)
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        # fea.append(np.mean(data[:,i]))
        regr.fit(x.reshape(-1, 1), np.ravel(data[:, i]))
        fea = fea + list(regr.coef_)
        # print(regr.coef_)
    return fea


def fea_extract2(data):  # feature extraction of two features
    fea = []
    # print(data.shape)
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        fea.append(np.mean(data[:, i]))
        # print(regr.coef_)
    return fea


SINGLE_ATTENTION_VECTOR = True


def attention_3d_block(inputs, TIME_STEPS):
    input_dim = int(inputs.shape[2])  # input_dim = 100
    a = Permute((2, 1))(inputs)  # a.shape = (?, 100, ?)
    a = Reshape((input_dim, TIME_STEPS))(a)  # a.shape = (?, 100, 30)this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)  # a.shape = (?, 100, 30)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)  # a.shape = (?, 30)
        a = RepeatVector(input_dim)(a)  # a.shape = (?, 100, 30) RepeatVector层将输入重复n次
    a_probs = Permute((2, 1))(a)  # a.shape = (?, 30, 100)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])  # [?, 30, 100]
    return output_attention_mul


def model_cross_TCN(inputs):
    TIME_STEPS = int(inputs.shape[1])
    input1 = attention_3d_block(inputs, TIME_STEPS)
    input1 = Permute((2, 1), name='att1')(input1)
    input1 = Reshape((winSize, Feasize, 1))(input1)
    x0 = residual_block(input1, 0, 64)
    x1 = residual_block(x0, 1, 64)
    x3 = Reshape((winSize, Feasize), name='tcn_out')(x1)
    TIME_STEPS = int(x3.shape[1])
    x3 = attention_3d_block(x3, TIME_STEPS)
    x3 = Permute((1, 2), name='att2')(x3)
    attention_mul = Flatten()(x3)
    dense_0 = Dense(100, activation='relu')(attention_mul)
    drop1 = Dropout(0.2)(dense_0)
    dense_1 = Dense(20, activation='relu')(drop1)
    return dense_1


def model_attention_applied_after_lstm():  # 两个输入：trainX_fea和inputs
    inputs = Input(shape=(winSize, Feasize, 1))
    input1 = Reshape((winSize, Feasize))(inputs)  # shape = (?, 30, 17)
    input1_v = Permute((2, 1))(input1)  # shape = (?, 17, 30)
    dense_2_v = model_cross_TCN(input1_v)
    drop2 = Dropout(0.2)(dense_2_v)
    output2 = Dense(1, activation='linear')(drop2)

    model = Model(inputs, output2)
    return model


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


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 120:
        lr *= 1e-4
    elif epoch > 90:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def scheduler(epochs):
    learning_rate_init = 0.001
    if epochs <= 80:
        learning_rate_init = 0.001
    elif epochs <= 100:
        learning_rate_init = 0.0001
    return learning_rate_init


def conv_block(inputs, filters, dilation):
    c1 = TimeDistributed(Conv1D(filters, kernel_size=5, strides=1, padding='causal', dilation_rate=dilation))(inputs)
    a1 = Activation(gated_activation)(c1)
    n1 = BatchNormalization(momentum=0.6)(a1)

    c2 = TimeDistributed(Conv1D(filters, kernel_size=5, strides=1, padding='causal', dilation_rate=dilation))(n1)
    a2 = Activation(gated_activation)(c2)
    n2 = BatchNormalization(momentum=0.6)(a2)
    return n2


def residual_block(inputs, factor, filters):
    dilation = 2 ** factor
    # Residual block
    n1 = conv_block(inputs, filters, dilation)
    residual = TimeDistributed(SeparableConv1D(1, kernel_size=1, padding='same'))(n1)
    # print residual.shape
    outputs = keras.layers.add([inputs, residual])
    return outputs


log_filepath = './sonar_dp_da'
import keras.backend as kb

if __name__ == '__main__':
    max_RUL = 125.0  # max RUL for training
    winSize = 40 # FD002: 40; FD002: 50; FD003: 40; FD004:50
    Feasize = 14
    train1X = sio.loadmat('./data_train_test/train3X.mat')
    train1X = train1X['train3X']
    train1Y = sio.loadmat('./data_train_test/train3Y.mat')
    train1Y = train1Y['train3Y']
    test1X = sio.loadmat('./data_train_test/test3X.mat')
    test1X = test1X['test3X']
    test1Y = sio.loadmat('./data_train_test/test3Y.mat')
    test1Y = test1Y['test3Y']

    trainX = train1X
    trainY = train1Y
    testX = test1X
    testY = test1Y

    trainX = np.reshape(trainX, [trainX.shape[0], winSize, Feasize, 1])
    trainY = trainY.T
    testY = testY.T
    testX = np.reshape(testX, [testX.shape[0], winSize, Feasize, 1])

    trainX_fea1 = []
    testX_fea1 = []
    trainX_fea2 = []
    testX_fea2 = []
    for i in range(len(trainX)):
        data_temp = trainX[i]
        trainX_fea1.append(fea_extract1(data_temp))
        trainX_fea2.append(fea_extract2(data_temp))

    for i in range(len(testX)):
        data_temp = testX[i]
        testX_fea1.append(fea_extract1(data_temp))
        testX_fea2.append(fea_extract2(data_temp))

    scale1 = preprocessing.MinMaxScaler().fit(trainX_fea1)
    trainX_fea1 = scale1.transform(trainX_fea1)
    testX_fea1 = scale1.transform(testX_fea1)

    scale2 = preprocessing.MinMaxScaler().fit(trainX_fea2)
    trainX_fea2 = scale2.transform(trainX_fea2)
    testX_fea2 = scale2.transform(testX_fea2)

    trainX_new = []
    testX_new = []
    for i in range(len(trainX)):
        data_temp0 = trainX[i]
        data_temp1 = np.reshape(trainX_fea1[i], [1, Feasize, 1])  # regr.coef_
        data_temp2 = np.reshape(trainX_fea2[i], [1, Feasize, 1])
        data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
        # data_temp = np.vstack((data_temp0, data_temp2))
        trainX_new.append(data_temp)
    trainX_new = np.array(trainX_new)

    for i in range(len(testX)):
        data_temp0 = testX[i]
        data_temp1 = np.reshape(testX_fea1[i], [1, Feasize, 1])  # regr.coef_
        data_temp2 = np.reshape(testX_fea2[i], [1, Feasize, 1])
        data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
        # data_temp = np.vstack((data_temp0, data_temp2))
        testX_new.append(data_temp)
    testX_new = np.array(testX_new)

    winSize = 42 # FD002: 42; FD002: 52; FD003: 42; FD004:52

    RMSE_t = []
    score_t = []
    ttime = []
    for i in range(10):  # run 10 times
        print('iteration: ', i)
        # radam = keras.optimizers.Adam(lr=scheduler(100), beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # lookahead = Lookahead(k=5, alpha=0.5)  # 初始化Lookahead
        # lookahead.inject(model)  # 插入到模型中
        # build network
        model = model_attention_applied_after_lstm()
        #print(model.summary())
        radam = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-03)
        model.compile(optimizer=radam,
                      loss='mse',
                      metrics=[rmse])

        tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        cbks = [tb_cb, change_lr]

        history = model.fit(trainX_new, trainY,
                            batch_size=512, verbose=2, nb_epoch=100, callbacks=cbks)
        start1 = time.time()
        yPreds = model.predict(testX_new)
        end1 = time.time()
        print('time2:', end1-start1)
        ttime.append(end1-start1)

        yPreds = yPreds.ravel()
        yPreds = max_RUL * yPreds
        test_rmse = np.sqrt(mean_squared_error(yPreds, max_RUL * testY))
        # test_rmse = kb.sqrt(kb.mean(kb.square(yPreds - testY), axis=-1))
        test_score = myScore(max_RUL * testY, yPreds)
        print('lastScore:', test_score, 'lastRMSE', test_rmse)
        RMSE_t.append(test_rmse)
        score_t.append(test_score)

        # tt_predict = model.predict(test1X_all)
        '''sio.savemat('./models/FD003_testX_new.mat', {'testX_new': testX_new})
        sio.savemat('./data_train_test/FD001_tt.mat', {"tt_predict": yPreds})
        sio.savemat('./models/FD001_testY_all_se.mat', {'testY_all': testY_all})

        sio.savemat('./models/FD001_yPreds_se.mat', {'yPreds': yPreds})
        sio.savemat('./models/test1_len.mat', {'test_len': test_len})
        json_string = model.to_json()
        open('FD004_weights.json', 'w').write(json_string)
        model.save_weights('FD004_weights.h5', overwrite=True)
        sio.savemat('FD001_testX_new.mat', {'testX_new': testX_new})
        #layer_name = 'flat'
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('att1').output)
        tt3_flat = intermediate_layer_model.predict(testX_new)
        sio.savemat('tt1_att1.mat', {"data": tt3_flat})

        intermediate_layer_model0 = Model(inputs=model.input, outputs=model.get_layer('att2').output)
        tt3_flat0 = intermediate_layer_model0.predict(testX_new)
        sio.savemat('tt1_att2.mat', {"data": tt3_flat0})

        intermediate_layer_model1 = Model(inputs=model.input, outputs=model.get_layer('tcn_out').output)
        tt3_tcn = intermediate_layer_model1.predict(testX_new)
        sio.savemat('tt1_tcn.mat', {"data": tt3_tcn})'''
        # sio.savemat('tt1_data',{'tt1_data':testX})'''

    rmse_mean = np.mean(RMSE_t)
    rmse_std = np.std(RMSE_t)
    score_mean = np.mean(score_t)
    score_std = np.std(score_t)
    ttime_mean = np.mean(ttime)
    print('rmse_mean:', rmse_mean, 'rmse_std:', rmse_std, 'score_mean:', score_mean, 'score_std:', score_std)
    print('ttime_mean:', ttime_mean)


