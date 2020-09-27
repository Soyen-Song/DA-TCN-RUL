# -*- coding: utf-8 -*-
################ data processing ##########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from scipy import interpolate
import scipy.io as sio
from numpy import *
from keras.layers import Activation, multiply
from keras.models import *
from keras.layers.core import *
import pywt


min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler = preprocessing.StandardScaler()
RUL_01 = np.loadtxt('./cmapss/RUL_FD002.txt')
train_01_raw = np.loadtxt('./cmapss/train_FD002.txt')
test_01_raw = np.loadtxt('./cmapss/test_FD002.txt')

train_01_raw[:, 2:] = min_max_scaler.fit_transform(train_01_raw[:, 2:])
test_01_raw[:, 2:] = min_max_scaler.transform(test_01_raw[:, 2:])

train_01_nor = train_01_raw
test_01_nor = test_01_raw

# train_01_nor = np.delete(train_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1)  # select sensor
# test_01_nor = np.delete(test_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1)  #
train_01_nor = np.delete(train_01_nor, [2,3,4,5, 9, 10, 14, 20, 22, 23], axis=1)  # select sensor
test_01_nor = np.delete(test_01_nor, [2,3,4,5, 9, 10, 14, 20, 22, 23], axis=1)  #

max_RUL = 125.0  # max RUL for training
winSize = 50 # FD002: 40; FD002: 50; FD003: 40; FD004:50
Feasize = 14
trainX = []
trainY = []
trainY_bu = []
testX = []
testY = []
testY_bu = []
testInd = []
testLen = []

for i in range(1, int(np.max(train_01_nor[:, 0])) + 1):
    ind = np.where(train_01_nor[:, 0] == i)
    ind = ind[0]
    data_temp = train_01_nor[ind, :]
    for j in range(len(data_temp) - winSize + 1):
        trainX.append(data_temp[j:j + winSize, 2:].tolist())
        train_RUL = len(data_temp) - winSize - j
        train_bu = max_RUL - train_RUL
    #    train_RUL_norm = train_RUL/float(len(data_temp))
        if train_RUL > max_RUL:
            train_RUL = max_RUL
            train_bu = 0.0
        trainY.append(train_RUL)
        trainY_bu.append(train_bu)

for i in range(1, int(np.max(test_01_nor[:, 0])) + 1): # i - the ith test dataset
    ind = np.where(test_01_nor[:, 0] == i) # 第i个test dataset所有的行数
    ind = ind[0]
    testLen.append(float(len(ind))) # 数据长度
    data_temp = test_01_nor[ind, :] # 第i个test dataset数据放到data_temp
    testY_bu.append(data_temp[-1, 1])
    if len(data_temp) < winSize:
        data_temp_a = []
        for myi in range(data_temp.shape[1]):
            x1 = np.linspace(0, winSize - 1, len(data_temp))
            x_new = np.linspace(0, winSize - 1, winSize)
            tck = interpolate.splrep(x1, data_temp[:, myi])
            a = interpolate.splev(x_new, tck)
            data_temp_a.append(a.tolist())
        data_temp_a = np.array(data_temp_a)
        data_temp = data_temp_a.T
        data_temp = data_temp[:, 2:]
    else:
        data_temp = data_temp[-winSize:, 2:]

    data_temp = np.reshape(data_temp, (1, data_temp.shape[0], data_temp.shape[1]))
    if i == 1:
        testX = data_temp
    else:
        testX = np.concatenate((testX, data_temp), axis=0)
    if RUL_01[i - 1] > max_RUL:
        testY.append(max_RUL)
        #testY_bu.append(0.0)
    else:
        testY.append(RUL_01[i - 1])

    #testY.append(RUL_01[i - 1])

trainX = np.array(trainX)
testX = np.array(testX)

trainY = np.array(trainY)/max_RUL # normalize to 0-1 for training
trainY_bu = np.array(trainY_bu)/max_RUL
testY = np.array(testY)/max_RUL
testY_bu = np.array(testY_bu)/max_RUL

testX_all = []
testY_all = []
test_len = []
for i in range(1, int(np.max(test_01_nor[:, 0])) + 1):
    ind = np.where(test_01_nor[:, 0] == i)
    ind = ind[0]
    data_temp = test_01_nor[ind, :]
    data_RUL = RUL_01[i - 1]
    test_len.append(len(data_temp) - winSize + 1)
    for j in range(len(data_temp) - winSize + 1):
        testX_all.append(data_temp[j:j + winSize, 2:].tolist())
        test_RUL = len(data_temp) + data_RUL - winSize - j
        if test_RUL > max_RUL:
            test_RUL = max_RUL
        testY_all.append(test_RUL)

testX_all = np.array(testX_all)
testY_all = np.array(testY_all)
test_len = np.array(test_len)

trainX_hw = []
for i in range(len(trainX)):
    tmp = []
    for j in range(trainX.shape[2]):
        x = trainX[i, :, j]
        (ca, cd) = pywt.dwt(x,'haar')
        cat = pywt.threshold(ca, np.std(ca), mode='soft')
        cdt = pywt.threshold(cd, np.std(cd), mode='soft')
        tx = pywt.idwt(cat, cdt, 'haar')
        tmp.append(tx)
    tmp = np.array(tmp)
    tmp = tmp.T
    trainX_hw.append(tmp)
trainX_hw = np.array(trainX_hw)

testX_hw = []
for i in range(len(testX)):
    tmp = []
    for j in range(testX.shape[2]):
        x = testX[i, :, j]
        (ca, cd) = pywt.dwt(x,'haar') # ca 低频率, 粗信号; cd 高频, 细粒度信号
        cat = pywt.threshold(ca, np.std(ca), mode='soft') # 此处软阈值函数内容: soft=max(0,ca-np.std(ca))
        cdt = pywt.threshold(cd, np.std(cd), mode='soft')
        tx = pywt.idwt(cat, cdt, 'haar')
        tmp.append(tx)
    tmp = np.array(tmp)
    tmp = tmp.T
    testX_hw.append(tmp)
testX_hw = np.array(testX_hw)

sio.savemat('train2X.mat', {"train2X": trainX})
sio.savemat('train2Y.mat', {"train2Y": trainY})
# sio.savemat('train3Y_bu.mat', {"train3Y_bu": trainY_bu})
sio.savemat('test2X.mat', {"test2X": testX})
sio.savemat('test2Y.mat', {"test2Y": testY})
'''





