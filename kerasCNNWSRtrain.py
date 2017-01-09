# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 17:25:33 2017

@author: Neeraj
"""

#from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import os 
os.chdir('/Users/apple/Documents/CodesResearch/PRL_CNNWSR')

#from keras.datasets import cifar10
#from keras.preprocessing.image import ImageGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import scipy.io as sio
import keras
batch_size = 500
#nb_classes = 10
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 33, 33
# number of color channels in put
img_channels = 1
# number of training images
ntr = 40000

# load data
print('Loading Data...')
Data = sio.loadmat('./training.mat')

trainX = Data['trainX'][:ntr,:]
trainY = Data['trainY'][:ntr,:]
trainX = trainX.reshape((ntr,img_channels,img_rows, img_cols))
testX = Data['trainX'][ntr:,:]
testX = testX.reshape((testX.shape[0],img_channels,img_rows, img_cols))
testY = Data['trainY'][ntr:,:]

Data = None
del Data
print(trainX.shape[0], 'train samples')
print(testX.shape[0], 'test samples')
print('Creating model...')
# create modell
model = Sequential()
model.add(Convolution2D(64,7,7,border_mode = 'same',input_shape=trainX.shape[1:]))
model.add(Activation("relu"))
#model.add(Dropout(0.1))

model.add(Convolution2D(32,5,5,border_mode = 'same',input_shape=trainX.shape[1:]))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Convolution2D(32,3,3,border_mode = 'same',input_shape=trainX.shape[1:]))
model.add(Activation('linear'))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(trainY.shape[1],activation = "linear"))
#model.add(Dropout(0.5))

#adm = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='rmsprop')


trainX = trainX.astype('float32')
testX= testX.astype('float32')
trainX /=255.0
testX /= 255.0

print('Training started...')
model.fit(trainX,trainY,
          batch_size = batch_size,
          nb_epoch = nb_epoch,
          validation_data = (testX,testY),
          shuffle = 'True')











