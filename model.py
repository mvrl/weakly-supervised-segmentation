# This file contains the U-Net model

# Author M. Usman Rafique

# U-NET code adopted from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

## Imports

import numpy as np
import utils
import random
import os
import keras
import keras.backend as K
#from keras.layers import Conv2D, Dropout, MaxPooling2D, LeakyReLU, Input, Dense, Lambda, Flatten, Concatenate, MaxPooling1D, MaxPool1D, AveragePooling2D, BatchNormalization
#from keras.layers import Maximum, Average, Activation, ZeroPadding2D, Add, UpSampling2D, merge, concatenate, Conv2DTranspose
from keras.layers import *

from keras import optimizers
from keras.models import Model
from numpy import newaxis
import keras.losses
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow as tf
from dataset import MappingChallengeDataset
import logging
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from config import Config
from keras.models import load_model
from keras.losses import binary_crossentropy
from keras import layers
from keras.backend import tf as ktf
from math import ceil
#from AdamW import AdamW








#####################################
## MODEL
#####################################
def get_model():
    ################## U NET, from ## from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    main_input = Input(shape=(304, 304, 3)) # padding with zeros to make it divisible by 16

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(main_input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0))(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer= l2(1e-5))(up9)  # previously used kernel_regularizer=l2(0.001)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid' , use_bias=False, name='output_original')(conv9)  # original, 1 feature map, activation = 'sigmoid'

    # We need a fake (place holder output) so that Keras allows us to pass image IDs. Image IDs are used to load true
    # segmentation tasks for the purposes of visualization. Making an auxiliary output

    flat_input = Flatten()(main_input)
    flat_input = Dense(1, trainable=False)(flat_input)  # Making it non-trainable

    aux_image_id = Dense(1, trainable=False, name='aux_id')(flat_input) # Also non-trainable

    # Another place holder for true segmentation masks. Only used for visualizations.
    placeholder_conv = Activation('linear', trainable=False, name='placeholder_conv')(conv10)


    # IMAGE ID might do the trick

    # Prepare a model object
    ############################## BIG CHANGE: added an output
    #model = Model(inputs=[main_input], outputs = [mask_output_flat, mask_output_flat, count_out_final, aux_image_id] )
    model = Model(inputs=[main_input], outputs = [conv10, placeholder_conv, aux_image_id] )

    return model