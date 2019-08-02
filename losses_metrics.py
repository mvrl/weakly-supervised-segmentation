# This file contains the loss functions and metrics

# Author M. Usman Rafique

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

from model import get_model



# Maximum number of ground truth instances to use in one image
MAX_GT_INSTANCES = 100

my_th = 0.5  # thresholding to binarize

## Metrics

# My IoU metric
def my_IoU(y_true, y_pred):
    y_pred_new = K.concatenate([y_pred[:,:,:,0]])
    #y_pred_new = K.reshape(y_pred_new, (300*300,1))
    y_pred_new = K.batch_flatten(y_pred_new)

    y_true_new = K.concatenate([y_true[:,:,:,0]])
    #y_true_new = K.reshape(y_true_new, (300*300,1))
    y_true_new = K.batch_flatten(y_true_new)

    ## NEW: using  thresholding
    y_pred_bin = K.greater(y_pred_new, my_th)
    y_pred_bin2 = K.cast(y_pred_bin, dtype='float32')

    a_int_b = K.sum(y_true_new * y_pred_bin2)
    a_u_b = K.sum(y_true_new) + K.sum(y_pred_bin2) - a_int_b

    # Uncomment to remove thresholding
    #a_int_b = K.sum(y_true * y_pred)
    #a_u_b = K.sum(y_true) + K.sum(y_pred) - a_int_b

    return (a_int_b/(a_u_b+ K.epsilon()))
# Accuracy
def my_acc(y_true, y_pred):
    y_pred_new = K.concatenate([y_pred[:,:,:,0]])
    #y_pred_new = K.reshape(y_pred_new, (300*300,1))
    y_pred_new = K.batch_flatten(y_pred_new)#(y_pred)#(y_pred_new)

    y_true_new = K.concatenate([y_true[:,:,:,0]])
    #y_true_new = K.reshape(y_true_new, (300*300,1))
    y_true_new = K.batch_flatten(y_true_new)#(y_true)#(y_true_new)

    ## NEW: using  thresholding
    y_pred_bin = K.greater(y_pred_new, my_th)
    y_pred_bin2 = K.cast(y_pred_bin, dtype='float32')

    # true positive
    y_pred_pos = K.sum(y_true_new * y_pred_bin2  )  #y_pred_new

    # No. of positives in GT
    y_pos_gt = K.sum(y_true)

    # False negatives
    my_pred_inv = 1.0 -  y_pred_bin2 #y_pred_new
    false_neg = K.sum(my_pred_inv * y_true_new)


    #return ( y_pred_pos / y_pos_gt )
    return (y_pred_pos / (y_pred_pos + false_neg + K.epsilon() ) )


## Proposed one-sided loss
# Hyper-parameters
eta = 0.3 #1  # penalyt for area
scale_fp = 0.9 #1  # false positive
gamma = 0.7 #1  # False negative
area_scale = 0.9

def disagg_loss(y_true, y_pred):

    # shape of input tensor
    sp = K.shape(y_true)

    # False positive
    true_inv = 1 - y_true   # invert GT masks
    my_o = K.zeros((sp[0], sp[1], sp[2], sp[3]))
    my_fp = y_pred * true_inv

    Lfp = 1 * scale_fp * keras.losses.mean_squared_error(my_o, my_fp)
    ##Lfp = 1 * scale_fp * keras.losses.binary_crossentropy(my_o, my_fp)  # did not do good


    # False negative
    pred_inv = 1 - y_pred  # inverse of prediction
    my_fn = pred_inv * y_true
    my_z = K.zeros((sp[0], sp[1], sp[2], sp[3]))

    Lfn = 1.0 * gamma * keras.losses.mean_squared_error(my_fn, my_z)

    # Area loss
    a_gt = K.sum(K.sum(K.batch_flatten(y_true)))    # GT Area
    a_pred = K.sum(K.sum(K.batch_flatten(y_pred)))  # predicted area

    # One-sided loss
    a_mag = K.maximum((a_pred - area_scale * a_gt ) / (a_gt + 1e-7), 0.0)  # loss is only if predicted area is bigger

    y_penalty = K.reshape(1 - y_pred, (sp[0], sp[1], sp[2]))
    y_penalty = K.maximum(y_penalty, 0.0)

    La = 1.0 * eta * a_mag * y_penalty


    return (Lfn +Lfp + La)  # Total loss


def empty_loss(y_true, y_pred):     # always returns ZERO
    return 0.0*keras.losses.MSE(y_true, y_pred)

keras.losses.disagg_loss = disagg_loss
keras.losses.empty_loss = empty_loss

keras.metrics.my_IoU = my_IoU
keras.metrics.my_acc = my_acc

