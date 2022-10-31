#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:46:18 2022

@author: Ewoud
"""

import pandas as pd
import numpy as np
import sys
from scipy.io import loadmat
import time

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from jax import random
import jax.numpy as jnp

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)
    
    
CPD_TRAINING_PATH_SERVER = '/home/eavanmourik/thesis_tensor_networks/CPD JAX'
if not CPD_TRAINING_PATH_SERVER in sys.path:
    sys.path.append(CPD_TRAINING_PATH_SERVER)
    
import cpd_functions
import cpd_training
import cpd_als
import cpd_linesearch
import feature_maps
from learning_parameters import LearningParameters
import callbacks
import experiment_helper

# =============================================================================
# Import data set and preprocessing
# =============================================================================
#### Data preperation

data_file_name = '../data set experiments/data sets/airfoil_self_noise.dat'
data_frame = pd.read_table(data_file_name)

X = data_frame.iloc[:, :-1].values
y = data_frame.iloc[:, -1].values

# Preprocess data
# Ensure that data is withtin [0,1]
min_X = np.min(X, 0)
max_X = np.max(X, 0)
X_p = (X-min_X)/(max_X-min_X)

mean_y = np.mean(y)
std_y = np.std(y)
y_p = (y-mean_y)/std_y

# =============================================================================
# Training parameters
# =============================================================================
#### Parameters
# Model
CP_rank = 5
D = X_p.shape[1]

# Mapping
M = 20
bound = 1
l = 0.1
mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
# mapping = feature_maps.BatchFourierFeatureMap

# Learning
learning_rate = 1
learning_rate_adam = 0.05
lambda_reg = 0.00001
epochs = 100
batch_size = X_p.shape[0]
val_split = 0.1
param = LearningParameters(lambda_reg, learning_rate, epochs, batch_size, val_split)

seed = 13
key = random.PRNGKey(seed)


for _ in range(5):
    tic = time.perf_counter()
    result_ls = cpd_training.batch_training(X_p, y_p, CP_rank, mapping, param, method = 'Linesearch Gradient Descent', key=key)
    toc = time.perf_counter()
    print(f'Normal: {toc-tic}')
    
    tic = time.perf_counter()
    result_ls = cpd_training.batch_training(X_p, y_p, CP_rank, mapping, param, method = 'Linesearch Gradient Descent JAX', key=key)
    toc = time.perf_counter()
    print(f'JAX: {toc-tic}')
    
