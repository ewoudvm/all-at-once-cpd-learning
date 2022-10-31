#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:51:21 2022

@author: Ewoud
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)
    
import cpd_functions
import cpd_training
import cpd_als
import cpd_linesearch
import feature_maps
from learning_parameters import LearningParameters
import callbacks
import plot_helper

plt.rcParams['font.size'] = 14
# =============================================================================
# Import data set and preprocessing
# =============================================================================
#### Data preperation

data_file = '../data set experiments/data sets/airfoil_self_noise.dat'
data_frame = pd.read_table(data_file)

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
M = 12
bound = 1
l = 0.1
mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
# mapping = feature_maps.BatchFourierFeatureMap

# Learning
learning_rate = 1
lambda_reg = 0.00001
epochs = 100
batch_size = X_p.shape[0]
mini_batch_size = 100
val_split = 0.1
param = LearningParameters(lambda_reg, learning_rate, epochs, batch_size, val_split)
param_mini_batch = LearningParameters(lambda_reg, learning_rate, epochs, mini_batch_size, val_split)

# ALS
num_sweeps = 20

# Seed and key
seed = 13
key = random.PRNGKey(seed)
# =============================================================================
# Initial weights
# =============================================================================
#### Initial weights

initial_weights = cpd_functions.random_uniform_cpd(D, M, CP_rank, key)
initial_weights_mini_batch = jnp.copy(initial_weights)
initial_weights_ALS = jnp.copy(initial_weights)

# =============================================================================
# Training model 
# =============================================================================
#### Training model

result_mini_batch = cpd_training.batch_training(X_p, y_p, CP_rank, mapping, param_mini_batch, method = "Steepest Gradient Descent", key=key, initial_weights=initial_weights_mini_batch, save_detailed_training_losses=True)
result_mini_batch.optimization_details['batch size'] = mini_batch_size
result_ALS = cpd_als.als_training(X_p, y_p, CP_rank, mapping, num_sweeps, val_split, lambda_reg, key, initial_weights=initial_weights_ALS)

# =============================================================================
# Plot results
# =============================================================================

# Normal iteration definition
fig, ax = plt.subplots(dpi=200)

ax.plot(result_mini_batch.loss_train, label=plot_helper.get_label(result_mini_batch.method), c = plot_helper.get_color(result_mini_batch.method), linewidth=2.5)
ax.plot(result_ALS.loss_train, label = plot_helper.get_label(result_ALS.method), c = plot_helper.get_color(result_ALS.method), linewidth=2.5)
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training loss')
ax.set_yscale('log')
ax.yaxis.grid(True, alpha=0.7, which = 'both')

# One iteration is one update of all factors
fig, ax = plt.subplots(dpi=200)
als_indices = np.arange(0, len(result_ALS.loss_train), D)
ax.plot(result_mini_batch.loss_train, label=plot_helper.get_label(result_mini_batch.method), c = plot_helper.get_color(result_mini_batch.method), linewidth=2.5)
ax.plot(result_ALS.loss_train[als_indices], label = plot_helper.get_label(result_ALS.method), c = plot_helper.get_color(result_ALS.method), linewidth=2.5)
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training loss')
ax.set_yscale('log')
ax.yaxis.grid(True, alpha=0.7, which = 'both')

# One iteration is one update of all factors
fig, ax = plt.subplots(dpi=200)
ax.plot(result_mini_batch.optimization_details['final_loss_train_det'], label=plot_helper.get_label(result_mini_batch.method), c = plot_helper.get_color(result_mini_batch.method),linewidth=2.5)
ax.plot(result_ALS.loss_train, label = plot_helper.get_label(result_ALS.method), c = plot_helper.get_color(result_ALS.method),linewidth=2.5)
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training loss')
ax.set_yscale('log')
ax.yaxis.grid(True, alpha=0.7, which = 'both')
