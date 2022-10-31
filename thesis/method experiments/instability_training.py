#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:01:47 2022

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
learning_rate_adam = 0.1
lambda_reg = 0.00001
epochs = 1000
batch_size = X_p.shape[0]
mini_batch_size = 100
val_split = 0.1
param = LearningParameters(lambda_reg, learning_rate, epochs, batch_size, val_split)
param_adam = LearningParameters(lambda_reg, learning_rate_adam, epochs, batch_size, val_split)

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
initial_weights_Adam = jnp.copy(initial_weights)


# =============================================================================
# Call back
# =============================================================================

cb = callbacks.CallBackGradient(D, M, CP_rank, max_iter=epochs)
cb_adam = callbacks.CallBackGradient(D, M, CP_rank, max_iter=epochs)
cb_norm = callbacks.CallBackCPDNorm(D, M, CP_rank, max_iter= epochs)
# =============================================================================
# Training model
# =============================================================================
#### Training model

result = cpd_training.batch_training(X_p, y_p, CP_rank, mapping, param, method = "Steepest Gradient Descent", key=key, initial_weights=initial_weights, callback=cb)
result_Adam = cpd_training.batch_training(X_p, y_p, CP_rank, mapping, param_adam, method = "Adam Gradient Descent", key=key, initial_weights=initial_weights_Adam, callback=cb_adam)
results = [result, result_Adam]

result_norm = cpd_training.batch_training(X_p, y_p, CP_rank, mapping, param, method = "Steepest Gradient Descent", key=key, initial_weights=initial_weights, callback=cb_norm)


# =============================================================================
# Plot results
# =============================================================================

# Plotting loss functions
colors =[plot_helper.SGD_COLOR, plot_helper. ADAM_COLOR]
line_styles=['-', '-']

plot_helper.plot_training_and_validation_loss(results, colors, line_styles, log_y_scale=True)
plot_helper.plot_training_and_validation_loss_and_mse(results)
# =============================================================================
# Investigate instability
# =============================================================================

fig, axes = plt.subplots(nrows=3, sharex = True, dpi=200, figsize = (6.4, 8))
start = 475
indices = np.arange(start,600)
results = [result]
ax = axes[0]
for result in results:

    # Plot instability
    ax.plot(indices,result.loss_train[indices], label=plot_helper.get_label(result.method), c = plot_helper.get_color(result.method))

ax.set_title('Training loss')
ax.yaxis.grid(True, alpha=0.5, which = 'both')
ax.xaxis.grid(True, alpha=0.5)
ax.set_yscale('log')
ax.set_ylabel('Loss')

# Plot angle between consecutive gradients

def angle_between_vectors(v1, v2):
    
    # make unit vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    
    return np.arccos(np.dot(v1_u, v2_u))

angles_gd_abs = np.zeros(indices.shape)
angles_gd_rel = np.zeros(indices.shape)
angles_adam = np.zeros(indices.shape)

for index, i in enumerate(indices):
    
    angles_gd_abs[index] = angle_between_vectors(cb.gradients[:, :, :, start].flatten(), cb.gradients[:, :, :, i].flatten())
    angles_gd_rel[index] = angle_between_vectors(cb.gradients[:, :, :, i].flatten(), cb.gradients[:, :, :, i+1].flatten())
    angles_adam[index] = angle_between_vectors(cb_adam.gradients[:, :, :, i].flatten(), cb_adam.gradients[:, :, :, i+1].flatten())
    
  
axes[1].plot(indices, angles_gd_abs, label='Absolute', c = plot_helper.get_color('Steepest Gradient Descent'))    



    

# axes[1].plot(indices, angles_adam, label='Adam angles', c = plot_helper.get_color('Adam Gradient Descent'))
axes[1].set_ylabel('Angle (rad)')
axes[1].set_title('Angles between gradients')
axes[1].yaxis.grid(True, alpha=0.5)
axes[1].xaxis.grid(True, alpha=0.5)
# axes[1].legend()

axes[2].plot(indices, cb_norm.norms[indices], c = plot_helper.get_color('Steepest Gradient Descent'))
axes[2].set_ylabel('Norm')
axes[2].set_title('CPD norm')
axes[2].yaxis.grid(True, alpha=0.5)
axes[2].xaxis.grid(True, alpha=0.5)
axes[2].set_xlabel('Iteration')

fig.tight_layout()

