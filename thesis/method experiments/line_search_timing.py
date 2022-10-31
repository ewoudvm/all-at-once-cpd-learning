#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:16:23 2022

@author: Ewoud
"""

import time
import sys
import pandas as pd
import numpy as np
from jax import random
import matplotlib.pyplot as plt
from matplotlib import style
# Set style to colorblind friendly style
style.use('seaborn-colorblind')

from jax import random, jit
import jax.numpy as jnp

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)

import feature_maps
import cpd_functions
import cpd_training
import cpd_weight_update

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    

# =============================================================================
# Parameters and data
# =============================================================================
num_exp = 11

l = 0.1
bound = 1

lambda_reg = 0.00001
learning_rate = 0.1
optimizer_state = None
iteration = 1

seed = 1
rng = np.random.default_rng(seed)
key = random.PRNGKey(seed)
keys = random.split(key, num_exp+1)

D = 10
N = 10000

X = rng.random((N, D))
y = rng.random(N)



steepest_gradient_descent_update = cpd_weight_update.Steepest_Gradient_Descent()
# update_factor = cpd_als.update_factor
# steepest_gradient_descent_update = jit(cpd_weight_update.Steepest_Gradient_Descent())
line_search_update = cpd_weight_update.Line_Search_Gradient_Descent()


# =============================================================================
# Time GD and LS
# =============================================================================

CP_rank = 10
M = 20

mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
Zs = mapping.batch_feature_map(X)


results_gd = np.zeros(num_exp)
results_ls = np.zeros(results_gd.shape)


for i in range(num_exp):
    
    weights = cpd_functions.random_normal_cpd(D, M, CP_rank, keys[i])
    weights_ls = jnp.copy(weights)

    tic = time.perf_counter()
    steepest_gradient_descent_update(weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration)
    toc = time.perf_counter()
    results_gd[i] = toc - tic
    
    d = 1
    tic = time.perf_counter()
    line_search_update(weights_ls, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration)
    toc = time.perf_counter()
    results_ls[i] = (toc - tic)


mean_results_gd = np.mean(results_gd[1:])
std_results_gd = np.std(results_gd[1:])
mean_results_ls = np.mean(results_ls[1:])
std_results_ls = np.std(results_ls[1:])               

fig, ax = plt.subplots()
bp = ax.boxplot([results_gd[1:],results_ls[1:]], labels = ['Steepest Gradient Descent', 'Line Search'])
ax.yaxis.grid(True)
ax.set_ylabel('Evaluation time')
ax.set_xlabel('Method')
ax.set_yscale('log')
