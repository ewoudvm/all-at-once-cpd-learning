#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:18:02 2022

@author: Ewoud
"""

import time
import sys
import numpy as np
from numpy.polynomial import polynomial as P
from jax import random
import matplotlib.pyplot as plt
from matplotlib import style, ticker
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
import cpd_als
import plot_helper

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
ALS_COLOR = plot_helper.ALS_COLOR
SGD_COLOR = plot_helper.SGD_COLOR

plt.rcParams['font.size'] = '12'

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
N = 1000

X = rng.random((N, D))
y = rng.random(N)

# steepest_gradient_descent_update = cpd_weight_update.Steepest_Gradient_Descent()
# update_factor = cpd_als.update_factor
steepest_gradient_descent_update = jit(cpd_weight_update.Steepest_Gradient_Descent())
update_factor = jit(cpd_als.update_factor)


# =============================================================================
# Loop over R
# =============================================================================

M = 20

mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
Zs = mapping.batch_feature_map(X)

Rs = [5, 10, 15, 20, 25, 30, 35, 40]

results_R_gd = np.zeros((num_exp, len(Rs)))
results_R_als = np.zeros(results_R_gd.shape)

for r, CP_rank in enumerate(Rs):
    
    weights = cpd_functions.random_uniform_cpd(D, M, CP_rank, keys[r+1])
    weights_als = jnp.copy(weights) 
    
    batch_ZW_full = cpd_als.cpd_als_batch_ZW_full(X, weights_als, mapping)
    gamma_full = cpd_functions.cpd_gamma_full(weights_als)
    
    for i in range(num_exp):
    
        tic = time.perf_counter()
        steepest_gradient_descent_update(weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration)
        toc = time.perf_counter()
        results_R_gd[i, r] = toc - tic
        
        d = 1
        tic = time.perf_counter()
        update_factor(weights_als, Zs[d, :, :], y, batch_ZW_full, gamma_full, lambda_reg, d)
        toc = time.perf_counter()
        results_R_als[i, r] = (toc - tic)*D
        # Times D, since all factors are updated once then


mean_results_R_gd = np.mean(results_R_gd[1:, :], axis=0)
std_results_R_gd = np.std(results_R_gd[1:, :], axis=0)
mean_results_R_als = np.mean(results_R_als[1:, :], axis = 0)
std_results_R_als = np.std(results_R_als[1:, :], axis=0)               



fig, ax = plt.subplots()

r_gd_model, r_gd_stats = P.polyfit(np.log(Rs), np.log(mean_results_R_gd), deg=1, full=True)
r_als_model, r_als_stats = P.polyfit(np.log(Rs), np.log(mean_results_R_als), deg=1, full=True)

ax.plot(Rs, np.exp(r_gd_model[1]*np.log(Rs)+r_gd_model[0]), ls=':', c=SGD_COLOR) #, label='Steepest Gradient Descent fit')
ax.errorbar(Rs, mean_results_R_gd, std_results_R_gd, label= plot_helper.get_label('Steepest Gradient Descent'), fmt='o', capsize=5, ecolor=SGD_COLOR, mec=SGD_COLOR, mfc=SGD_COLOR)

ax.plot(Rs, np.exp(r_als_model[1]*np.log(Rs)+r_als_model[0]),ls=':',c=ALS_COLOR) #, label='ALS fit')
ax.errorbar(Rs, mean_results_R_als, std_results_R_als, label=plot_helper.get_label('ALS'), fmt='o', capsize=5, ecolor=ALS_COLOR, mec=ALS_COLOR, mfc=ALS_COLOR)

ax.yaxis.grid(True)
ax.set_ylabel('Run time (s)', fontsize=14)
ax.set_xlabel('CP-rank', fontsize=14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Update of all factors', fontsize=16)

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels)
fig.tight_layout()

print('Parameters for R model')
print(f'ALS model: a = {np.exp(r_als_model[0]):.6f},  b = {r_als_model[1]:.2f}')
print(f'GD model: a = {np.exp(r_gd_model[0]):.6f},  b = {r_gd_model[1]:.2f}')

# =============================================================================
# Loop over M
# =============================================================================

CP_rank = 10



Ms = [10, 20, 30, 40, 50, 60]

results_M_gd = np.zeros((num_exp, len(Ms)))
results_M_als = np.zeros(results_M_gd.shape)

for m, M in enumerate(Ms):
    
    weights = cpd_functions.random_uniform_cpd(D, M, CP_rank, keys[m+1])
    weights_als = jnp.copy(weights) 
    
    mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
    Zs = mapping.batch_feature_map(X)
    
    batch_ZW_full = cpd_als.cpd_als_batch_ZW_full(X, weights_als, mapping)
    gamma_full = cpd_functions.cpd_gamma_full(weights_als)
    
    for i in range(num_exp):
    
        tic = time.perf_counter()
        steepest_gradient_descent_update(weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration)
        toc = time.perf_counter()
        results_M_gd[i, m] = toc - tic
        
        d = 1
        tic = time.perf_counter()
        update_factor(weights_als, Zs[d, :, :], y, batch_ZW_full, gamma_full, lambda_reg, d)
        toc = time.perf_counter()
        results_M_als[i, m] = (toc - tic)*D
        # Times D, since all factors are updated once then


mean_results_M_gd = np.mean(results_M_gd[1:, :], axis=0)
std_results_M_gd = np.std(results_M_gd[1:, :], axis=0)
mean_results_M_als = np.mean(results_M_als[1:, :], axis = 0)
std_results_M_als = np.std(results_M_als[1:, :], axis=0)               


fig, ax = plt.subplots()

m_gd_model, m_gd_stats = P.polyfit(np.log(Ms), np.log(mean_results_M_gd), deg=1, full=True)
m_als_model, m_als_stats = P.polyfit(np.log(Ms), np.log(mean_results_M_als), deg=1, full=True)

ax.plot(Ms, np.exp(m_gd_model[1]*np.log(Ms)+m_gd_model[0]), ls=':', c=SGD_COLOR, linewidth=2) #, label=plot_helper.get_label('Steepest Gradient Descent') +' fit')
ax.errorbar(Ms, mean_results_M_gd, std_results_M_gd, label= plot_helper.get_label('Steepest Gradient Descent'), fmt='o', capsize=5, ecolor=SGD_COLOR, mec=SGD_COLOR, mfc=SGD_COLOR)

ax.plot(Ms, np.exp(m_als_model[1]*np.log(Ms)+m_als_model[0]),ls=':',c=ALS_COLOR, linewidth=2) #, label=plot_helper.get_label('ALS') +' fit')
ax.errorbar(Ms, mean_results_M_als, std_results_M_als, label=plot_helper.get_label('ALS'), fmt='o', capsize=5, ecolor=ALS_COLOR, mec=ALS_COLOR, mfc=ALS_COLOR)

ax.yaxis.grid(True)
ax.set_ylabel('Run time (s)', fontsize=14)
ax.set_xlabel('M', fontsize=14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Update of all factors',fontsize=16)

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels)
fig.tight_layout()

print('Parameters for M model')
print(f'ALS model: a = {np.exp(m_als_model[0]):.6f},  b = {m_als_model[1]:.2f}')
print(f'GD model: a = {np.exp(m_gd_model[0]):.6f},  b = {m_gd_model[1]:.2f}')
