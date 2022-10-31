#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:37:16 2022

@author: Ewoud
"""

import sys
import numpy as np
from jax import random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)

import cpd_weight_update
import cpd_functions
import feature_maps

plt.rcParams['font.size'] = '14'

seed = 77
key = random.PRNGKey(seed)
rng = np.random.default_rng(seed)

N = 100
Ds = [2, 4, 8, 10, 15, 20]
D = Ds[0]
Ms = [10, 20, 40]
M = 20
CP_rank = 10
X = rng.random((N, D))
y = rng.random(N)
lambda_reg = 1 # 0.00000000001

bound = 1
l = 0.1
mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
Zs = mapping.batch_feature_map(X)

weights = cpd_functions.random_normal_cpd(D, M, CP_rank, key)

gradient, _, _ = cpd_weight_update.batch_gradient(weights, Zs, y, lambda_reg)

gradient_norm = np.linalg.norm(gradient, axis=(1,2))
print(gradient_norm)

num_exp = 10
gradient_res_norms = np.zeros((len(Ds), num_exp))
gradient_reg_norms = np.zeros(gradient_res_norms.shape)


def regularization_gradient(weights, lambda_reg):
    # Regularization term
    # Compute full gamma
    weights_T = jnp.swapaxes(weights, 1, 2)
    gamma = jnp.matmul(weights_T, weights)
    gamma_full = jnp.prod(gamma, axis=0)
    
    # Compute gradient of regularization term
    gradient = 2*lambda_reg*jnp.matmul(weights, gamma_full / gamma)
    
    return gradient


fig_res, ax_res = plt.subplots(dpi=200)
fig_reg, ax_reg = plt.subplots(dpi=200)

for m, M in enumerate(Ms):
    
    mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
    
    for d, D in enumerate(Ds):
        for i in range(num_exp):
            
            X = rng.random((N, D))
            y = rng.random(N)
            
            Zs = mapping.batch_feature_map(X)
            
            key, subkey = random.split(key)
            weights = cpd_functions.random_uniform_cpd(D, M, CP_rank, subkey, normalize=True)
            # weights = cpd_functions.random_uniform_cpd(D, M, CP_rank, key, normalize=False)
    
            gradient_res, _ = cpd_weight_update.mse_gradient(weights, Zs, y, N)
            gradient_res_norm = np.linalg.norm(gradient_res, axis=(1,2))
            gradient_res_norms[d, i] = np.mean(gradient_res_norm)
            
            gradient_reg = regularization_gradient(weights, lambda_reg)
            gradient_reg_norm = np.linalg.norm(gradient_reg, axis=(1,2))
            gradient_reg_norms[d, i] = np.mean(gradient_reg_norm)
    

    mean_gradient_res_norms = np.mean(gradient_res_norms, axis=1)
    std_gradient_res_norms = np.std(gradient_res_norms, axis=1)
    ax_res.errorbar(Ds, mean_gradient_res_norms, std_gradient_res_norms, label=f'M = {M}', fmt='o', capsize=5)
    
    mean_gradient_reg_norms = np.mean(gradient_reg_norms, axis=1)
    std_gradient_reg_norms = np.std(gradient_reg_norms, axis=1)
    ax_reg.errorbar(Ds, mean_gradient_reg_norms, std_gradient_reg_norms, label=f'M = {M}', fmt='o', capsize=5)
                   

# Pretty figures
ax_res.yaxis.grid(True, linestyle='-', which = 'both', alpha=0.7)
ax_res.set(axisbelow=True)
ax_res.set_yscale('log')
ax_res.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_res.set_xlabel('Number of features D')
ax_res.set_ylabel('Mean norm')
ax_res.legend()
ax_res.set_title('Gradient of MSE term')

fig_res.tight_layout()

ax_reg.yaxis.grid(True, linestyle='-', which = 'both', alpha=0.7)
ax_reg.set(axisbelow=True)
ax_reg.set_yscale('log')
ax_reg.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_reg.set_xlabel('Number of features D')
ax_reg.set_ylabel('Mean norm')
ax_reg.legend()
ax_reg.set_title('Gradient of regularization term')

fig_reg.tight_layout()





