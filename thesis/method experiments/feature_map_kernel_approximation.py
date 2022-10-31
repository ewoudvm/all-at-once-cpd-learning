#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:21:38 2022

@author: Ewoud
"""
import sys
import numpy as np
from matplotlib import cm
from numpy import linalg as la
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import style
# Set style to colorblind friendly style
style.use('seaborn-colorblind')

from jax import random

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)

import feature_maps
import cpd_functions
import cpd_training

def feature_map(x_d, M, U_d, l):
    
    m = np.arange(1,M+1)
    
    omega = np.pi*m/U_d/2
    S = np.sqrt(2*np.pi)*l*np.exp(-omega**2*l**2/2)
    Z_x = 1/np.sqrt(U_d)*np.sqrt(S)*np.sin((np.pi*m*(x_d+U_d))/(2*U_d))
    
    return Z_x


M = 20
l = 0.1
bound = 1
xs = np.arange(0.1, 1, 1)
fig, ax = plt.subplots()
for i in range(len(xs)):
    
    ax.scatter(np.arange(1, M+1), feature_map(xs[i], M, bound, l))

ax.set_xlabel('m')
ax.set_ylabel(r'$z(x)_m$')
xticks = np.arange(1, 21)
ax.set_xticks(xticks)
ax.set_title('Values of feature map vector')


ls = np.arange(0.05, 1, 0.001)
sqrtS = np.sqrt(np.sqrt(2*np.pi)*ls*np.exp(-ls*ls*np.pi*np.pi/8))
fig, ax = plt.subplots()
ax.plot(ls, sqrtS)
ax.set_xlabel('l')
ax.set_ylabel(r'$\sqrt{S}$')
ax.set_title('Feature map amplitude as a function of the lenght-scale')

print(f'Max sqrt(S): {np.max(sqrtS)}, {ls[np.argmax(sqrtS)]}')


# =============================================================================
# Kernel approximation    
# =============================================================================

def rank_one_inner_prod(X, Y):

    
    prod = 1
    for i in range(X.shape[0]):
        prod = prod*np.dot(X[i, :], Y[i, :])
        
    return prod


l = 0.1
order = 12
bound = 1

x_0 = np.array([0,0])
x = np.array([np.arange(0, bound, bound/50), np.zeros(50)]).T

# Kernel value
k_rbf = np.exp(la.norm(x_0-x, axis=1)**2/-2.0/l/l)

Ms = [4, 8, 12, 16]

fig, ax = plt.subplots()

for M in Ms:

    mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
    
    #Kernel approximation
    Z_x_0 = np.array([feature_map(x_0[0], M, bound, l), feature_map(x_0[1], M, bound, l)])
    
    Z_x = mapping.batch_feature_map(x)
    
    k_approx = np.zeros(x.shape[0])
    for n in range(x.shape[0]):
    
        k_approx[n] = rank_one_inner_prod(Z_x_0, Z_x[:, :, n])

    ax.plot(x[:,0], k_approx, label=f'M = {M}')


ax.scatter(x[:,0], k_rbf, marker = 'o', label='kernel')
ax.set_xlabel('x')
ax.set_ylabel('Kernel value')
ax.set_title('Kernel approximation for different order feature maps')
ax.legend()

# =============================================================================
# Gaussian kernel approximation plot
# =============================================================================


# X_pos, Y_pos = np.mgrid[-1:1:51j, -1:1:51j]
# X = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
# D = 2

# M = 12
# M_low = 2 # int(four_order/2)
# R = 1
# bound = 1
# l = 0.1

# key = random.PRNGKey(1)

# # Create random weights
# weights =  cpd_functions.random_uniform_cpd(D, M, R, key= key)
# weights_low = cpd_functions.random_uniform_cpd(D, M_low, R, key= key)

# # Create rank-one CPD that is a all ones
# weights_eye = np.ones((D, M, R))
# weights_eye_low = np.ones((D, M_low, R))

# mapping = feature_maps.BatchFourierFeatureMap(M, bound, l)
# mapping_low = feature_maps.BatchFourierFeatureMap(M_low, bound, l)

# Zs = mapping.batch_feature_map(X)
# Zs_low = mapping_low.batch_feature_map(X)

# y_pred = cpd_training.prediction(weights, Zs)
# y_pred_low = cpd_training.prediction(weights_low, Zs_low)
# y_pred_eye = cpd_training.prediction(weights_eye, Zs)
# y_pred_eye_low = cpd_training.prediction(weights_eye_low, Zs_low)


# # Plotting 3d feature map with random weights
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(projection='3d')
# ax2.scatter(X[:, 0], X[:,1], y_pred, c=y_pred, cmap=cm.coolwarm, marker='^')
# ax2.set_xlabel('Feature 1')
# ax2.set_ylabel('Feature 2')
# ax2.set_zlabel('Output')


# #Plotting feature map times CPD of all ones
# fig3 = plt.figure()
# ax3 = fig3.add_subplot()

# x_1 = X[X[:,1]==0.0, 0]
# y_rbf = norm.pdf(x_1, 0, 0.3)
# y_rbf = y_rbf/np.max(y_rbf)*np.max(y_pred_eye[X[:,1]==0.0])


# ax3.scatter(X[X[:,1]==X[:,0], 0],  y_pred_eye[X[:,1]==X[:,0]], c='b',  marker='^', label='High order')
# ax3.scatter(X[X[:,1]==X[:,0], 0],  y_pred_eye_low[X[:,1]==X[:,0]], c='r',  marker='*', label='Low_order')
# ax3.scatter(X[X[:,1]==0.0, 0],  y_rbf, c='k',  marker='o', label='Gaussian')

# ax3.set_xlabel('Feature 1')
# ax3.set_ylabel('Y')
# ax3.legend()

# plt.show()