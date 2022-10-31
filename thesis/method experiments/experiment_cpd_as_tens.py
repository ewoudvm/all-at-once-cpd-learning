#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:31:06 2022

@author: Ewoud
"""

import sys
import time
import numpy as np
from jax import random, jit
import jax.numpy as jnp

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)

from cpd_functions import random_uniform_cpd, cpd_norm


D = 20
M = 20
CP_rank= 10

seed = 1
key = random.PRNGKey(seed)

cpd_tens = random_uniform_cpd(D, M, CP_rank, key)
cpd_list = [cpd_tens[d, :, :] for d in range(D)]


def cpd_list_norm(cpd_list):
    
    R = cpd_list[0].shape[1]
    gamma = jnp.ones((R,R))
    for factor in cpd_list:
        gamma = gamma * jnp.matmul(factor.T, factor)
        
    norm = jnp.sum(gamma)
    return norm

num_runs = 1000
num_exp = 10
times_tens = np.zeros(num_exp)
times_list = np.zeros(num_exp)
cpd_list_norm_jit = jit(cpd_list_norm)
cpd_norm_jit = jit(cpd_norm)


for i in range(num_exp):
    tic = time.perf_counter()
    for _ in range(num_runs):
        norm = cpd_list_norm(cpd_list)
    toc = time.perf_counter()
    # print(f'List : {toc-tic}')
    times_list[i] = toc-tic
    
    tic = time.perf_counter()
    for _ in range(num_runs):
        norm = cpd_norm(cpd_tens)
    toc = time.perf_counter()
    times_tens[i] = toc-tic
    # print(f'Tens : {toc-tic}')

print(f'Mean tens: {np.mean(times_tens)}')
print(f'Stdv tens: {np.std(times_tens)}')

print(f'Mean list: {np.mean(times_list)}')
print(f'Stdv list: {np.std(times_list)}')

num_runs = 1000
num_exp = 10
times_tens_jit = np.zeros(num_exp+1)
times_list_jit = np.zeros(num_exp+1)

for i in range(num_exp+1):
    tic = time.perf_counter()
    for _ in range(num_runs):
        norm = cpd_list_norm_jit(cpd_list).block_until_ready()
    toc = time.perf_counter()
    # print(f'List : {toc-tic}')
    times_list_jit[i] = toc-tic
    
    tic = time.perf_counter()
    for _ in range(num_runs):
        norm = cpd_norm_jit(cpd_tens).block_until_ready()
    toc = time.perf_counter()
    times_tens_jit[i] = toc-tic
    # print(f'Tens : {toc-tic}')

print(f'Mean tens jit: {np.mean(times_tens_jit[1:])}')
print(f'Stdv tens jit: {np.std(times_tens_jit[1:])}')

print(f'Mean list jit: {np.mean(times_list_jit[1:])}')
print(f'Stdv list jit: {np.std(times_list_jit[1:])}')
print(f'D = {D}')
