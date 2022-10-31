#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:48:52 2022

@author: Ewoud
"""

import pytest
import numpy as np
import sys

from jax import random
import jax.numpy as jnp
from scipy.linalg import norm

# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '/Users/Ewoud/Documents/Ewoud/Systems&Control/Thesis/thesis_tensor_networks/CPD JAX'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)
    
import cpd_functions
import cpd_functions_normalized


rng = np.random.default_rng()
        
def test_normalize_cpd():
    
    D = 5
    M = 5
    CP_rank = 2
    seed = rng.integers(0, 100)
    key = random.PRNGKey(seed)
    
    cpd = cpd_functions.random_uniform_cpd(D, M, CP_rank, key)
    mu = jnp.ones(CP_rank)
    
    cpd_normalized, new_mu = cpd_functions_normalized.normalize_cpd(cpd, mu)
    
    tens = cpd_functions_normalized.cpd_to_tensor(cpd, mu)
    tens_normalized = cpd_functions_normalized.cpd_to_tensor(cpd_normalized, new_mu)

    assert np.allclose(tens, tens_normalized)
    
def test_normalize_cpd_is_normalized():
    
    D = 5
    M = 5
    CP_rank = 2
    seed = rng.integers(0, 100)
    key = random.PRNGKey(seed)
    
    cpd = cpd_functions.random_uniform_cpd(D, M, CP_rank, key)
    mu = jnp.ones(CP_rank)
    
    cpd_normalized, new_mu = cpd_functions_normalized.normalize_cpd(cpd, mu)
    
    for d in range(D):
        norm = jnp.linalg.norm(cpd_normalized[d, :, :], axis=0)
        assert np.allclose(norm, jnp.ones(CP_rank))
    

def test_cpd_to_tensor():
    
    a = np.array([2, 3, 4])
    b = np.array([1, 4, 2])
    c = np.array([2,-1 ,1])
    
    factors = np.zeros((3,3,2))
    factors[0,:,0] = a
    factors[0,:,1] = a
    factors[1, :, 0] = b
    factors[1, :, 1] = b
    factors[2, :, 0] = c
    factors[2, :, 1] = c
    
    tensor = cpd_functions.cpd_to_tensor(factors)
    
    ans_rank_one = np.zeros((3, 3, 3))
    temp = np.array([[2, 8, 4], [3, 12, 6],[4, 16, 8]])
    ans_rank_one[:, :, 0] = temp*2
    ans_rank_one[:, :, 1] = temp*-1
    ans_rank_one[:, :, 2] = temp

    # Times two since CPD consists of twice this rank one tensor
    ans = ans_rank_one*2
    
    assert np.allclose(ans, tensor)

def test_cpd_norm():
    
    D = 5
    M = 5
    CP_rank = 2
    seed = rng.integers(0, 100)
    key = random.PRNGKey(seed)
    
    cpd = cpd_functions.random_uniform_cpd(D, M, CP_rank, key)
    ans = cpd_functions.cpd_norm(cpd)
    
    cpd_normalized, mu = cpd_functions_normalized.normalize_cpd(cpd, jnp.ones(CP_rank))
    cpd_normalized_norm = cpd_functions_normalized.cpd_norm(cpd_normalized, mu)
    
    assert np.allclose(ans, cpd_normalized_norm)
    

