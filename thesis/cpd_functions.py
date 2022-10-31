#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:54:00 2022

@author: Ewoud
"""

import numpy as np

import jax.numpy as jnp
from jax import random


def cpd_to_tensor(factors):
        
    """
    Converts CPD to the tensor format of the CPD by performing all the outer products
    of the factors.

    Parameters
    ----------
    factors : Numpy ndarray
        Factors of CPD stored as DxMxR tensor.

    Returns
    -------
    tensor : Numpy ndarray
        Tensor corresponding to cpd.

    """
    
    tensor = 0
    CP_rank = factors.shape[2]
    D = factors.shape[0]
    
    for r in range(CP_rank):
        
        #Construct rank one tensor
        rank_one = np.multiply.outer(factors[0,:,r], factors[1, :, r])
        for d in range(2,D):
            rank_one = np.multiply.outer(rank_one, factors[d, :, r])
                
        tensor = tensor + rank_one
        
    return tensor


def cpd_norm(factors):
    
    """
    Computes the norm of the CPD for which the factors are passed as input.

    Parameters
    ----------
    factors : Numpy ndarray
        Factors of CPD stored as DxMxR tensor.

    Returns
    -------
    norm : float
        Norm of the CPD.

    """
    
    gamma_full = cpd_gamma_full(factors)
    norm = jnp.sum(gamma_full)

    return norm


def random_uniform_cpd(D : int, M : int, CP_rank: int, key : random.KeyArray, bounds = [-1, 1], normalize=True):
    """
    Generates are CPD by drawing the vectors of the rank-one tensor from a
    uniform distribution defined by the bounds. When normalized, the vectors
    are normalized by dividing them by their norm.
    
    The CPD represents a tensor with D dimensions. Each dimension is of size M.
    

    Parameters
    ----------
    D : int
        Number of dimensions of tensor.
    M : int
        Size each dimension.
    CP_rank : int
        CP-rank.
    key : random.KeyArray
        key that is used in jax.random functions.
    bounds : list, optional
        List containing the upper and lower bound of the uniform distribution. The default is [-1, 1].
    normalize : Boolean, optional
        If true, the vectors are normalized. The default is True.

    Returns
    -------
    factors : Numpy ndarray
        CPD stored as DxMxR tensor.

    """
    
    # Parameters of uniform distribution
    a = bounds[0]
    b = bounds[1]
    
    factors = jnp.zeros((D, M, CP_rank))
    
    for d in range(D):
   
        # Split key for correct randomness
        key, subkey = random.split(key)
        factor = (b-a)*random.uniform(subkey, (M, CP_rank))+a
        
        # Normalize factor
        if normalize:
            # factor = factor/norm(factor)
            factor = factor/jnp.linalg.norm(factor, axis=0)
        
        factors = factors.at[d, :, :].set(factor)

    return factors

def random_normal_cpd(D : int, M : int, CP_rank: int, key : random.KeyArray, normalize=True, sigma=1):
    """
    Generates are CPD by drawing the vectors of the rank-one tensor from a
    normal distribution with zero mean and sigma as standard deviation. 
    When normalized, the vectors are normalized by dividing them by their norm.
    
    The CPD represents a tensor with D dimensions. Each dimension is of size M.
    

    Parameters
    ----------
    D : int
        Number of dimensions of tensor.
    M : int
        Size each dimension.
    CP_rank : int
        CP-rank.
    key : random.KeyArray
        key that is used in jax.random functions.
    normalize : Boolean, optional
        If true, the vectors are normalized. The default is True.
    sigma : float, optional
        Standard deviation of normal distribution. The default is 1.

    Returns
    -------
    factors : Numpy ndarray
        CPD stored as DxMxR tensor.

    """
        
    factors = jnp.zeros((D, M, CP_rank))
    
    for d in range(D):
        
        # Split key for correct randomness
        key, subkey = random.split(key)
        factor = random.normal(subkey, (M, CP_rank))*sigma
        
        if normalize:
            # factor = factor/norm(factor)
            factor = factor/jnp.linalg.norm(factor, axis=0)
            
        factors = factors.at[d, :, :].set(factor)
        
    return factors


def cpd_batch_ZW_full(factors, Zs):
    """
    Compute the Hadamard product sequence *_{d=1}^D Z(X_d)^T W_d without looping
    over the factors. It makes use of the tensor representations of the CPD and 
    transformed data and works on batches of data.

    Parameters
    ----------
    factors : Numpy ndarray
        CPD stored as DxMxR tensor.
    Zs : Numpy ndarray
        Transformed input data stored as DxMxN tensor.

    Returns
    -------
    ZW_full : Numpy ndarray
        Result stored as NxR tensor.

    """
    
    Zs_T = jnp.swapaxes(Zs, 1 , 2)
    ZW = jnp.matmul(Zs_T, factors)
    ZW_full = jnp.prod(ZW, axis=0)
        
    return ZW_full


def cpd_gamma_full(factors):
    """
    Compute Gamma of a CPD, where Gamma is defined as Hadamard product sequence of the 
    matrix product W^{(d)T}W^{(d)} for all factors d = 1, ..., D. 

    Parameters
    ----------
    cpd : Numpy ndarray
        CPD stored as DxMxR tensor.

    Returns
    -------
    gamma_full : Numpy ndarray
        Gamma of size RxR where R is the is the CP rank of the CPD. 

    """  
    
    factors_T = jnp.swapaxes(factors, 1, 2)
    gamma = jnp.matmul(factors_T, factors)

    gamma_full = jnp.prod(gamma, axis=0)
            
    return gamma_full