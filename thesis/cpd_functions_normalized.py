#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:54:00 2022

@author: Ewoud
"""

import numpy as np
import jax.numpy as jnp

import cpd_functions


def cpd_to_tensor(factors, mu):
        
    """
    Converts CPD to the tensor format of the CPD by performing all the outer products
    of the factors.

    Parameters
    ----------
    factors : Numpy ndarray
        Factors of CPD stored as DxMxR tensor. The CPD is assumed to be normalized, but this is not a requirement.
    mu : Numpy 1d array
        Norms of each rank-one tensor. Mu should be a vector of size R.

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
                
        tensor = tensor + rank_one*mu[r]
        
    return tensor


def cpd_norm(factors, mu):
    """
    Computes the norm of the CPD for which the factors are passed as input.

    Parameters
    ----------
    factors : Numpy ndarray
        Factors of CPD stored as DxMxR tensor.
    mu : Numpy 1d array
        Norms of each rank-one tensor. Mu should be a vector of size R.

    Returns
    -------
    norm : float
        Norm of the CPD.

    """
    
    gamma_full = cpd_functions.cpd_gamma_full(factors)
    norm = jnp.matmul(mu, jnp.matmul(gamma_full, mu))

    return norm

def normalize_cpd(factors, mu):
    """
    Normalized the CPD by normalizing the factor matrices and moving the norm
    to the vector mu. A factor matrix is normalized when all the column vectors
    have unit length.

    Parameters
    ----------
    factors : Numpy ndarray
        Factors of CPD stored as DxMxR tensor.
    mu : Numpy 1d array
        Norms of each rank-one tensor. Mu should be a vector of size R.

    Returns
    -------
    factors : Numpy ndarray
        Factors of CPD stored as DxMxR tensor where the factors are normalized.
    mu : Numpy 1d array
        Update norms vector.

    """
    
    
    D = factors.shape[0]
    
    for d in range(D):
        
        norms = jnp.linalg.norm(factors[d,:,:], axis=0)
        factors = factors.at[d,:,:].divide(norms)
        mu = mu*norms
    
    return factors, mu

