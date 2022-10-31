#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:38:41 2022

Multiple methods are implemented to compute the coefficients of the line search
polynomial. See the thesis for more information about the line search problem
and the derivation of its exact solution.

@author: Ewoud
"""

import jax.numpy as jnp
from jax import  vmap, jit

# =============================================================================
# Line search
# =============================================================================
#### Line search


def h_alpha_grad_coeffs_loop(weights, step_direction, Zs, y, lambda_reg):
    """
    Compute coefficients for line search polynomial fby looping over the samples.
    
    Coefficients are given in order that:
        p(x) = coeffs[0] + coeffs[1]*x + .... + coeffs[n]*x**(n-1)
    
    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    step_direction : Numpy ndarray
        Step direction for weight update stored as DxMxR tensor.
    Zs : Numpy ndarray
        Transformed input data stored as DxMxN tensor.
    y : Numpy ndarray
        Array of size N with true output.
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    coeffs : Numpy ndarray
        Array containing coefficients of line search polynomial.

    """
    
    # Parameters
    N = y.shape[0]
        
    #### Error term
    coeffs = 0
    for n in range(N):
        coeffs = coeffs + h_alpha_error_coeff_single_sample(weights, step_direction, Zs[:, :, n], y[n], lambda_reg)

    
    #### Regularization
    
    reg_coeffs = regularization_coeffs(weights, step_direction, lambda_reg)
    
    coeffs = coeffs/N + reg_coeffs
    
    return coeffs

@jit
def h_alpha_grad_coeffs(weights, step_direction, Zs, y, lambda_reg):
    
    """
    Compute coefficients for line search polynomial.
    Coefficients are computed using the vmap functionallity of JAX.
    
    Coefficients are given in order that:
        p(x) = coeffs[0] + coeffs[1]*x + .... + coeffs[n]*x**(n-1)
    
    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    step_direction : Numpy ndarray
        Step direction for weight update stored as DxMxR tensor.
    Zs : Numpy ndarray
        Transformed input data stored as DxMxN tensor.
    y : Numpy ndarray
        Array of size N with true output.
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    coeffs : Numpy ndarray
        Array containing coefficients of line search polynomial.

    """
    
    # Parameters
    N = y.shape[0]

        
    #### Error term
    h_alpha_error_coeff_vmap_batch = vmap(h_alpha_error_coeff_single_sample, 
                                                   in_axes=[None, None, 2, 0, None], out_axes=0)
    
    coeffs_batch = h_alpha_error_coeff_vmap_batch(weights, step_direction, Zs, y, lambda_reg)
    coeffs = jnp.sum(coeffs_batch, axis=0)
    
    #### Regularization
    
    reg_coeffs = regularization_coeffs(weights, step_direction, lambda_reg)
    
    coeffs = coeffs/N + reg_coeffs
    
    return coeffs
    
@jit
def h_alpha_error_coeff_single_sample(weights, step_direction, Z_x, y_n, lambda_reg):
    """
    Compute coefficients of line search polynomial for a single sample.
    Coefficients are given in order that:
        p(x) = coeffs[0] + coeffs[1]*x + .... + coeffs[n]*x**(n-1)

    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    step_direction : Numpy ndarray
        Step direction for weight update stored as DxMxR tensor.
    Z_x : Numpy ndarray
        Transformed input sample stored as DxM tensor.
    y : float
        True output
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    coeffs : Numpy ndarray
        Array containing coefficients of line search polynomial.

    """
    
    #W_new = W + alpha*dW
    
    D = weights.shape[0]
    R = weights.shape[2]
    
    lin_coeffs_total = jnp.zeros(D+1)
    squared_coeffs_total = jnp.zeros(2*D+1)
    
    # Seperate coefficient for y^2
    a_y2 =  y_n*y_n

    # First order
    W = weights[0]
    dW = step_direction[0]
    Z_x_d = Z_x[0,:]

    a_0 = jnp.matmul(Z_x_d.T, W)
    a_1 = jnp.matmul(Z_x_d.T, dW)
    
    lin_coeffs = jnp.zeros((D+1,R))
    lin_coeffs = lin_coeffs.at[0].set(a_0)
    lin_coeffs = lin_coeffs.at[1].set(a_1)
        
    for d in range(1,D):
        
        # To prevent confusion with zero indexing
        end_index_lin = d+1
        
        W = weights[d]
        dW = step_direction[d]
        Z_x_d = Z_x[d,:]

        a_0 = jnp.matmul(Z_x_d.T, W)
        a_1 = jnp.matmul(Z_x_d.T, dW)
    
        # Compute coefficient efficiently by update multiple terms at once. Terms that are not relavant yet are zero, so they have no influece.
        outer_prod_column_1 = lin_coeffs[1:end_index_lin]*a_0
        outer_prod_column_2 = lin_coeffs[:end_index_lin-1]*a_1

        
        lin_coeffs = lin_coeffs.at[0].set(lin_coeffs[0]*a_0)
        lin_coeffs = lin_coeffs.at[end_index_lin].set(lin_coeffs[end_index_lin-1]*a_1)
        lin_coeffs = lin_coeffs.at[1:end_index_lin].set(outer_prod_column_1+outer_prod_column_2)
    
    
    # Compute squared coefficients from linear coefficients using FOIL method (see wikipedia)
    # https://stackoverflow.com/questions/5413158/multiplying-polynomials-in-python
    lin_coeffs_summed = jnp.sum(lin_coeffs, axis = 1)
    squared_coeffs = jnp.convolve(lin_coeffs_summed, lin_coeffs_summed)
        
    # Add coefficients for each sample
    lin_coeffs_total = - 2*y_n*lin_coeffs_summed
    squared_coeffs_total = squared_coeffs
        
        
    
    coeffs = jnp.zeros((2*D+1))
    coeffs = coeffs.at[0:D+1].set(lin_coeffs_total)
    coeffs = coeffs.at[0].add(a_y2)
    coeffs = coeffs + squared_coeffs_total
        
    return coeffs

@jit
def h_alpha_grad_coeffs_batch(weights, step_direction, Zs, y, lambda_reg):
    """
    Compute coefficients for line search polynomial.
    Coefficients are computed for a batch of samples at once using a batch implementation.
    
    Coefficients are given in order that:
        p(x) = coeffs[0] + coeffs[1]*x + .... + coeffs[n]*x**(n-1)
    
    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    step_direction : Numpy ndarray
        Step direction for weight update stored as DxMxR tensor.
    Zs : Numpy ndarray
        Transformed input data stored as DxMxN tensor.
    y : Numpy ndarray
        Array of size N with true output.
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    coeffs : Numpy ndarray
        Array containing coefficients of line search polynomial.

    """
    
    # Parameters
    N = y.shape[0]
    D = Zs.shape[0]
    R = weights.shape[2]
    
    #### Error term
    
    # Seperate coefficient for y^2
    a_y2 = jnp.dot(y, y)

    # First order
    W = weights[0]
    dW = step_direction[0]
    Zs_d = Zs[0,:, :]

    a_0 = jnp.matmul(Zs_d.T, W)
    a_1 = jnp.matmul(Zs_d.T, dW)

    # For each sample the coefficient is computed, so N coefficients of size (D+1)*R
    lin_coeffs = jnp.zeros((D+1, N, R))
    lin_coeffs = lin_coeffs.at[0].set(a_0)
    lin_coeffs = lin_coeffs.at[1].set(a_1)
    
    for d in range(1,D):
        
        # To prevent confusion with zero indexing
        end_index_lin = d+1
        
        W = weights[d]
        dW = step_direction[d]
        Zs_d = Zs[d,:, :]

        a_0 = jnp.matmul(Zs_d.T, W)
        a_1 = jnp.matmul(Zs_d.T, dW)
    
        # Compute coefficient efficiently by update multiple terms at once. Terms that are not relavant yet are zero, so they have no influece.
        outer_prod_column_1 = lin_coeffs[1:end_index_lin]*a_0
        outer_prod_column_2 = lin_coeffs[:end_index_lin-1]*a_1

        lin_coeffs = lin_coeffs.at[0].set(lin_coeffs[0]*a_0)
        lin_coeffs = lin_coeffs.at[end_index_lin].set(lin_coeffs[end_index_lin-1]*a_1)
        lin_coeffs = lin_coeffs.at[1:end_index_lin].set(outer_prod_column_1+outer_prod_column_2)
    
    
    # Compute squared coefficients from linear coefficients using FOIL method (see wikipedia)
    # https://stackoverflow.com/questions/5413158/multiplying-polynomials-in-python
    # Sum over CP_rank axis
    lin_coeffs_summed = jnp.sum(lin_coeffs, axis = 2)
    
    
    # Combine coefficients for all samples
    lin_coeffs_total = -2*jnp.matmul(lin_coeffs_summed, y)
        
    squared_coeffs = vmap(jnp.convolve, in_axes = 1, out_axes = 1)(lin_coeffs_summed, lin_coeffs_summed)
    squared_coeffs_total = jnp.sum(squared_coeffs, axis=1)
    
    coeffs = squared_coeffs_total
    coeffs = coeffs.at[0:D+1].add(lin_coeffs_total)
    coeffs = coeffs.at[0].add(a_y2)        
    
    #### Regularization
    
    reg_coeffs = regularization_coeffs(weights, step_direction, lambda_reg)
        
    coeffs = coeffs/N + reg_coeffs
    
    return coeffs

@jit
def regularization_coeffs(weights, step_direction, lambda_reg):
    """
    Compute coefficients for regularization term of line search polynomial.
    
    Coefficients are given in order that:
        p(x) = coeffs[0] + coeffs[1]*x + .... + coeffs[n]*x**(n-1)
    
    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    step_direction : Numpy ndarray
        Step direction for weight update stored as DxMxR tensor.
    lambda_reg : float
        Regularization parameter.
        
    """
    
    D = weights.shape[0]
    R = weights.shape[2]
    
    reg_coeffs = jnp.zeros((2*D+1, R, R))
    
    # First order
    W = weights[0]
    dW = step_direction[0]
    
    a_0 = jnp.matmul(W.T, W)
    a_1 = jnp.matmul(dW.T, W)+jnp.matmul(W.T, dW)
    a_2 = jnp.matmul(dW.T, dW)
    
    reg_coeffs = reg_coeffs.at[0].set(a_0)
    reg_coeffs= reg_coeffs.at[1].set(a_1)
    reg_coeffs= reg_coeffs.at[2].set(a_2)
    
    for d in range(1,D):
        
        
        W = weights[d]
        dW = step_direction[d]
        
        a_0 = jnp.matmul(W.T, W)
        a_1 = jnp.matmul(dW.T, W)+jnp.matmul(W.T, dW)
        a_2 = jnp.matmul(dW.T, dW)
    
        # To eliminate confusion with 0-indexing
        dimension = d+1
        # Index of last entry in coeff array for current dimension
        # Ex: if d = 0 --> first dimension --> 3 coeffcients: a_0 a_1 a_3 --> last index is 2 (zero indexig)
        # Ex: if d = 2 --> three dimensions --> 7 coefficients --> last index is 6 (zero indexing)
        end_index = dimension*2
        
        # First two and last two dimension seperately because vecs smaller than 3
        # First
        updated_coeffs = jnp.ones((2*D+1, R, R))
        
        updated_coeffs = updated_coeffs.at[0].set(reg_coeffs[0]*a_0)
        updated_coeffs = updated_coeffs.at[1].set(reg_coeffs[0]*a_1 + reg_coeffs[1]*a_0)
        
        # Last
        updated_coeffs = updated_coeffs.at[end_index-1].set(reg_coeffs[end_index-3]*a_2 + reg_coeffs[end_index-2]*a_1)
        updated_coeffs = updated_coeffs.at[end_index].set(reg_coeffs[end_index-2]*a_2)
        
        outer_prod_1 = reg_coeffs[2:end_index-1]*a_0
        outer_prod_2 = reg_coeffs[1:end_index-2]*a_1
        outer_prod_3 = reg_coeffs[:end_index-3]*a_2
        
        updated_coeffs = updated_coeffs.at[2:end_index-1].set(outer_prod_1+outer_prod_2+outer_prod_3)
        
        reg_coeffs = updated_coeffs
        
    return lambda_reg*jnp.sum(reg_coeffs, axis=(1,2))

@jit
def derivative_coefficients_of_polynomial(coeffs):
    """
    Take derivate of polynomial with coeffs as coefficients. 
    Coefficients are given in order that:
        p(x) = coeffs[0] + coeffs[1]*x + .... + coeffs[n]*x**(n-1)

    Parameters
    ----------
    coeffs : Numpy ndarray
        Coefficients of polynomial.

    Returns
    -------
    deriv_coeffs: Numpy ndarray
        Coefficients of polynomial after the derivative is taken.

    """
 
    deriv_coeffs = jnp.arange(0, coeffs.shape[0])*coeffs
        
    return deriv_coeffs[1:]

@jit
def roots_no_zeros(coeffs):
    """
    Compute roots of polynomial with coeffs as coefficients.
    Coefficients are given in the order that:
        p(x) = coeffs[0]*x**n + coeffs[1]*x**(n-1) + ... + coeffs[n]
    
    
    Function basesd on JAX function roots and _roots_no_zeros.
    Adapted to make jittable.

    Parameters
    ----------
    coeffs : Numpy ndarray
        Coefficients of polynomial

    Returns
    -------
    roots : Numpy ndarray
        Roots of polynomial

    """

    # Based on _roots_no_zeros and roots JAX function
    # Copied code because JAX jnp.roots function was not jittalbe, because
    # input depended on inputs of overaching function
    # The context ensures that there are no trailing zeros, so
    # therefore it was possible to copy code
    
    A = jnp.diag(jnp.ones((coeffs.size - 2,)), -1)
    A = A.at[0, :].set(-coeffs[1:] / coeffs[0])
    return jnp.linalg.eigvals(A)
    
    
    
    
    