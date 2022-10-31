#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:10:54 2022

@author: Ewoud
"""

import numpy as np

from sklearn.model_selection import train_test_split

import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import solve

# Custom modules
from cpd_functions import random_normal_cpd,  cpd_gamma_full
from cpd_training import OptimizationResult, loss_function, regularization_loss, ExperimentOptimizationResult
from feature_maps import BatchFeatureMap
from callbacks import CallBack
import cpd_linesearch
import callbacks


# =============================================================================
# Training
# =============================================================================
#### Training

def als_training(X, y, CP_rank, mapping: BatchFeatureMap, num_sweeps: int , val_split, lambda_reg, key: random.KeyArray, initial_weights=None, verbose=True, callback: CallBack = None) -> OptimizationResult:
    """
    
    Implementation of ALS algorithm for the CPD constrained kernel machine with
    as loss function the MSE plus the squared Frobenius norm of the weights times the regularization parameter:
            1/N * (y- <Z(x), W>)^2 + l * <W,W>.
            

    Parameters
    ----------
    X : Numpy ndarray
        Input data.
    y : Numpy ndarray
        Output data. For a classification problem the output must have labels [-1, 1]
    CP_rank : int
        CP-rank.
    mapping : BatchFeatureMap
        Mapping to transform the input data with.
    num_sweeps: int
        Number of sweeps to perform. One sweep is an update of all factors in the order 1,...,D.
    val_split: float
        Validation split. Must be between 0 and 1.
    lambda_reg: float
        Regularization parameter.
    key : random.KeyArray
        key that is used for any random operation. No default value is used, so it must always be passed explicitly.
    initial_weights : Numpy ndarray, optional
        Optional initial weights which should be a DxMxR ndarray. When no or wrong initial weight are passed, random normal normalized weights are used. The default is None.
    verbose : Boolean, optional
        If true, print intermediate training loss. The default is True.
    callback : CallBack, optional
        Callback that can be used to store any additional intermediate results during training. The default is None.

    Returns
    -------
    OptimizationResult
        Training result. 

    """
        
    # Parameters
    D = X.shape[1] # Is number of features
    M = mapping.order()
    method = 'ALS' 

    # Check callback
    if not issubclass(type(callback), CallBack) and callback is not None:
        print('callback not of CallBack class, so no callback is used')
        callback = None
    elif issubclass(type(callback), CallBack):
        callback.set_method(method)

    # Check if initial_weights have correct dimensions
    if initial_weights is None:
        key, subkey = random.split(key)
        weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
    else:
        #TODO: Update these checks
        if initial_weights.shape[0] != D:
            key, subkey = random.split(key)
            weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
        elif initial_weights.shape[1] != M:
            key, subkey = random.split(key)
            weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
        elif initial_weights.shape[2] != CP_rank:
            key, subkey = random.split(key)
            weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
        else:
            weights = initial_weights
            
    # Validation split
    key, subkey = random.split(key)
    random_state = int(subkey[0])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state = random_state)
    
    # Number of samples in training data
    N = X_train.shape[0]
    
    # Construct sweep sequence. On each iteration a full sweep is done
    sweep_sequence = np.arange(0, D)
    
    
    len_sweep_sequence = len(sweep_sequence)
    
    # Mean squared error that is computed and saved for each iteration
    mse_train = np.zeros(num_sweeps*len_sweep_sequence) 
    loss_train = np.zeros(num_sweeps*len_sweep_sequence)
    mse_val = np.zeros(num_sweeps*len_sweep_sequence+1)
    loss_val = np.zeros(num_sweeps*len_sweep_sequence+1)
    

    
    # Initialize objective function parts
    # Use all data on each iteration
    # Use seperate ALS function, since all data is used at once, so it is 
    # benneficial to split computation up per dimension to reduce memory requirement
    batch_ZW_full = cpd_als_batch_ZW_full(X_train, weights, mapping)
    gamma_full = cpd_gamma_full(weights)
    
    # Validation objective
    Zs_val = mapping.batch_feature_map(X_val)
    
    #JIT functions for performance
    total_loss_jit = jit(loss_function)
    reg_loss_jit = jit(regularization_loss)
    mapping_one_dimension = jit(mapping.one_dimension_feature_map)
    
    #Check whether to use normal or large scale update function
    # Use large scale function to avoid memory overload.
    if N > 1000:
        update_factor_jit = jit(update_factor_large_scale)
    else:
        update_factor_jit = jit(update_factor)
    
    # Iteration loop
    for i in range(num_sweeps):
        
        # Sweeps over the dimensions
        # Each dimensions is updated twice for each iteration
        for dim in range(len_sweep_sequence):
            
            # Compute loss and mse before updating so the loss before the first update is also recorded.
            preds = jnp.sum(batch_ZW_full, axis=1)
            errors = y_train-preds
            reg_loss = lambda_reg*jnp.sum(gamma_full)
            current_mse_train = jnp.dot(errors, errors)/N
            mse_train[i*len_sweep_sequence+dim] = current_mse_train
            loss_train[i*len_sweep_sequence+dim] = current_mse_train + reg_loss
        
            
            total_loss_val = total_loss_jit(weights, Zs_val, y_val, lambda_reg)
            mse_val[i*len_sweep_sequence+dim] = total_loss_val - reg_loss
            loss_val[i*len_sweep_sequence+dim] = total_loss_val
            
            # Call callback
            # Call before updating weights first time, so callback is also called for first weights
            if callback is not None:
                if type(callback) is callbacks.CallBackCPDNorm:
                    callback(weights)    
                else:
                    Zs = mapping.batch_feature_map(X_train)
                    callback(weights, Zs, y_train, lambda_reg, i)
            
            # Current dimension in sweep sequence
            d = sweep_sequence[dim]
            
            # Update factor and objective function components
            Zs_d = mapping_one_dimension(X_train[:,d])
            weights, batch_ZW_full, gamma_full = update_factor_jit(weights, Zs_d, y_train, 
                                                            batch_ZW_full, gamma_full, lambda_reg, d)
            
        if verbose:
            print(f'Sweep: {i}')
            print(f'Training loss: {loss_train[i*len_sweep_sequence]}')
            print(f'Validation loss: {loss_val[i*len_sweep_sequence]}')
            
            
        # End of factor loop
            
      # End of iteration loop 
     
    # Compute final validation loss
    total_loss_val = total_loss_jit(weights, Zs_val, y_val, lambda_reg)
    reg_loss = reg_loss_jit(weights, lambda_reg)
    
    loss_val[-1] = total_loss_val
    mse_val[-1] = total_loss_val-reg_loss
      
    # result = OptimizationResult(weights, mapping, method, mse_train, loss_train, mse_val, loss_val)
    result = ExperimentOptimizationResult(weights, mapping, method, mse_train, loss_train, mse_val, loss_val, {})
    
    return result

def als_els_training(X, y, CP_rank, mapping: BatchFeatureMap, num_sweeps, val_split, lambda_reg, key: random.KeyArray, initial_weights=None, verbose=True, callback: CallBack = None) -> OptimizationResult:
    """
    
    Implementation of ELS ALS algorithm for the CPD constrained kernel machine with
    as loss function the MSE plus the squared Frobenius norm of the weights times the regularization parameter:
            1/N * (y- <Z(x), W>)^2 + l * <W,W>.
            
    See Rajih et al (2008) for more information about the Enhanced Line search algorithm.
    
    Note that this training method was not used for the thesis and it working has therefore not been checked thouroughly. 

    Parameters
    ----------
    X : Numpy ndarray
        Input data.
    y : Numpy ndarray
        Output data. For a classification problem the output must have labels [-1, 1]
    CP_rank : int
        CP-rank.
    mapping : BatchFeatureMap
        Mapping to transform the input data with.
    num_sweeps: int
        Number of sweeps to perform. One sweep is an update of all factors in the order 1,...,D.
    val_split: float
        Validation split. Must be between 0 and 1.
    lambda_reg: float
        Regularization parameter.
    key : random.KeyArray
        key that is used for any random operation. No default value is used, so it must always be passed explicitly.
    initial_weights : Numpy ndarray, optional
        Optional initial weights which should be a DxMxR ndarray. When no or wrong initial weight are passed, random normal normalized weights are used. The default is None.
    verbose : Boolean, optional
        If true, print intermediate training loss. The default is True.
    callback : CallBack, optional
        Callback that can be used to store any additional intermediate results during training. The default is None.

    Returns
    -------
    OptimizationResult
        Training result. 

    """
        
    # Parameters
    D = X.shape[1] # Is number of features
    M = mapping.order()
    method = 'ALS ELS' 

    # Check callback
    if not issubclass(type(callback), CallBack) and callback is not None:
        print('callback not of CallBack class, so no callback is used')
        callback = None
    elif issubclass(type(callback), CallBack):
        callback.set_method(method)

    # Check if initial_weights have correct dimensions
    if initial_weights is None:
        key, subkey = random.split(key)
        weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
    else:
        #TODO: Update these checks
        if initial_weights.shape[0] != D:
            key, subkey = random.split(key)
            weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
        elif initial_weights.shape[1] != M:
            key, subkey = random.split(key)
            weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
        elif initial_weights.shape[2] != CP_rank:
            key, subkey = random.split(key)
            weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
        else:
            weights = initial_weights
            
    # Validation split
    key, subkey = random.split(key)
    random_state = int(subkey[0])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state = random_state)
    
    # Number of samples in training data
    N = X_train.shape[0]
    
    # Mean squared error that is computed and saved for each iteration
    num_updates = num_sweeps*(D+1)
    mse_train = np.zeros(num_updates) 
    loss_train = np.zeros(num_updates)
    mse_val = np.zeros(num_updates)
    loss_val = np.zeros(num_updates)

    #JIT functions for performance
    total_loss_jit = jit(loss_function)
    mapping_one_dimension = jit(mapping.one_dimension_feature_map)
    
    #Check whether to use normal or large scale update function

    if N > 1000:
        compute_new_factor_jit = jit(compute_new_factor_large_scale)
    else:
        compute_new_factor_jit = jit(compute_new_factor)
    
    # Initialize objective function parts
    # Use all data on each iteration
    # Use seperate ALS function, since all data is used at once, so it is 
    # benneficial to split computation up per dimension to reduce memory requirement
    batch_ZW_full = cpd_als_batch_ZW_full(X_train, weights, mapping)
    gamma_full = cpd_gamma_full(weights)
    
    # Validation objective
    Zs_val = mapping.batch_feature_map(X_val)
    
    # Store differences between last and current weights
    weights_it_2 = jnp.copy(weights)
    weights_it_1 = jnp.copy(weights)
    weights_it = jnp.zeros(weights.shape)
    g_it = weights_it_1 - weights_it_2
    
    # Sweep loop
    for i in range(num_sweeps):
   
        # Compute training loss and mse before updating (just like with GD)
        preds = jnp.sum(batch_ZW_full, axis=1)
        errors = y_train-preds
        reg_loss = lambda_reg*jnp.sum(gamma_full)
        current_mse_train = jnp.dot(errors, errors)/N
        mse_train[i*(D+1)] = current_mse_train
        loss_train[i*(D+1)] = current_mse_train + reg_loss
        
        # Compute validation MSE and loss before each update
        # Before, because then validation is also computed before first update
        
        # TODO: Can make validation use same principle as training --> only storing ZW_full
        # and update for each new weight to reduce computational load
        total_loss_val = total_loss_jit(weights_it_1, Zs_val, y_val, lambda_reg)
        mse_val[i*(D+1)] = total_loss_val - reg_loss
        loss_val[i*(D+1)] = total_loss_val
        
        
        #Perform line search
        
        # Compute coefficients of line search function
        # The line search function is a polynomial function in alpha
        #TODO: Rewrite for large data sets
        Zs = mapping.batch_feature_map(X_train)
        alpha_coeffs = cpd_linesearch.h_alpha_grad_coeffs_batch(weights_it_2, g_it, Zs, y_train, lambda_reg) 

        # Take derivatives of coefficients of the polynomial
        alpha_coeffs_deriv = cpd_linesearch.derivative_coefficients_of_polynomial(alpha_coeffs)
        
        # Compute roots to find optimal alphas
        sols = jnp.roots(jnp.flip(alpha_coeffs_deriv))
        
        # Find best alpha
        improve = False

        # Save best solution
        loss_best = current_mse_train + reg_loss
        weights_best = weights_it_2
                
        for alpha in sols:
            # complex roots not possible
            if not jnp.iscomplex(alpha):
                alpha = jnp.real(alpha)
                new_loss = jnp.polyval(jnp.flip(alpha_coeffs), alpha)
                if new_loss <= loss_best:
                    weights_best = weights_it_2 + alpha * g_it
                    loss_best = new_loss
                    improve = True         
        
        if not improve:
            print('No improvement')
        
        weights_it = weights_best
        weights_it_2 = weights_best
        batch_ZW_full = cpd_als_batch_ZW_full(X_train, weights_it, mapping)
        gamma_full = cpd_gamma_full(weights_it)   
        
        # Update factors using ALS
        for d in range(D):
            
            # Compute training loss and mse before updating (just like with GD)
            preds = jnp.sum(batch_ZW_full, axis=1)
            errors = y_train-preds
            reg_loss = lambda_reg*jnp.sum(gamma_full)
            current_mse_train = jnp.dot(errors, errors)/N
            mse_train[i*(D+1)+d+1] = current_mse_train
            loss_train[i*(D+1)+d+1] = current_mse_train + reg_loss
            
            # Compute validation MSE and loss before each update
            # Before, because then validation is also computed before first update
            
            # TODO: Can make validation use same principle as training --> only storing ZW_full
            # and update for each new weight to reduce computational load
            total_loss_val = total_loss_jit(weights_it, Zs_val, y_val, lambda_reg)
            mse_val[i*(D+1)+d+1] = total_loss_val - reg_loss
            loss_val[i*(D+1)+d+1] = total_loss_val
            
            # Call callback
            # Call before updating weights first time, so callback is also called for first weights
            if callback is not None:
                Zs = mapping.batch_feature_map(X_train)
                callback(weights, Zs, y_train, lambda_reg)
            
            # Update factor and objective function components
            Zs_d = mapping_one_dimension(X_train[:,d])
            
            #TODO: Put all this in function to jit
            # Compute components of objective for current factor
            batch_ZW_no_d = batch_ZW_full / jnp.matmul(Zs_d.T, weights_it[d, :, :])
            gamma_no_d = gamma_full / jnp.matmul(weights_it[d, :, :].T, weights_it[d, :, :])
            
            # Compute new factor
            W_new = compute_new_factor_jit(weights_it, Zs_d, y_train, batch_ZW_no_d, gamma_no_d, lambda_reg, d)
            
            # Update weights and weight related values
            weights_it = weights_it.at[d, :, :].set(W_new)
            batch_ZW_full = batch_ZW_no_d * jnp.matmul(Zs_d.T, W_new)
            gamma_full = gamma_no_d * jnp.matmul(W_new.T, W_new)
    
        weights_it_1 = weights_it
        g_it = weights_it_1 - weights_it_2
        
        
        if verbose:
            print(f'Sweep: {i}')
            print(f'Training loss: {loss_train[i*(D+1)]}')
            print(f'Validation loss: {loss_val[i*(D+1)]}')
        
        # End iteration
            
      # End of iteration loop 
    result = OptimizationResult(weights_it_1, mapping, method, mse_train, loss_train, mse_val, loss_val)
    
    return result

     
# =============================================================================
# Update factors
# =============================================================================
#### Update factors

def update_factor(weights, Zs_d, y, batch_ZW_full, gamma_full, lambda_reg, d):
    """
    Update factor d using Alternating least squares.

    Based on implementation of Wesel et al. (2021) Large-Scale learning with Fourier Features and Tensor Decompositions.

    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    Zs_d : Numpy ndarray
        Transformed data of feature d. MxN tensor.
    y : Numpy ndarray
        Array of size N containing true output.
    batch_ZW_full : Numpy ndarray
        Hadamard product sequence *_{d=1}^D Z(X_d)^T W_d for current weights. Tensor of size NxR.
    gamma_full : Numpy ndarray
        Hadamard product sequence *_{d=1}^D W_d^T W_d. Tensor of size RxR.
    lambda_reg : float
        Regularization parameter.
    d : int
        Number dimension of factor that is update.

    Returns
    -------
    weights_new : Numpy ndarray
        Updated weights
    batch_ZW_full_new : Numpy ndarray
        Updated Hadamard sequence with the new values for the weights taken into account.
    gamma_full_new : Numpy ndarray
        Updated Hadamard sequence with the new values for the weights taken into account.

    """
        
    # Compute components of objective for current factor
    batch_ZW_no_d = batch_ZW_full / jnp.matmul(Zs_d.T, weights[d, :, :])
    
    gamma_no_d = gamma_full / jnp.matmul(weights[d, :, :].T, weights[d, :, :])
    
    # Compute new factor
    W_new = compute_new_factor(weights, Zs_d, y, batch_ZW_no_d, gamma_no_d, lambda_reg, d)
    
    # Update weights and weight related values
    weights_new = weights.at[d, :, :].set(W_new)
    batch_ZW_full_new = batch_ZW_no_d * jnp.matmul(Zs_d.T, W_new)
    gamma_full_new = gamma_no_d * jnp.matmul(W_new.T, W_new)
    
    return weights_new, batch_ZW_full_new, gamma_full_new


def update_factor_large_scale(weights, Zs_d, y, batch_ZW_full, gamma_full, lambda_reg, d):
    
    """
    Update factor d using Alternating least squares. 
    This function is used when a larger number of samples is used to prevent an memory overload.
    
    Based on implementation of Wesel et al. (2021) Large-Scale learning with Fourier Features and Tensor Decompositions.

    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    Zs_d : Numpy ndarray
        Transformed data of feature d. MxN tensor.
    y : Numpy ndarray
        Array of size N containing true output.
    batch_ZW_full : Numpy ndarray
        Hadamard product sequence *_{d=1}^D Z(X_d)^T W_d for current weights. Tensor of size NxR.
    gamma_full : Numpy ndarray
        Hadamard product sequence *_{d=1}^D W_d^T W_d. Tensor of size RxR.
    lambda_reg : float
        Regularization parameter.
    d : int
        Number dimension of factor that is update.

    Returns
    -------
    weights_new : Numpy ndarray
        Updated weights
    batch_ZW_full_new : Numpy ndarray
        Updated Hadamard sequence with the new values for the weights taken into account.
    gamma_full_new : Numpy ndarray
        Updated Hadamard sequence with the new values for the weights taken into account.

    """

    # Compute components of objective for current factor
    batch_ZW_no_d = batch_ZW_full / jnp.matmul(Zs_d.T, weights[d, :, :])
    gamma_no_d = gamma_full / jnp.matmul(weights[d, :, :].T, weights[d, :, :])
    
    # Compute new factor
    W_new = compute_new_factor_large_scale(weights, Zs_d, y, batch_ZW_no_d, gamma_no_d, lambda_reg, d)
    
    # Update weights and weight related values
    weights_new = weights.at[d, :, :].set(W_new)
    batch_ZW_full_new = batch_ZW_no_d * jnp.matmul(Zs_d.T, W_new)
    gamma_full_new = gamma_no_d * jnp.matmul(W_new.T, W_new)
    
    return weights_new, batch_ZW_full_new, gamma_full_new


def compute_new_factor(weights, Zs_d, y, batch_ZW_no_d, gamma_no_d, lambda_reg, d):
  
    """
    Compute new factor d using Alternating least squares.

    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    Zs_d : Numpy ndarray
        Transformed data of feature d. MxN tensor.
    y : Numpy ndarray
        Array of size N containing true output.
    batch_ZW_no_d : Numpy ndarray
        Hadamard product sequence *_{p=1,p not d}^D Z(X_p)^T W_p for current weights. Tensor of size NxR.
    gamma_full : Numpy ndarray
        Hadamard product sequence *_{p=1, p not d}^D W_p^T W_p. Tensor of size RxR.
    lambda_reg : float
        Regularization parameter.
    d : int
        Number dimension of factor that is update.

    Returns
    -------
    W_new : Numpy ndarray
        Updated factor. Tensor of size MxR

    """
    
    # Parameters
    M = Zs_d.shape[0]
    N = Zs_d.shape[1]

    # Compute updated version of factor using normal equation
    # C based on own equations --> quicker based on profiler
    C = compute_C(Zs_d, batch_ZW_no_d, d)
    
    # C from Frederiek script
    # C = dotkron(Zs.matrices[d].T,batch_ZW_no_d);
    
    reg = lambda_reg*jnp.kron(gamma_no_d, jnp.eye(M))
    
    # Parts of normal equation and solver
    a = jnp.matmul(C.T, C) + N * reg 
    b = jnp.matmul(C.T, y) 

    w = solve(a, b, assume_a='sym')
    
    # Reshape with 'F' order to use same order as MATLAB, since code is based
    # on Matlab code by Frederiek Wesel
    W_new = w.reshape((M, -1),order='F')
    
    return W_new

def compute_new_factor_large_scale(weights, Zs_d, y, batch_ZW_no_d, gamma_no_d, lambda_reg, d):
    
    """
    Compute new factor d using Alternating least squares. Used for large number of samples.

    Parameters
    ----------
    weights : Numpy ndarray
        CPD weights stored as DxMxR tensor.
    Zs_d : Numpy ndarray
        Transformed data of feature d. MxN tensor.
    y : Numpy ndarray
        Array of size N containing true output.
    batch_ZW_no_d : Numpy ndarray
        Hadamard product sequence *_{p=1,p not d}^D Z(X_p)^T W_p for current weights. Tensor of size NxR.
    gamma_full : Numpy ndarray
        Hadamard product sequence *_{p=1, p not d}^D W_p^T W_p. Tensor of size RxR.
    lambda_reg : float
        Regularization parameter.
    d : int
        Number dimension of factor that is update.

    Returns
    -------
    W_new : Numpy ndarray
        Updated factor. Tensor of size MxR

    """
    
    # Parameters
    M = Zs_d.shape[0]
    N = Zs_d.shape[1]
    
    
    reg = lambda_reg*jnp.kron(gamma_no_d, jnp.eye(M))

    # Compute parts using large scale dot kron from Frederieks script
    CC, Cy = large_scale_dotkron(Zs_d.T, batch_ZW_no_d, y)
    
    a = CC+N*reg
    b = Cy
    
    # Solve for weight vector
    w = solve(a, b, assume_a='sym')
    
    # Reshape with 'F' order to use same order as MATLAB, since code is based
    # on Matlab code by Frederiek Wesel
    W_new = w.reshape((M, -1),order='F')
    
    return W_new

# =============================================================================
# Helper functions
# =============================================================================
#### Helper functions

def dotkron(L, R, batchSize=100):
    
    r1, c1 = L.shape
    r2, c2 = R.shape
    
    if r1 != r2:
        raise ValueError('Matrices should have equal number of rows')
        
    if r1 > 1e5:
        y = jnp.zeros(r1, c1*c2)
        for n in range(0, r1, batchSize):
            idx = min(n+batchSize-1,r1)
            y =y.at[n:idx,:].set(jnp.tile(L[n:idx,:],(1,c2))*jnp.kron(R[n:idx,:], jnp.ones((1, c1))))
    
    else:
        y = jnp.tile(L,(1,c2))*jnp.kron(R, jnp.ones((1, c1)))
      
    return y


def large_scale_dotkron(A, B, y):
    
    batch_size = 10000
    
    N, DA = A.shape
    _, DB = B.shape
    CC = jnp.zeros((DA*DB, DA*DB))
    Cy = jnp.zeros(DA*DB)
    
    # Indices to split arrays at into smaller subarrays
    indices = list(range(batch_size, N, batch_size))
    
    As = jnp.split(A, indices, axis=0)
    Bs = jnp.split(B, indices, axis=0)
    ys = jnp.split(y, indices, axis=0)
    
    for i, Asub in enumerate(As):
        
        Bsub = Bs[i]
        ysub = ys[i]
        
        temp = jnp.tile(Asub, (1, DB)) * jnp.kron(Bsub, jnp.ones((1, DA)))
            
        CC = CC + jnp.matmul(temp.T, temp)
        Cy = Cy + jnp.matmul(temp.T, ysub)
    
    return CC, Cy
    

def compute_C(Zs_d, batch_ZW_no_d, d: int):
    """
    Compute C as in implementation of Wesel et al. (2021) Large-Scale learning with Fourier Features and Tensor Decompositions.
    Here the tensorized version of C is computed first and then reshaped to into the matrix C.
    
    See thesis for a more detailed description and expression for C. 

    Parameters
    ----------
    Zs_d : Numpy ndarray
        Transformed data for feature D. Tensor of size MxN.
    batch_ZW_no_d : Numpy ndarray
        Hadamard product sequence *_{p=1,p not d}^D Z(X_p)^T W_p for current weights. Tensor of size NxR.
    d : int
        Number of dimension to compute C for.

    Returns
    -------
    C : Numpy ndarray
        C to use in ALS algorithm.

    """
    
    
    batch_ZW_no_d_T = batch_ZW_no_d.T
    Zs_outer_batch_ZWd = Zs_d[:, jnp.newaxis, :] * batch_ZW_no_d_T[jnp.newaxis, : , :]
    Zs_outer_batch_ZWd_T = jnp.transpose(Zs_outer_batch_ZWd, (2,0,1))
    C = Zs_outer_batch_ZWd_T.reshape((Zs_outer_batch_ZWd_T.shape[0],-1), order='F')
    
    return C


def cpd_als_batch_ZW_full(X, weights, mapping : BatchFeatureMap):
    
    """
    Compute the Hadamard product sequence *_{d=1}^D Z(X_d)^T W_d without looping
    over the factors. It computes the mapping for one feature at the time to
    prevent a memory overlaod.

    Parameters
    ----------
    X : Numpy ndarray
        Input data with N samples and D features.
    weights : Numpy ndarray
        CPD stored as DxMxR tensor.
    mapping : batchFeatureMap
        Mapping to transform input data with.

    Returns
    -------
    ZW_full : Numpy ndarray
        Result stored as NxR tensor.

    """
    
    D = X.shape[1]
    N = X.shape[0]
    R = weights.shape[2]
    
    ZW_full = jnp.ones((N, R))
    
    for d in range(D):
        
        W_d = weights[d]
        Z_d = mapping.one_dimension_feature_map(X[:, d])
        
        ZW_full = ZW_full * jnp.matmul(Z_d.T, W_d)
        
        
    return ZW_full
    
            
            
            