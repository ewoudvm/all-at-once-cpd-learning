#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:55:36 2022

@author: Ewoud
"""

import numpy as np

import jax.numpy as jnp
from jax import random
from jax import grad, jit

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# Custom modules
from cpd_functions import random_normal_cpd
from cpd_functions_normalized import normalize_cpd, cpd_norm
from cpd_training import OptimizationResult, ExperimentOptimizationResult
from feature_maps import BatchFeatureMap
from learning_parameters import LearningParameters
import cpd_weight_update_normalized

#TODO: Can feature_map be added later --> Zo feature map does not has to be called on full data


AVAILABLE_METHODS = ['AD Steepest Gradient Descent',
                     'Linesearch Gradient Descent', 'Linesearch Gradient Descent JAX',
                     'Adam Gradient Descent']
        

# =============================================================================
# Optimization result class
# =============================================================================

class OptimizationResultNormalized(OptimizationResult):
    
    """
    A class that represent an optimization result. It contains several attributes
    of the training process as well as the resulting model and losses.
    Specifically used for a normalized CPD.
    
    ...
    
    Attributes
    ----------
    weights: Numpy ndarray
        DxMxR array containing the final weights after training.
    mu: Numpy ndarray
        Array of size R containing the norms of the normalized CPD.
    mapping: BatchFeatureMap
        The feature map that was used.
    method: string
        The optimization method that was used. Is one of the methods contained
        in AVAILABLE_METHODS.
    mse_train: Numpy ndarray
        The MSE during training at each iteration evaluated on the whole training data set.
    loss_train: Numpy ndarray
        The total loss during training at each iteration evaluated on the whole training data set.
    mse_val: Numpy ndarray
        The MSE during training at each iteration evaluated on the validation data set.
    loss_val: Numpy ndarray
        The total loss during training at each iteration evaluated on the validation data set.   

    """
    
    def __init__(self, weights, mu, mapping, method, mse_train, loss_train, mse_val, loss_val):
        self.weights = weights
        self.mu = mu
        self.mapping = mapping
        self.method = method
        self.mse_train = mse_train
        self.loss_train = loss_train
        self.mse_val = mse_val
        self.loss_val = loss_val
        
class ExperimentOptimizationResultNormalized(ExperimentOptimizationResult):
    
    """
    A class that represent an optimization result. It contains several attributes
    of the training process as well as the resulting model and losses. It is 
    a subclass of OptimizationResult and adds to possibility to add any optimization details.
    Specifically used for a normalized CPD.
    
    ...
    
    Attributes
    ----------
    weights: Numpy ndarray
        DxMxR array containing the final weights after training.
    mu: Numpy ndarray
        Array of size R containing the norms of the normalized CPD.
    mapping: BatchFeatureMap
        The feature map that was used.
    method: string
        The optimization method that was used. Is one of the methods contained
        in AVAILABLE_METHODS.
    mse_train: Numpy ndarray
        The MSE during training at each iteration evaluated on the whole training data set. 
    loss_train: Numpy ndarray
        The total loss during training at each iteration evaluated on the whole training data set.
    mse_val: Numpy ndarray
        The MSE during training at each iteration evaluated on the validation data set.
    loss_val: Numpy ndarray
        The total loss during training at each iteration evaluated on the validation data set.   
    optimization_details: dict
        Dictionary that can be used to save any additional optimization details. 
        Since it is not specified what these might be, it cannot be garantueed that it contains any specific information.

    """
    
    def __init__(self, weights, mu, mapping, method, mse_train, loss_train, mse_val, loss_val, optimization_details):
        self.weights = weights
        self.mu = mu
        self.mapping = mapping
        self.method = method
        self.mse_train = mse_train
        self.loss_train = loss_train
        self.mse_val = mse_val
        self.loss_val = loss_val
        
        # Add dictionary to store arbitraty optimization details and additional results
        self.optimization_details = optimization_details

# =============================================================================
# Training function
# =============================================================================

def batch_training(X, y, CP_rank, mapping: BatchFeatureMap, param: LearningParameters, method: str, key: random.KeyArray, initial_weights=None, verbose=True, callback= None) -> OptimizationResultNormalized:
    
    """
    
    Implementation of all-at-once optimization framework for the CPD constrained kernel machine.
    Specifically used for training in which the CPD is normalized after each weight update.
    
    Any update method can be used that conforms the following signature:
        'weights, loss, optimizer_state = update_function(weights, mu, Zs, y_batch, lambda_reg, learning_rate, optimizer_state, i)'
    This update function must be specified in cpd_weight_update module and added here such that is used when the correct 'method' input is given.
    
    It uses a mini-batch implementation, but when the batch size is set to the number of training samples, this corresponds to full-batch training.
    Regression and binary classification problems can be solved.
    
    Note, this version of the training works slightly differently than the general cpd_training module.
    For example, ere the training losses are computed over the batch that is used to perform the weigth update.
    

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
    param : LearningParameters
        Contains many of the hyperparameters used for training.
    method : str
        All-at-once optimization method to use.
    key : random.KeyArray
        key that is used for any random operation. No default value is used, so it must always be passed explicitly.
    initial_weights : Numpy ndarray, optional
        Optional initial weights which should be a DxMxR ndarray. When no or wrong initial weight are passed, random normal normalized weights are used. The default is None.
    verbose : Boolean, optional
        If true, print intermediate training loss. The default is True.
    callback : CallBack, optional
        Callback that can be used to store any additional intermediate results during training. The default is None.
    save_detailed_training_losses : Boolean, optional
        If true, . The default is False.

    Raises
    ------
    ValueError
        Raises an error when a method that is not available is passed.

    Returns
    -------
    OptimizationResultNormalized
        Training result. 

    """

    #Parameters
    D = X.shape[1] # Is number of features
    M = mapping.order()
    learning_rate = param.learning_rate
    lambda_reg = param.lambda_reg
    num_iter = param.epochs
    batch_size = param.batch_size
    val_split = param.val_split

    # Check if initial_weights have correct dimensions
    if initial_weights is None:
        key, subkey = random.split(key)
        weights = random_normal_cpd(D, M, CP_rank, subkey, normalize=True)
    else:
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
        
    #Create normalized weights
    weights, mu = normalize_cpd(weights, jnp.ones(CP_rank))
        
    # Any update method can be added as long as it follows the same format
    # An update method should have:
        # Input:   (weight, mu, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration)
        # Output:  (new_weight, loss, new_optimizer_state)
    # Note that the update functions should have these inputs even when it does not use them    
    if method == 'AD Steepest Gradient Descent':
        update_function = jit(cpd_weight_update_normalized.AD_Steepest_Gradient_Descent())
        optimizer_state = None
    elif method == 'Linesearch Gradient Descent':
        update_function = cpd_weight_update_normalized.Line_Search_Gradient_Descent()
        optimizer_state = None
    elif method == 'Linesearch Gradient Descent JAX':
        update_function = jit(cpd_weight_update_normalized.Line_Search_Gradient_Descent_JAX())
        optimizer_state = None
    elif method == 'Adam Gradient Descent':
        update_function = jit(cpd_weight_update_normalized.Adam_Gradient_Descent())
        m = jnp.zeros((D, M ,CP_rank))
        v = jnp.zeros((D, M, CP_rank))
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        optimizer_state = (m, v, beta_1, beta_2, epsilon)
    else:
        raise ValueError(f'Unknown method: {method} \n Available methods are: {AVAILABLE_METHODS}')

        
    # Validation split
    key, subkey = random.split(key)
    random_state = int(subkey[0])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state = random_state)
        
    # Batches
    num_sam = X_train.shape[0]
    if batch_size > num_sam:
        batch_size = num_sam
        
    num_batches = int(num_sam/batch_size)
    
    # MSE and Loss saved per update of the weights
    mse_train = np.zeros(num_iter*num_batches) #MSE per update
    loss_train = np.zeros(num_iter*num_batches)
    
    #Validation loss calculated per epoch
    mse_val = np.zeros(num_iter)
    loss_val = np.zeros(num_iter)
    
    
    # JIT functions for better efficiency
    total_loss_jit = jit(loss_function)
    reg_loss_jit = jit(regularization_loss)
    mapping_function = jit(mapping.batch_feature_map)
    
    # Compute feature mapping on validation set
    # Outside of loop, so only computed once
    Zs_val = mapping_function(X_val)
    
    # Iteration loop
    i = 0
    while i < num_iter:
        
        # Compute validation MSE and loss before each update
        # Before, because then validation is also computed before first update
        total_loss_val = total_loss_jit(weights, mu, Zs_val, y_val, lambda_reg)
        reg_loss = reg_loss_jit(weights, mu, lambda_reg)
        
        loss_val[i] = total_loss_val
        mse_val[i] = total_loss_val-reg_loss
        
        
        # Shuffle data to leave out random part when data_set size is not a
        # multiple of batch_size
        # Also insure that all batches are always the same size
        # This is usefull for jitting
        X_s, y_s = shuffle(X_train, y_train, random_state=i)
        
        # Loop over batches
        for b in range(num_batches):
            
            #Current batch
            X_batch = X_s[b*batch_size:(b+1)*batch_size, :]
            y_batch = y_s[b*batch_size:(b+1)*batch_size]
                
            # Perform feature transformation
            Zs = mapping_function(X_batch)
        
            # Compute regularization loss before updateing weights
            reg_loss = reg_loss_jit(weights, mu , lambda_reg)
        
            # Perform weight update
            weights, loss, optimizer_state = update_function(weights, mu, Zs, y_batch, lambda_reg, 
                                                            learning_rate, optimizer_state, i)
            
            #Normalize weights again
            weights, mu = normalize_cpd(weights, mu)
            
            # Train cost
            loss_train[i*num_batches+b] = loss
            mse_train[i*num_batches+b] = loss - reg_loss
            

        # End batch loop
        
        
        # Print intermediate results
        if not i%10 and verbose:
            print(f"Current iteration: {i}")
            print(f'Current training loss: {loss_train[i*num_batches]}')
            print(f"Current validation loss: {loss_val[i]}")
        
        i+=1 
        # End iteration loop
    
    
    #After termination
    final_mse_train     = mse_train[0:i*num_batches]
    final_loss_train    = loss_train[0:i*num_batches]
    final_mse_val       = mse_val[0:i]
    final_loss_val      = loss_val[0:i]
    
    optimization_details = dict(normalized='True')
    
    # result = OptimizationResultNormalized(weights, mu, mapping, method, final_mse_train, final_loss_train, final_mse_val, final_loss_val)
    result = ExperimentOptimizationResultNormalized(weights, mu, mapping, method, final_mse_train, final_loss_train, final_mse_val, final_loss_val, optimization_details)
    
    return result



# =============================================================================
# Prediction and Loss functions
# =============================================================================

@jit
def prediction(weights, mu, Zs):
    """
    Computes the prediction for the given normalized CPD and mapped input.

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights/factors.
    mu: Numpy ndarray
        Array of size R containing norms of normalized CPD.
    Zs : Numpy ndarray
        DxMxN ndarray containing the transformed input data.    
    
    Returns
    -------
    pred : Numpy 1D array
        Array containing the predictions. 

    """ 
    
    # Compute full ZW
    Zs_T = jnp.swapaxes(Zs, 1 , 2)
    ZW = jnp.matmul(Zs_T, weights)
    batch_ZW = jnp.prod(ZW, axis=0)
    
    #Compute predicitions
    preds = jnp.matmul(batch_ZW, mu)
    
    return preds

@jit
def loss_function(weights, mu, Zs, y, lambda_reg):
    """
    Computes loss which is the MSE term plus the regularization term.
    The regularization loss is the squared Frobenius norm of the CPD times the regularization parameter.

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights/factors
    mu : Numpy ndarray
        Array of size R containing norms of normalized CPD.
    Zs : Numpy ndarray
        DxMxN ndarray containing the transformed input data.
    y : Numpy ndarray
        Array containing the true output values.
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    loss : float
        total loss.

    """
    
    # MSE
    N = Zs.shape[2]
    
    preds = prediction(weights, mu, Zs)
    errors = y-preds
    mse = jnp.dot(errors, errors)/N
    
    #Regularization loss
    reg_loss = regularization_loss(weights, mu , lambda_reg)
    
    loss = mse + reg_loss
    
    return loss

@jit
def regularization_loss(weights, mu , lambda_reg):
    """
    Computes regularization loss, which is the squared Frobenius norm of the CPD times the regularization parameter.

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights/factors.
    mu : Numpy ndarray
        Array of size R containing norms of normalized CPD. 
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    loss : float
        Regularization loss.

    """
    
    norm = cpd_norm(weights, mu)
    
    loss = lambda_reg*norm
    
    return loss


