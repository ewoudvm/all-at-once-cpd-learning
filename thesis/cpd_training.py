#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:55:36 2022

@author: Ewoud
"""

import numpy as np

import jax.numpy as jnp
from jax import random
from jax import jit

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Custom modules
from cpd_functions import cpd_norm, random_normal_cpd
from feature_maps import BatchFeatureMap
from learning_parameters import LearningParameters
import cpd_weight_update
from callbacks import CallBack
import callbacks


# Available all-at-once optimization methods.
# If a new method is added to cpd_weight_update, add this method here as well
AVAILABLE_METHODS = ['Steepest Gradient Descent', 'AD Steepest Gradient Descent',
                     'Linesearch Gradient Descent', 'Linesearch Gradient Descent JAX',
                     'Adam Gradient Descent']

# If data has more samples than this, the training loss is evaluated in subbatches
# of data with this size to avoid a memory overload.
LARGE_SCALE_BATCH_SIZE = 1000


# =============================================================================
# Optimization result class
# =============================================================================

class OptimizationResult:
    
    """
    A class that represent an optimization result. It contains several attributes
    of the training process as well as the resulting model and losses.
    
    ...
    
    Attributes
    ----------
    weights: Numpy ndarray
        DxMxR array containing the final weights after training.
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
    
    def __init__(self,weights, mapping, method, mse_train, loss_train, mse_val, loss_val):
        self.weights = weights
        self.mapping = mapping
        self.method = method
        self.mse_train = mse_train
        self.loss_train = loss_train
        self.mse_val = mse_val
        self.loss_val = loss_val
        

class ExperimentOptimizationResult(OptimizationResult):
       
    """
    A class that represent an optimization result. It contains several attributes
    of the training process as well as the resulting model and losses. It is 
    a subclass of OptimizationResult and adds to possibility to add any optimization details.
        
    ...
    
    Attributes
    ----------
    weights: Numpy ndarray
        DxMxR array containing the final weights after training.
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
    
    def __init__(self, weights, mapping, method, mse_train, loss_train, mse_val, loss_val, optimization_details : dict):
        
        self.weights = weights
        self.mapping = mapping
        self.method = method
        self.mse_train = mse_train
        self.loss_train = loss_train
        self.mse_val = mse_val
        self.loss_val = loss_val
        
        # Add dictionary to store arbitraty optimization details and additional results
        self.optimization_details = optimization_details
    
class CombinationExperimentOptimizationResult(OptimizationResult):
    
    """
    A class that represent an optimization result where two methods are used after each other.
    First, a first method is used and its results are contained in first_results. 
    The second method uses the final weights of the first result as its initial values.
    
    ...
    
    Attributes
    ----------
    first_result:
        Results of first optimization method.
    second_result:
        Results of second optimization method.

    """
    
    def __init__(self, first_result, second_result):
        
        self.first_result = first_result
        self.second_result = second_result
    
    

# =============================================================================
# Training function
# =============================================================================

def batch_training(X, y, CP_rank: int, mapping: BatchFeatureMap, param: LearningParameters, method: str, key: random.KeyArray, initial_weights=None, verbose=True, callback : CallBack = None, save_detailed_training_losses=False) -> OptimizationResult:
    """
    
    Implementation of all-at-once optimization framework for the CPD constrained kernel machine with
    as loss function the MSE plus the squared Frobenius norm of the weights times the regularization parameter:
            1/N * (y- <Z(x), W>)^2 + l * <W,W>.
    
    Any update method can be used that conforms the following signature:
        'weights, loss, optimizer_state = update_function(weights, Zs, y_batch, lambda_reg, learning_rate, optimizer_state, i)'
    This update function must be specified in cpd_weight_update module and added here such that is used when the correct 'method' input is given.
    
    It uses a mini-batch implementation, but when the batch size is set to the number of training samples, this corresponds to full-batch training.
    Regression and binary classification problems can be solved.
    

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
        If true, compute and print intermediate training loss. Intermediate validation loss is always computed. The default is True.
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
    OptimizationResult
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
        
        
    # Any update method can be added as long as it follows the same format
    # An update method should have:
        # Input:   (weight, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration)
        # Output:  (new_weight, loss, new_optimizer_state)
    # Note that the update functions should have these inputs even when it does not use them    
    
    if method == 'Steepest Gradient Descent':
        #update_jit =jit(update_steepest_gradient_descent)
        update_function = jit(cpd_weight_update.Steepest_Gradient_Descent())
        optimizer_state = None
    elif method == 'AD Steepest Gradient Descent':
        update_function = jit(cpd_weight_update.AD_Steepest_Gradient_Descent())
        optimizer_state = None
    elif method == 'Linesearch Gradient Descent':
        update_function = cpd_weight_update.Line_Search_Gradient_Descent()
        optimizer_state = None
    elif method == 'Linesearch Gradient Descent JAX':
        update_function = jit(cpd_weight_update.Line_Search_Gradient_Descent_JAX())
        optimizer_state = None
    elif method == 'Adam Gradient Descent':
        update_function = jit(cpd_weight_update.Adam_Gradient_Descent())
        m = jnp.zeros((D, M ,CP_rank))
        v = jnp.zeros((D, M, CP_rank))
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        optimizer_state = (m, v, beta_1, beta_2, epsilon)
    else:
        raise ValueError(f'Unknown method: {method} \n Available methods are: {AVAILABLE_METHODS}')
            
    # Check callback and add method to callback
    if not issubclass(type(callback), CallBack) and callback is not None:
        print('callback not of CallBack class, so no callback is used')
        callback = None
    elif issubclass(type(callback), CallBack):
        callback.set_method(method)
        
    # Validation split
    key, subkey = random.split(key)
    random_state = int(subkey[0])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state = random_state)
        
    # Batches
    num_sam = X_train.shape[0]
    if batch_size > num_sam:
        batch_size = num_sam
        
    num_batches = int(num_sam/batch_size)
    
    # Batch size to use for full batch prediction to use smaller subbatches
    if LARGE_SCALE_BATCH_SIZE > num_sam:
        large_scale_batch_size = num_sam
    else:
        large_scale_batch_size = LARGE_SCALE_BATCH_SIZE
        
    num_large_scale_batches = int(num_sam/large_scale_batch_size)
    
    # MSE and Loss saved per update of the weights on batch used for update
    mse_train_det = np.zeros(num_iter*num_batches) #MSE per update
    loss_train_det = np.zeros(num_iter*num_batches)
    
    # Training loss per epoch on full data set
    loss_train = np.zeros(num_iter+1)
    mse_train = np.zeros(num_iter+1)
    
    #Validation loss calculated per epoch
    mse_val = np.zeros(num_iter+1)
    loss_val = np.zeros(num_iter+1)
    
    # JIT functions for better efficiency
    total_loss_jit = jit(loss_function)
    mse_loss_jit = jit(mse_function)
    mse_loss_leftover_jit = jit(mse_function) # Seperate jit function for leftover when using subbatches due to different size
    reg_loss_jit = jit(regularization_loss)
    mapping_function = jit(mapping.batch_feature_map)
    mapping_function_training = jit(mapping.batch_feature_map) # Seperate jit function for leftover when using subbatches due to different size
    mapping_function_training_leftover = jit(mapping.batch_feature_map)
    
    # Compute feature mapping on validation set
    # Outside of loop, so only computed once
    Zs_val = mapping_function(X_val)
    
    # Iteration loop
    i = 0
    while i < num_iter: # No early stopping is used, but this could be implemented here.
        
        # Compute validation MSE and loss before each update
        # Before, because then validation is also computed before first update
        total_loss_val = total_loss_jit(weights, Zs_val, y_val, lambda_reg)
        reg_loss = reg_loss_jit(weights, lambda_reg)
        
        loss_val[i] = total_loss_val
        mse_val[i] = total_loss_val-reg_loss
        
        # Shuffle data to leave out random part when data_set size is not a
        # multiple of batch_size
        # Also insure that all batches are always the same size
        # This is usefull for jitting
        X_s, y_s = shuffle(X_train, y_train, random_state=i)
        
        if verbose:
            
            # Compute training loss on full training data set.
            for b in range(num_large_scale_batches):
                #Current batch
                X_batch = X_s[b*large_scale_batch_size:(b+1)*large_scale_batch_size, :]
                y_batch = y_s[b*large_scale_batch_size:(b+1)*large_scale_batch_size]
                    
                # Perform feature transformation
                Zs = mapping_function_training(X_batch)
                
                current_mse = mse_loss_jit(weights, Zs, y_batch, num_sam)
                loss_train[i] = loss_train[i] + current_mse
                mse_train[i] = mse_train[i] + current_mse
            
            #Final batch that otherwise gets left out, is in general different size than rest
            X_batch = X_s[(b+1)*large_scale_batch_size:, :]
            y_batch = y_s[(b+1)*large_scale_batch_size:]
                
            # Perform feature transformation
            Zs = mapping_function_training_leftover(X_batch)
            
            current_mse = mse_loss_leftover_jit(weights, Zs, y_batch, num_sam)
            loss_train[i] = loss_train[i] + current_mse
            mse_train[i] = mse_train[i] + current_mse
            
            # Add regularization term
            loss_train[i] = loss_train[i] + reg_loss_jit(weights, lambda_reg)
        
        
        #Weight update loop
        
        # Loop over batches
        for b in range(num_batches):
            
            #Current batch
            X_batch = X_s[b*batch_size:(b+1)*batch_size, :]
            y_batch = y_s[b*batch_size:(b+1)*batch_size]
                
            # Perform feature transformation
            Zs = mapping_function(X_batch)
        
            # Compute regularization loss before updateing weights
            reg_loss = reg_loss_jit(weights, lambda_reg)
            
            # Call callback
            # Call before updating weights first time, so callback is also called for first weights
            if callback is not None:
                if type(callback) is callbacks.CallBackCPDNorm:
                    callback(weights)    
                else:
                    Zs_cb = mapping.batch_feature_map(X_train)
                    callback(weights, Zs_cb, y_train, lambda_reg, i)
        
        
            # Returned loss is value before updating weights
            weights, loss, optimizer_state = update_function(weights, Zs, y_batch, lambda_reg, 
                                                            learning_rate, optimizer_state, i)
            # Train cost
            loss_train_det[i*num_batches+b] = loss
            mse_train_det[i*num_batches+b] = loss - reg_loss
            

        # End batch loop
        # Print intermediate results
        if not i%10 and verbose:
            print(f"Current iteration: {i}")
            print(f'Current training loss: {loss_train[i]}')
            print(f"Current validation loss: {loss_val[i]}")
        
        i+=1 
        # End iteration loop
        
    
    
    if verbose:
    # Final training loss
    # Compute training loss on full training data set.
        
        for b in range(num_large_scale_batches):
            #Current batch
            X_batch = X_s[b*large_scale_batch_size:(b+1)*large_scale_batch_size, :]
            y_batch = y_s[b*large_scale_batch_size:(b+1)*large_scale_batch_size]
                
            # Perform feature transformation
            Zs = mapping_function_training(X_batch)
            
            current_mse = mse_loss_jit(weights, Zs, y_batch, num_sam)
            loss_train[i] = loss_train[i] + current_mse
            mse_train[i] = mse_train[i] + current_mse
        
        #Final batch that otherwise gets left out, is in general different size than rest
        X_batch = X_s[(b+1)*large_scale_batch_size:, :]
        y_batch = y_s[(b+1)*large_scale_batch_size:]
            
        # Perform feature transformation
        Zs = mapping_function_training_leftover(X_batch)
        
        current_mse = mse_loss_leftover_jit(weights, Zs, y_batch, num_sam)
        loss_train[i] = loss_train[i] + current_mse
        mse_train[i] = mse_train[i] + current_mse
        
        #Add regularization loss
        loss_train[i] = loss_train[i] + reg_loss_jit(weights, lambda_reg)
        
    #Final validation loss
    total_loss_val = total_loss_jit(weights, Zs_val, y_val, lambda_reg)
    loss_val[i] = total_loss_val
    mse_val[i] = total_loss_val-reg_loss
    
    #After termination (in case of early stopping, but that is not implemented yet)
    final_mse_train_det     = mse_train_det[0:i*num_batches]
    final_loss_train_det    = loss_train_det[0:i*num_batches]
    final_loss_train        = loss_train[0:(i+1)]
    final_mse_train         = mse_train[0:(i+1)]
    final_mse_val           = mse_val[0:(i+1)]
    final_loss_val          = loss_val[0:(i+1)]
    
    
    if save_detailed_training_losses:
        optimization_details = dict(final_mse_train_det=final_mse_train_det, final_loss_train_det = final_loss_train_det)
    else:
        optimization_details = {}
    
    # result = OptimizationResult(weights, mapping, method, final_mse_train, final_loss_train, final_mse_val, final_loss_val)
    result = ExperimentOptimizationResult(weights, mapping, method, final_mse_train, final_loss_train, final_mse_val, final_loss_val, optimization_details)
    
    return result



# =============================================================================
# Prediction and Loss functions
# =============================================================================

@jit
def prediction(weights, Zs):
    
    """
    Computes the prediction for the given weights and mapped input.

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights.
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
    preds = jnp.sum(batch_ZW, axis=1)
    
    return preds

@jit
def mse_function(weights, Zs, y, batch_size : int):
    """
    Compute MSE for passed model and data. The batch size is passed seperately,
    sicne the function might be used on a subbatch of the data. 

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights.
    Zs : Numpy ndarray
        DxMxN ndarray containing the transformed input data.
    y : Numpy ndarray
        Array containing the true output values.
    batch_size : int
        Size of batch for which to compute the MSE.

    Returns
    -------
    mse : float
        MSE

    """
    
    preds = prediction(weights, Zs)
    errors = y-preds
    mse = jnp.dot(errors, errors)/batch_size
    
    return mse
    


@jit
def loss_function(weights, Zs, y, lambda_reg):
    """
    Computes loss which is the MSE term plus the regularization term.
    The regularization loss is the squared Frobenius norm of the CPD times the regularization parameter.

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights.
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
    
    preds = prediction(weights, Zs)
    errors = y-preds
    mse = jnp.dot(errors, errors)/N
    
    #Regularization loss
    reg_loss = regularization_loss(weights, lambda_reg)
    
    loss = mse + reg_loss
    
    return loss

@jit
def regularization_loss(weights, lambda_reg):
    """
    Computes regularization loss, which is the squared Frobenius norm of the CPD times the regularization parameter.

    Parameters
    ----------
    weights : Numpy ndarray
        DxMxR CPD weights.
    lambda_reg : float
        Regularization parameter.

    Returns
    -------
    loss : float
        Regularization loss.

    """
    
    
    norm = cpd_norm(weights)
    
    loss = lambda_reg*norm
    
    return loss


