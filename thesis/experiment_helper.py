#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:35:34 2022

@author: Ewoud
"""

import time
import datetime
import pickle

from jax import random, jit
import jax.numpy as jnp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Custom modules
import cpd_training
import cpd_functions
import cpd_weight_update
import cpd_als
import feature_maps
import learning_parameters
from callbacks import CallBack
import callbacks


class Experiment:
    
    """
    Class for an experiment. It contains the parameters of the experiment and
    the results.
    
    ...
    
    Attributes
    ----------
    data_file_name: str
        File name of data file.
    results : list[cpd_training.OptimizationResult]
        List of results.
    mappings : list[feature_maps.BatchFeatureMap]
        List of used mappings. If only one mapping is used, it must still be stored in a list for compatability.
    learning_parameters: list[learning_parameters.Learning_Parameters]
        List of used learning parameters.
    seed : int
        Seed that is used in experiment.
    initial_weights : list, optional
        Initial weights that are used. Default is None.
    callbacks: list[Callback], optional
        List of used callbacks. Default is None.
    
    """
    
    def __init__(self, 
                 data_file_name: str,
                 results: list[cpd_training.OptimizationResult],
                 mappings: list[feature_maps.BatchFeatureMap],
                 learning_parameters: list[learning_parameters],
                 seed,
                 initial_weights: list = None,
                 callbacks: list[CallBack] = None):
        
        self.data_file_name = data_file_name
        self.results = results
        self.mappings = mappings
        self.learning_parameters = learning_parameters
        self.seed = seed
        self.initial_weights = initial_weights
        self.callbacks = callbacks
    
class TimingExperiment(Experiment):
    
    """
    Class for an experiment. It contains the parameters of the experiment and
    the results.
    This class specifically also contains timing results. 
    
    ...
    
    Attributes
    ----------
    data_file_name: str
        File name of data file.
    results : list[cpd_training.OptimizationResult]
        List of results.
    times: list
        List of run times.
    mappings : list[feature_maps.BatchFeatureMap]
        List of used mappings. If only one mapping is used, it must still be stored in a list for compatability.
    learning_parameters: list[learning_parameters.Learning_Parameters]
        List of used learning parameters.
    seed : int
        Seed that is used in experiment.
    initial_weights : list, optional
        Initial weights that are used. Default is None.
    callbacks: list[Callback], optional
        List of used callbacks. Default is None.
    
    """
    
    def __init__(self, 
                 data_file_name: str,
                 results: list[cpd_training.OptimizationResult],
                 times: list,
                 mappings: list[feature_maps.BatchFeatureMap],
                 learning_parameters: list[learning_parameters],
                 seed,
                 initial_weights: list = None,
                 callbacks: list[CallBack] = None):
        
        self.data_file_name = data_file_name
        self.results = results
        self.times = times
        self.mappings = mappings
        self.learning_parameters = learning_parameters
        self.seed = seed
        self.initial_weights = initial_weights
        self.callbacks = callbacks
        
class MethodCombiationExperiment(Experiment):
    
    """
    Class for an experiment. It contains the parameters of the experiment and
    the results.
    Class can be used for experiment where methods are combined.
    
    ...
    
    Attributes
    ----------
    data_file_name: str
        File name of data file.
    results_per_method : list[list[cpd_training.OptimizationResult]]
        List of results per method that is used.
    index_method_switches: list[int]
        List at which indices the methods switch.
    mappings : list[feature_maps.BatchFeatureMap]
        List of used mappings. If only one mapping is used, it must still be stored in a list for compatability.
    learning_parameters: list[learning_parameters.Learning_Parameters]
        List of used learning parameters.
    seed : int
        Seed that is used in experiment.
    initial_weights : list, optional
        Initial weights that are used. Default is None.
    callbacks: list[Callback], optional
        List of used callbacks. Default is None.
    
    """
    
    def __init__(self,
                 data_file_name: str,
                 results_per_method: list[list[cpd_training.OptimizationResult]],
                 index_method_switches: list[int],
                 mappings: list[feature_maps.BatchFeatureMap],
                 learning_parameters: list[learning_parameters],
                 seed,
                 initial_weights: list = None,
                 callbacks: list[CallBack] = None):
        
        self.data_file_name = data_file_name
        self.results_per_method = results_per_method
        self.index_method_switches = index_method_switches
        self.mappings = mappings
        self.learning_parameters = learning_parameters
        self.seed = seed
        self.initial_weights = initial_weights
        self.callbacks = callbacks
        

# =============================================================================
# Conduct experiment
# =============================================================================

def als_adam_experiment(X, y, CP_rank, mapping: feature_maps.BatchFeatureMap, param: learning_parameters.LearningParameters, 
                        num_sweeps : int, seed : int, num_exp: int, data_file_name : str, verbose = True,
                        cb_adam = None, cb_als = None) -> Experiment:
    """
    Function to conduct an experiment with the ALS and Adam method.

    Parameters
    ----------
    X : Numpy ndarray
        Input data of size NxD.
    y : Numpy ndarray
        True output data, array of size N.
    CP_rank : int
        CP-rank.
    mapping : feature_maps.BatchFeatureMap
        Feature map to use for transformation of input data.
    param : learning_parameters.LearningParameters
        Learning parameters / hyper parameters.
    num_sweeps : int
        Number of sweeps for ALS method.
    seed : int
        Seed to use.
    num_exp : int
        Number of experiment to conduct.
    data_file_name : str
        Name of data file. This is solely used for storing, so not for any data loading.
    verbose : Boolean, optional
        Passed to training functions. The default is True.
    cb_adam : Callback, optional
        Any callback to be used with the Adam method. The default is None.
    cb_als : Callback, optional
        Any callback to be sued with the ALS method. The default is None.

    Returns
    -------
    experiment: Experiment
        Return experiment result and details. 

    """
    
    lambda_reg = param.lambda_reg
    val_split = param.val_split
    
    key = random.PRNGKey(seed)
    keys = random.split(key, num=num_exp+1)
    training_keys = random.split(keys[0], num=num_exp)
    
    M = mapping.order()
    D = X.shape[1]
    
    results_als = []
    results_adam = []

    for i in range(num_exp):
        
        print(f'Experiment {i}')
        
        # Initial weights
        initial_weights = cpd_functions.random_normal_cpd(D, M, CP_rank, keys[i+1])
        initial_weights_als = jnp.copy(initial_weights)
        initial_weights_adam = jnp.copy(initial_weights)
        
        # ADAM
        tic = time.perf_counter()
        result_adam = cpd_training.batch_training(X, y, CP_rank, mapping, param, method = 'Adam Gradient Descent', key=training_keys[i], initial_weights = initial_weights_adam, verbose=verbose, callback=cb_adam)
        toc = time.perf_counter()
        result_adam.optimization_details['Time'] = toc - tic
        results_adam.append(result_adam)
        
        # ALS
        tic = time.perf_counter()
        result_als = cpd_als.als_training(X, y, CP_rank, mapping, num_sweeps, val_split, lambda_reg, training_keys[i], initial_weights=initial_weights_als, verbose = verbose, callback=cb_als)
        toc = time.perf_counter()
        result_als.optimization_details['Time'] = toc - tic
        results_als.append(result_als)



    # =============================================================================
    # Storing results
    # =============================================================================
    #### Storing results

    results = results_als+results_adam
    mappings = [mapping]
    params = [param]
    callbacks = [cb_adam, cb_als]
    experiment = Experiment(data_file_name = data_file_name,
                                           results=results,
                                           mappings=mappings,
                                           learning_parameters=params,
                                           seed=seed,
                                           callbacks=callbacks)
    
    return experiment


def adam_experiment(X, y, CP_rank, mapping: feature_maps.BatchFeatureMap, param: learning_parameters.LearningParameters, 
                        seed : int, num_exp: int, data_file_name : str, verbose = True, cb_norm = True, normalize = True):
    
    """
    Function to conduct an experiment with the Adam method.

    Parameters
    ----------
    X : Numpy ndarray
        Input data of size NxD.
    y : Numpy ndarray
        True output data, array of size N.
    CP_rank : int
        CP-rank.
    mapping : feature_maps.BatchFeatureMap
        Feature map to use for transformation of input data.
    param : learning_parameters.LearningParameters
        Learning parameters / hyper parameters.
    seed : int
        Seed to use.
    num_exp : int
        Number of experiment to conduct.
    data_file_name : str
        Name of data file. This is solely used for storing, so not for any data loading.
    verbose : Boolean, optional
        Passed to training functions. The default is True.
    cb_norm : Boolean, optional
        If true, a CallbackCPDNorm is used to store the CPD norm during training. The default is True.
    normalize : Boolean, optional
        If true, the initial weights are normalized. The default is True.

    Returns
    -------
    experiment: Experiment
        Return experiment result and details. 

    """
    
    epochs = param.epochs
    batch_size = param.batch_size
    
    key = random.PRNGKey(seed)
    keys = random.split(key, num=num_exp+1)
    training_keys = random.split(keys[0], num=num_exp)
    
    M = mapping.order()
    D = X.shape[1]
    
    results_adam = []
    callbacks_list = []

    for i in range(num_exp):
        
        print(f'Experiment {i}')
        
        # Initial weights
        initial_weights = cpd_functions.random_normal_cpd(D, M, CP_rank, keys[i+1], normalize=normalize)
        initial_weights_adam = jnp.copy(initial_weights)
        

        
        if cb_norm:
            D = X.shape[1]
            max_iter = int(X.shape[0]/batch_size)*epochs
            cb = callbacks.CallBackCPDNorm(D, M, CP_rank, max_iter)
            callbacks_list.append(cb)
        else:
            cb = None
        
        # ADAM
        tic = time.perf_counter()
        result_adam = cpd_training.batch_training(X, y, CP_rank, mapping, param, method = 'Adam Gradient Descent', key=training_keys[i], initial_weights = initial_weights_adam, verbose=verbose, callback=cb)
        toc = time.perf_counter()
        result_adam.optimization_details['Time'] = toc - tic
        results_adam.append(result_adam)

    # =============================================================================
    # Storing results
    # =============================================================================
    #### Storing results

    results = results_adam
    mappings = [mapping]
    params = [param]
    experiment = Experiment(data_file_name = data_file_name,
                                           results=results,
                                           mappings=mappings,
                                           learning_parameters=params,
                                           seed=seed,
                                           callbacks=callbacks_list)
    
    return experiment

def als_experiment(X, y, CP_rank, mapping: feature_maps.BatchFeatureMap, num_sweeps:int, param: learning_parameters.LearningParameters, 
                        seed : int, num_exp: int, data_file_name : str, verbose = True, cb_norm = None, normalize = True):
    
    """
    Function to conduct an experiment with the ALS method.

    Parameters
    ----------
    X : Numpy ndarray
        Input data of size NxD.
    y : Numpy ndarray
        True output data, array of size N.
    CP_rank : int
        CP-rank.
    mapping : feature_maps.BatchFeatureMap
        Feature map to use for transformation of input data.
    num_sweeps: int
        Number of sweeps to use.
    param : learning_parameters.LearningParameters
        Learning parameters / hyper parameters.
    seed : int
        Seed to use.
    num_exp : int
        Number of experiment to conduct.
    data_file_name : str
        Name of data file. This is solely used for storing, so not for any data loading.
    verbose : Boolean, optional
        Passed to training functions. The default is True.
    cb_norm : Boolean, optional
        If true, a CallbackCPDNorm is used to store the CPD norm during training. The default is True.
    normalize : Boolean, optional
        If true, the initial weights are normalized. The default is True.

    Returns
    -------
    experiment: Experiment
        Return experiment result and details. 

    """
    
    lambda_reg = param.lambda_reg
    val_split = param.val_split
    
    key = random.PRNGKey(seed)
    keys = random.split(key, num=num_exp+1)
    training_keys = random.split(keys[0], num=num_exp)
    
    M = mapping.order()
    D = X.shape[1]
    
    results_als = []
    callbacks_list = []

    for i in range(num_exp):
        
        print(f'Experiment {i}')
        
        # Initial weights
        initial_weights = cpd_functions.random_normal_cpd(D, M, CP_rank, keys[i+1], normalize = normalize)
        initial_weights_als = jnp.copy(initial_weights)
        
        if cb_norm:
            cb = callbacks.CallBackCPDNorm(D, M, CP_rank, D*num_sweeps)
            callbacks_list.append(cb)
        else:
            cb = None
        
        # ALS
        tic = time.perf_counter()
        result_als = cpd_als.als_training(X, y, CP_rank, mapping, num_sweeps, val_split, lambda_reg, training_keys[i], initial_weights=initial_weights_als, verbose = verbose, callback=cb)
        toc = time.perf_counter()
        result_als.optimization_details['Time'] = toc - tic
        results_als.append(result_als)
        
    # =============================================================================
    # Storing results
    # =============================================================================
    #### Storing results

    results = results_als
    mappings = [mapping]
    params = [param]
    experiment = Experiment(data_file_name = data_file_name,
                                           results=results,
                                           mappings=mappings,
                                           learning_parameters=params,
                                           seed=seed,
                                           callbacks=callbacks_list)
    
    return experiment

def adam_als_combination_experiment(switch_index: int, X, y, CP_rank, mapping: feature_maps.BatchFeatureMap, 
                                    param: learning_parameters.LearningParameters, 
                                    num_sweeps: int, seed : int, num_exp: int, data_file_name : str, verbose = True):
    """
    
    Function to conduct experiment in which the Adam method is used first and then ALS. 

    Parameters
    ----------
    switch_index : int
        Index to switch from Adam to ALS method.
    X : Numpy ndarray
        Input data of size NxD.
    y : Numpy ndarray
        True output data, array of size N.
    CP_rank : int
        CP-rank.
    mapping : feature_maps.BatchFeatureMap
        Feature map to use for transformation of input data.
    num_sweeps: int
        Number of sweeps to use.
    param : learning_parameters.LearningParameters
        Learning parameters / hyper parameters.
    seed : int
        Seed to use.
    num_exp : int
        Number of experiment to conduct.
    data_file_name : str
        Name of data file. This is solely used for storing, so not for any data loading.
    verbose : Boolean, optional
        Passed to training functions. The default is True.

    Returns
    -------
    experiment : TYPE
        DESCRIPTION.

    """
    
    
    lambda_reg = param.lambda_reg
    val_split = param.val_split
    
    key = random.PRNGKey(seed)
    keys = random.split(key, num=num_exp+1)
    training_keys = random.split(keys[0], num=num_exp)
    
    epochs = param.epochs
    # Number of epochs of adam method for adam als combo
    epochs_adam_combo = switch_index
    
    #Number of sweeps of als method for adam als combo
    epochs_left = epochs - epochs_adam_combo
    D = X.shape[1]
    num_sweeps_als_combo = int(epochs_left/D)
    
    #New parameters for Adam ALS combo
    param_adam_combo = learning_parameters.LearningParameters(param.lambda_reg, param.learning_rate,
                                                            epochs_adam_combo, param.batch_size, 
                                                            param.val_split)
    
    M = mapping.order()
    
    results_adam = []
    results_als = []
    results_adam_als_combo = []

    for i in range(num_exp):

        print(f'Experiment {i}')
        
        # Initial weights
        initial_weights = cpd_functions.random_normal_cpd(D, M, CP_rank, keys[i+1])
        initial_weights_als = jnp.copy(initial_weights)
        initial_weights_adam = jnp.copy(initial_weights)
        initial_weights_adam_combo = jnp.copy(initial_weights)
        
        # ADAM
        tic = time.perf_counter()
        result_adam = cpd_training.batch_training(X, y, CP_rank, mapping, param, method = 'Adam Gradient Descent', key=training_keys[i], initial_weights = initial_weights_adam, verbose=verbose)
        toc = time.perf_counter()
        result_adam.optimization_details['Time'] = toc - tic
        results_adam.append(result_adam)
        
        # ALS
        tic = time.perf_counter()
        result_als = cpd_als.als_training(X, y, CP_rank, mapping, num_sweeps, val_split, lambda_reg, training_keys[i], initial_weights=initial_weights_als, verbose = verbose)
        toc = time.perf_counter()
        result_als.optimization_details['Time'] = toc - tic
        results_als.append(result_als)
        
        # Combo
        tic = time.perf_counter()
        result_adam_combo = cpd_training.batch_training(X, y, CP_rank, mapping, param_adam_combo, method = 'Adam Gradient Descent', key=training_keys[i], initial_weights = initial_weights_adam_combo, verbose=verbose)
        toc = time.perf_counter()
        result_adam_combo.optimization_details['Time'] = toc - tic
        #Copy final weights of Adam combo to become initial weights of ALS combo
        initial_weights_als_combo = jnp.copy(result_adam_combo.weights)
        tic = time.perf_counter()
        result_als_combo = cpd_als.als_training(X, y, CP_rank, mapping, num_sweeps_als_combo, val_split, lambda_reg, training_keys[i], initial_weights=initial_weights_als_combo, verbose = verbose)
        toc = time.perf_counter()
        result_als_combo.optimization_details['Time'] = toc - tic
        
        combo_result = cpd_training.CombinationExperimentOptimizationResult(result_adam_combo, result_als_combo)
        
        results_adam_als_combo.append(combo_result)
        
    # End experiment loop    
        
    results = results_als+results_adam+results_adam_als_combo
    mappings = [mapping]
    params = [param, param_adam_combo]
    experiment = Experiment(data_file_name = data_file_name,
                                           results=results,
                                           mappings=mappings,
                                           learning_parameters=params,
                                           seed=seed)
    
    return experiment
        
# =============================================================================
# Storing and loading results
# =============================================================================
#### Storing and loading results

def store_results(experiment_name: str, experiment: Experiment, large_experiment=False):

    """
    Method for storing an experiment using Python pickling.
    """
    
    datetime_stamp = str(datetime.datetime.now())
    
    # If the experiment is large use compression
    if large_experiment:
        # file_name = experiment_name + '--' + datetime_stamp + '.pbz2'
        # full_file_name = 'experiment results/'+file_name
        # with bz2.BZ2File(full_file_name, 'w') as f:
        #     pickle.dump(experiment, f)
        
        #NOTE: Design decision to throw away training data when experiment is to large
        for result in experiment.results:
            result.loss_train = None
            result.mse_train = None
          
            
    file_name = experiment_name + '--' + datetime_stamp + '.pickle'
    full_file_name = 'experiment results/'+file_name
    with open(full_file_name, 'wb') as f:
        pickle.dump(experiment, f)
         
    # else:
    #     file_name = experiment_name + '--' + datetime_stamp + '.pickle'
    #     full_file_name = 'experiment results/'+file_name
    #     with open(full_file_name, 'wb') as f:
    #         pickle.dump(experiment, f)
    
    return 'Stored and finished experiment'

def load_results(file_name: str) -> Experiment:
    
    """
    Method for loading an experiment that is stored as a pickled file.
    
    NOTE: THIS IS NOT SAVE!! SO IF THE ORIGIN OF THE PICKLE FILE IS UNKNOWN DO
    NOT USE THIS!!
    """
    
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data

# =============================================================================
# Search convergence
# =============================================================================

def find_convergence(results: list[cpd_training.OptimizationResult]):
    
    #Create nested list of losses where each method has seperate list of losses
    losses_train = []
    losses_val = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses_train.append([])
            losses_val.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses_train[-1].append(result.loss_train)
            losses_val[-1].append(result.loss_val)
            
        else:
            method_index = assigned_methods.index(method)
            losses_train[method_index].append(result.loss_train)
            losses_val[method_index].append(result.loss_val)
    
    convergence_indices = {}
    
    # Compute Adam convergence
    adam_method_index = assigned_methods.index('Adam Gradient Descent')
    
    #Losses corresponding to current method
    current_losses_train = losses_train[adam_method_index]
    current_losses_val = losses_val[adam_method_index]
    
    #Compute mean losses
    adam_mean_losses_train = np.mean(current_losses_train,axis=0)
    adam_mean_losses_val = np.mean(current_losses_val, axis=0)
    
    adam_min_mean_losses_train = np.amin(adam_mean_losses_train)
    adam_min_mean_losses_val = np.amin(adam_mean_losses_val)
    
    adam_index_train = np.argmin(adam_mean_losses_train)
    adam_index_val = np.argmin(adam_mean_losses_val)
    
    convergence_indices['Adam Gradient Descent'] = (adam_index_train, adam_index_val)
    
    #Compute mean and stdev of each plot
    for i, method in enumerate(assigned_methods):
        
        if method != 'Adam Gradient Descent':
            
            #Losses corresponding to current method
            current_losses_train = losses_train[i]
            current_losses_val = losses_val[i]
            
            #Compute mean losses
            mean_losses_train = np.mean(current_losses_train,axis=0)
            mean_losses_val = np.mean(current_losses_val, axis=0)
            
            index_train = np.argmax(mean_losses_train < adam_min_mean_losses_train)
            if index_train == 0:
                index_train = len(mean_losses_train)-1
            index_val = np.argmax(mean_losses_val < adam_min_mean_losses_val)
            if index_val == 0:
                index_val = len(mean_losses_val) -1
            
            convergence_indices[method] = (index_train, index_val)
            
    
    return convergence_indices


# =============================================================================
# Compute gradient norm to get indication of lambda value
# =============================================================================

def gradient_norms(X, y, mapping, CP_rank, batch_size, key):
    
    """
    Method that can be used to get an indication of the initial gradient value of the two terms of the loss function.
    
    """
    
    D = X.shape[1]
    M = mapping.order()
    lambda_reg = 1

    # Validation split
    key, subkey = random.split(key)
    random_state = int(subkey[0])
    val_split = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state = random_state)

    mapping_function = jit(mapping.batch_feature_map)

    b = 0
    i = 0

    X_s, y_s = shuffle(X_train, y_train, random_state=i)

    #Current batch
    X_batch = X_s[b*batch_size:(b+1)*batch_size, :]
    y_batch = y_s[b*batch_size:(b+1)*batch_size]
        
    # Perform feature transformation
    Zs = mapping_function(X_batch)


    weights = cpd_functions.random_normal_cpd(D, M, CP_rank, key)


    mse_gradient, _  = cpd_weight_update.inner_product_gradient(weights, Zs, y_batch, batch_size)

    @jit
    def regularization_gradient(weights, lambda_reg):
        # Regularization term
        # Compute full gamma
        weights_T = jnp.swapaxes(weights, 1, 2)
        gamma = jnp.matmul(weights_T, weights)
        gamma_full = jnp.prod(gamma, axis=0)
        
        # Compute gradient of regularization term
        gradient = 2*lambda_reg*jnp.matmul(weights, gamma_full / gamma)
        
        return gradient

    reg_gradient = regularization_gradient(weights, lambda_reg)

    print(f'Min reg gradient: {np.min(np.abs(reg_gradient))}')
    print(f'Min mse gradient: {np.min(np.abs(mse_gradient))}')
    
    print(f'Mean mse gradient: {np.mean(np.abs(mse_gradient))}')
    print(f'Median mse gradient: {np.median(np.abs(mse_gradient))}')

    mse_gradient_norms = np.linalg.norm(mse_gradient, axis=(1,2))
    mse_gradient_mean_norm = np.mean(mse_gradient_norms)

    reg_gradient_norms = np.linalg.norm(reg_gradient, axis=(1,2))
    reg_gradient_mean_norm = np.mean(reg_gradient_norms)
    
    return mse_gradient_mean_norm, reg_gradient_mean_norm