#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:15:28 2022

This file contains many different functions that can be used to plot and print the results of experiment.
No documentation was added to the functions, since they are purely used to visualize results.

@author: Ewoud
"""

import matplotlib.pyplot as plt
from matplotlib import cm, style
import matplotlib
import pandas as pd
import numpy as np
import re

import cpd_training
from callbacks import CallBack
import experiment_helper

# Set style to colorblind friendly style
style.use('seaborn-colorblind')

# =============================================================================
# Color specifications
# =============================================================================

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
ALS_COLOR = cycle[0]
SGD_COLOR = cycle[1]
SGD_AD_COLOR = cycle[4]
LS_COLOR = cycle[2]
ADAM_COLOR = cycle[3]

# Set font size
plt.rcParams['font.size'] = '14'
# plt.rcParams['font.family'] = 'serif'
# matplotlib.rc('font', serif='Times New Roman') 
# =============================================================================
# Help functions
# =============================================================================

# These methods are used to ensure that each plot has the same name and color for a method.

def get_label(method: str):
    
    if method == 'Steepest Gradient Descent':
        label = 'SteGD'
    elif method == 'AD Steepest Gradient Descent':
        label = 'AD SteGD'
    elif method == 'Linesearch Gradient Descent':
        label = 'Line search GD'
    elif method == 'Linesearch Gradient Descent JAX':
        label = 'Line search GD JAX'
    elif method == 'Adam Gradient Descent':
        label = 'Adam GD' 
    else:
        label = method
        
    return label

def get_color(method: str):
    
    if method == 'Steepest Gradient Descent':
        color = SGD_COLOR
    elif method == 'AD Steepest Gradient Descent':
        color = SGD_AD_COLOR
    elif method == 'Linesearch Gradient Descent':
        color = LS_COLOR
    elif method == 'Linesearch Gradient Descent JAX':
        color = LS_COLOR
    elif method == 'Adam Gradient Descent':
        color = ADAM_COLOR
    elif method == 'ALS':
        color = ALS_COLOR
    else:
        color = cycle[5]
        
    return color

# =============================================================================
# Plotting functions
# =============================================================================
#### Plotting functions

# These methods

def plot_training_and_validation_loss(results, colors, line_styles,log_y_scale=False, figsize=[6.4, 4.8], legend=True):
    
    # Plotting loss functions
        # Create plots
    fig_loss, ax_loss = plt.subplots(figsize=figsize , dpi=200)
    fig_loss_val, ax_loss_val = plt.subplots(figsize=figsize, dpi=200)

    # Plot all results
    for i, result in enumerate(results):

        method      = result.method
        loss_train  = result.loss_train
        loss_val    = result.loss_val
        

        label = get_label(method)
        if result.optimization_details:

            for key, value in result.optimization_details.items():
                if key != 'final_mse_train_det' and key != 'final_loss_train_det' and key != 'Time':
                    label = label + ', ' + key + ': ' + str(value)
                

        ax_loss.plot(loss_train, label=label, c=colors[i], ls=line_styles[i], linewidth=2.5)
        ax_loss_val.plot(loss_val, label=label, c=colors[i], ls=line_styles[i], linewidth=2.5)

    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    if log_y_scale:
        ax_loss.set_yscale('log')
    ax_loss.set_title('Training loss')
    if legend:
        ax_loss.legend()
    ax_loss.yaxis.grid(True,alpha=0.7, which='both')

    ax_loss_val.set_xlabel('Iteration')
    ax_loss_val.set_ylabel('Loss')
    if log_y_scale:
        ax_loss_val.set_yscale('log')
    ax_loss_val.set_title('Validation loss')
    if legend:
        ax_loss_val.legend()
    ax_loss_val.yaxis.grid(True,alpha=0.7, which='both')
    
    fig_loss.tight_layout()
    fig_loss_val.tight_layout()
    
    

def plot_training_and_validation_loss_and_mse(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None):
        
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    # Create plots
    fig_mse, ax_mse = plt.subplots(dpi=200)
    fig_loss, ax_loss = plt.subplots(dpi=200)
    fig_mse_val, ax_mse_val = plt.subplots(dpi=200)
    fig_loss_val, ax_loss_val = plt.subplots(dpi=200)
    
    # Plot all results
    for i, result in enumerate(results):
    
        method      = result.method
        mse_train   = result.mse_train
        loss_train  = result.loss_train
        mse_val     = result.mse_val
        loss_val    = result.loss_val
        
        end_index_train = len(mse_train)
        end_index_val = len(mse_val)
        
        if provided_end_index:
            if end_index_train > end_index:
                end_index_train = end_index
            if end_index_val > end_index:
                end_index_val = end_index
       
        x_indx_train = np.arange(start_index, end_index_train)
        x_indx_val = np.arange(start_index, end_index_val)
        ax_mse.plot(x_indx_train, mse_train[start_index:end_index_train], label=method, linewidth=2)
        ax_loss.plot(x_indx_train, loss_train[start_index:end_index_train], label=method, linewidth=2)
        ax_mse_val.plot(x_indx_val, mse_val[start_index:end_index_train], label=method, linewidth=2)
        ax_loss_val.plot(x_indx_val, loss_val[start_index:end_index_train], label=method, linewidth=2)
        
        
    # Create pretty figures
    ax_mse.set_xlabel('Iteration')
    ax_mse.set_ylabel('MSE')
    ax_mse.set_title('MSE on training data')
    ax_mse.legend()
    # ax_mse.set_ylim(0.5, 1.05)
    ax_mse.set_yscale('log')
    
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training loss')
    ax_loss.legend()
    # ax_loss.set_ylim(0.5, 1.05)
    ax_loss.set_yscale('log')
    
    ax_mse_val.set_xlabel('Iteration')
    ax_mse_val.set_ylabel('MSE')
    ax_mse_val.set_title('MSE on validation data')
    ax_mse_val.legend()
    # ax_mse_val.set_ylim(0.5, 1.05)
    ax_mse_val.set_yscale('log')
    
    ax_loss_val.set_xlabel('Iteration')
    ax_loss_val.set_ylabel('Loss')
    ax_loss_val.set_title('Validation loss')
    ax_loss_val.legend()
    # ax_loss_val.set_ylim(0.5, 1.05)
    ax_loss_val.set_yscale('log')
    
def plot_decision_boundaries(X_p, y_p, results: list[cpd_training.OptimizationResult], colors, line_styles, resolution = 100, legend=True):
        
    
    # Setup data for boundary plot
    x1 = np.linspace(0, 1, resolution)
    x2 = np.linspace(0, 1, resolution)

    x1v, x2v = np.meshgrid(x1, x2)
    Xplot = np.array([x1v.ravel(), x2v.ravel()]).T

    fig, ax = plt.subplots(dpi=200)

    # Create proxy for legend
    proxy = []
    names = []


    for i, result in enumerate(results):
        
        method      = result.method
        names.append(method)
        weights     = result.weights
        mapping     = result.mapping

        Zsplot = mapping.batch_feature_map(Xplot)

        y_plot_pred = cpd_training.prediction(weights, Zsplot)
        y_plot_pred = np.reshape(y_plot_pred, x1v.shape)

        # Create contour
        cont = ax.contour(x1v, x2v, y_plot_pred, [0], colors = colors[i], linestyles=line_styles[i], linewidths=2.5)

        # Create legend
        proxy.append(plt.Line2D((0,1),(0.5,0.5), color = cont.collections[0].get_edgecolor()[0], linestyle = cont.collections[0].get_linestyle()[0]))


    # Plot data itself
    ax.scatter(X_p[y_p==-1,0], X_p[y_p==-1,1], c='b', alpha=0.2)
    ax.scatter(X_p[y_p==1,0], X_p[y_p==1,1], c='r', alpha=0.2)
    
    # Create pretty data
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision boundaries')
    if legend:
        ax.legend(proxy, names, loc='upper right')
    
    
def plot_gradient_norms_per_factor(callbacks: list[CallBack]):
    
    
    fig, ax = plt.subplots()
    
    for cb in callbacks:
        norms = np.zeros((cb.num_calls, cb.D))

        for d in range(cb.D):
            for i in range(cb.num_calls):
                norms[i, d] = np.linalg.norm(cb.gradients[d, :, :, i])
            
            ax.plot(norms[:, d], label=f'Factor {d}, {cb.method}')
                

    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Norm of gradient')
    ax.set_title('Norm of gradient during training')
    ax.legend()
    plt.tight_layout()
    
    
def plot_multiple_experiments(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None, plot_training = False, plot_mse_val=False):
    
    
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    # Create plots
    if plot_training:
        fig_loss, ax_loss = plt.subplots()      
        
    if plot_mse_val:
        fig_mse_val, ax_mse_val = plt.subplots()
        
    fig_loss_val, ax_loss_val = plt.subplots()
    
    # Get colormaps
    cmap_green = cm.get_cmap('Greens')
    cmap_red = cm.get_cmap('Reds')
    cmap_purple = cm.get_cmap('Purples')
    cmap_blue = cm.get_cmap('Blues')
    cmap_grey = cm.get_cmap('Greys')
    cmap_copper = cm.get_cmap('copper')
    cmap_summer = cm.get_cmap('summer')
    cmap_cool = cm.get_cmap('cool')
    cmap_winter = cm.get_cmap('winter')
    cmap_autumn = cm.get_cmap('autumn')
    cmap_wistia = cm.get_cmap('Wistia')
    cmap_twilight = cm.get_cmap('twilight')
    
    cmaps = [cmap_green, cmap_red, cmap_purple, cmap_blue, cmap_grey, 
             cmap_copper, cmap_summer, cmap_cool, cmap_winter, cmap_autumn,
             cmap_wistia, cmap_twilight]
    
    # Assign colormap to each method
    current_cmap = 0
    assigned_cmaps = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            assigned_cmaps.append(cmaps[current_cmap])
            current_cmap += 1
            assigned_methods.append(method)
    
    # Create proxy for legend
    proxy = []
    for cmap in assigned_cmaps:
        proxy.append(plt.Line2D((0,1),(0.5,0.5), color = cmap(0.75)))
    
    # Plot all results
    method_calls = np.zeros(len(assigned_methods))
    for result in results:
    
        # Load result parameters
        method      = result.method
        mse_train   = result.mse_train
        loss_train  = result.loss_train
        mse_val     = result.mse_val
        loss_val    = result.loss_val
        
        # Compute end indices
        if plot_training:
            end_index_train = len(mse_train)
            
        end_index_val = len(loss_val)
        
        if provided_end_index:
            if plot_training:
                if end_index_train > end_index:
                    end_index_train = end_index
            if end_index_val > end_index:
                end_index_val = end_index
       
        # Get color
        method_index = assigned_methods.index(method)
        cmap = assigned_cmaps[method_index]
        color = cmap(0.25+0.1*method_calls[method_index])
        method_calls[method_index] += 1
       
        if plot_training:  
            x_indx_train = np.arange(start_index, end_index_train)
            ax_loss.plot(x_indx_train, loss_train[start_index:end_index_train], c=color)
            
        if plot_mse_val:
            x_indx_train = np.arange(start_index, end_index_train)
            ax_mse_val.plot(x_indx_train, mse_val[start_index:end_index_train], c=color)

        x_indx_val = np.arange(start_index, end_index_val)
        ax_loss_val.plot(x_indx_val, loss_val[start_index:end_index_val], c=color)
        
        
    # Create pretty figures
    if plot_training:
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss on training data')
        ax_loss.legend(proxy, assigned_methods, loc='upper right')
    
    if plot_mse_val:
        ax_mse_val.set_xlabel('Iteration')
        ax_mse_val.set_ylabel('Loss')
        ax_mse_val.set_title('MSE on validation data')
        ax_mse_val.legend(proxy, assigned_methods, loc='upper right')
    
    ax_loss_val.set_xlabel('Iteration')
    ax_loss_val.set_ylabel('Loss')
    ax_loss_val.set_title('Loss on validation data')
    ax_loss_val.legend(proxy, assigned_methods, loc = 'upper right')
    
    
def plot_validation_multiple_experiments_averaged(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None):
    
    
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    # Create plots
    fig_loss_val, ax_loss_val = plt.subplots()
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_val)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_val)
    
    #Compute mean and stdev of each plot
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        # Compute start and end indices
        end_index_val = len(mean_losses)
        
        if provided_end_index:
            if end_index_val > end_index:
                end_index_val = end_index
                
        x_indx_val = np.arange(start_index, end_index_val)
        
        # Plot mean losses
        ax_loss_val.plot(x_indx_val, mean_losses[start_index:end_index_val], c=get_color(method), label=get_label(method))
        
        #Plot std 
        #TODO: For now use fill between: could also be error bars
        std_losses = np.std(current_losses, axis=0)
        ax_loss_val.fill_between(x_indx_val, (mean_losses-std_losses)[start_index:end_index], (mean_losses+std_losses)[start_index:end_index], alpha=0.1, color=get_color(method))
    
    ax_loss_val.set_xlabel('Iteration')
    ax_loss_val.set_ylabel('Loss')
    ax_loss_val.set_title('Average validation loss')
    ax_loss_val.legend()
    
    
def plot_training_multiple_experiments_averaged(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None):
    
    
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    # Create plots
    fig_loss, ax_loss = plt.subplots(dpi=200)
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_train)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_train)
    
    #Compute mean and stdev of each plot
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        if not provided_end_index:
            end_index = len(mean_losses)
        x_indx = np.arange(start_index, end_index)
        
        # Plot mean losses
        ax_loss.plot(x_indx, mean_losses[start_index:end_index], c=get_color(method), label=get_label(method),linewidth=2.5)
        
        #Plot std 
        #TODO: For now use fill between: could also be error bars
        std_losses = np.std(current_losses, axis=0)
        ax_loss.fill_between(x_indx, (mean_losses-std_losses)[start_index:end_index], (mean_losses+std_losses)[start_index:end_index], alpha=0.3, color=get_color(method))
    
    ax_loss.yaxis.grid(True, alpha=0.7, which='both')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_yscale('log')
    ax_loss.set_title('Training loss')
    ax_loss.legend()
    
def plot_training_multiple_experiments_averaged_with_convergence(results: list[cpd_training.OptimizationResult], convergence_index : int):
    
   
    
    # Create plots
    # fig_loss, ax_loss = plt.subplots()
    # fig_loss, ax_loss = plt.subplots(figsize = [8, 4.8], dpi=200)
    fig_loss, ax_loss = plt.subplots(dpi=200)
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_train)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_train)
    
    #Compute mean and stdev of each plot
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        end_index = len(mean_losses)
        start_index = 0
        x_indx = np.arange(start_index, end_index)
        
        # Plot mean losses
        ax_loss.plot(x_indx, mean_losses[start_index:end_index], c=get_color(method), label=get_label(method), linewidth=2.5)
        
        #Plot std 
        std_losses = np.std(current_losses, axis=0)
        ax_loss.fill_between(x_indx, (mean_losses-std_losses)[start_index:end_index], (mean_losses+std_losses)[start_index:end_index], alpha=0.3, color=get_color(method))
    
    # Plot convergence line
    # ax_loss.axvline(convergence_index, c ='k', ls='--', alpha = 0.8, label='Convergence')
    colors = [ADAM_COLOR, ALS_COLOR]
    ax_loss.vlines(convergence_index, 0, 1, transform=ax_loss.get_xaxis_transform(), colors=colors, linestyles='dashed', linewidth=2, label='Convergence')
    
    
    
    ax_loss.yaxis.grid(True, alpha=0.7, which='both')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_yscale('log')
    ax_loss.set_title('Training loss')
    handles, labels = ax_loss.get_legend_handles_labels()
    convergence_handle = plt.Line2D((0,1),(0.5, 0.5), color = 'k', ls = 'dashed', linewidth = 2)
    handles_copy = [handles[0], handles[1], convergence_handle]
    # handles_copy = [handles[0], handles[1]]
    ax_loss.legend(handles_copy, labels)

def plot_combo_training_multiple_experiments_averaged(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None):
    
    
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    # Create plots
    fig_loss, ax_loss = plt.subplots(dpi=200)
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    combo_losses = [[],[]]
    for result in results:
        if type(result) is cpd_training.CombinationExperimentOptimizationResult:
            first_training_loss = result.first_result.loss_train
            second_training_loss = result.second_result.loss_train
            combo_losses[0].append(first_training_loss)
            combo_losses[1].append(second_training_loss)
        else:
            method      = result.method
            if method not in assigned_methods:
                # Add empty list for this method
                losses.append([])
                assigned_methods.append(method)
                
                # New method is last
                losses[-1].append(result.loss_train)
                
            else:
                method_index = assigned_methods.index(method)
                losses[method_index].append(result.loss_train)
    
    #Compute mean and stdev of each plot
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        if not provided_end_index:
            end_index = len(mean_losses)
        x_indx = np.arange(start_index, end_index)
        
        # Plot mean losses
        ax_loss.plot(x_indx, mean_losses[start_index:end_index], c=get_color(method), label=get_label(method), linewidth=2.5)
        
        #Plot std 
        std_losses = np.std(current_losses, axis=0)
        ax_loss.fill_between(x_indx, (mean_losses-std_losses)[start_index:end_index], (mean_losses+std_losses)[start_index:end_index], alpha=0.3, color=get_color(method))
    
    #Combo losses
    first_method = 'Adam Gradient Descent'
    second_method = 'ALS'
    als_combo_color = '#827100'
    # als_combo_color = '#0d673c'
    
    mean_first_losses = np.mean(combo_losses[0],axis=0)
    mean_second_losses = np.mean(combo_losses[1], axis=0)
    
    std_first_losses = np.std(combo_losses[0],axis=0)
    std_second_losses = np.std(combo_losses[1],axis=0)
    
    end_first_method = len(mean_first_losses)
    x_indx_first = np.arange(0, end_first_method)
    x_indx_second = np.arange(end_first_method, end_first_method+len(mean_second_losses))
    # ax_loss.plot(x_indx_first, mean_first_losses, c=get_color(first_method), label=get_label(first_method), ls=':')
    ax_loss.plot(x_indx_second, mean_second_losses, c=als_combo_color, label=get_label(second_method)+' combo', ls='--', linewidth=2.5)
    ax_loss.fill_between(x_indx_second, mean_second_losses-std_second_losses, mean_second_losses+std_second_losses, alpha=0.3, color = als_combo_color)

    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_yscale('log')
    ax_loss.set_title('Training loss')
    ax_loss.legend()
    ax_loss.yaxis.grid(True, alpha=0.7, which='both')
    

def plot_validation_multiple_experiments_averaged_with_subplots(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None):
    
    
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_val)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_val)
    
    # Count number of methods
    # Here a method is the string before a comma, ie the solver method
    assigned_solver_methods = []
    for i, method in enumerate(assigned_methods):
        
        solver_method = re.split(',', method)[0]
        if solver_method not in assigned_solver_methods:
            assigned_solver_methods.append(solver_method)
            
    num_solver_methods = len(assigned_solver_methods)
    if num_solver_methods % 2 == 0:
        n_cols = 2
    else:
        n_cols = 3
    n_rows = int(np.ceil(num_solver_methods/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10,5))
    
    #Compute mean and stdev of each method and plot
    for i, method in enumerate(assigned_methods):
        
        #Check solver method
        solver_method = re.split(', ', method)[0]
        details = re.split(', ', method)[1]
        
        #Get correct subplot
        solver_method_index = assigned_solver_methods.index(solver_method)
        ax = axes.ravel()[solver_method_index]
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        # Compute start and end indices
        end_index_val = len(mean_losses)
        
        if provided_end_index:
            if end_index_val > end_index:
                end_index_val = end_index
                
        x_indx_val = np.arange(start_index, end_index_val)
        
        
        # Plot mean losses
        ax.plot(x_indx_val, mean_losses[start_index:end_index_val], label=details)
        
        #Plot std 
        std_losses = np.std(current_losses, axis=0)
        ax.fill_between(x_indx_val, (mean_losses-std_losses)[start_index:end_index], (mean_losses+std_losses)[start_index:end_index], alpha=0.1)
    
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(solver_method)

    # Plot legend in upper right subplot
    axes.ravel()[n_cols-1].legend()
    # handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # fig.legend(handles,labels, loc=(0.87, 0.75))
    fig.suptitle('Average validation loss for different solver methods')
    fig.tight_layout()
    
def plot_training_multiple_experiments_averaged_with_subplots(results: list[cpd_training.OptimizationResult], start_index = 0, end_index = None):
    
    
    # Check arguments
    if end_index == None:
        provided_end_index = False
    else:
        provided_end_index = True
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_train)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_train)
    
    # Count number of methods
    # Here a method is the string before a comma, ie the solver method
    assigned_solver_methods = []
    for i, method in enumerate(assigned_methods):
        
        solver_method = re.split(',', method)[0]
        if solver_method not in assigned_solver_methods:
            assigned_solver_methods.append(solver_method)
            
    num_solver_methods = len(assigned_solver_methods)
    if num_solver_methods % 2 == 0:
        n_cols = 2
    else:
        n_cols = 3
    n_rows = int(np.ceil(num_solver_methods/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10,5))
    
    #Compute mean and stdev of each method and plot
    for i, method in enumerate(assigned_methods):
        
        #Check solver method
        solver_method = re.split(', ', method)[0]
        details = re.split(', ', method)[1]
        
        #Get correct subplot
        solver_method_index = assigned_solver_methods.index(solver_method)
        ax = axes.ravel()[solver_method_index]
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        # Compute start and end indices
        end_index_val = len(mean_losses)
        
        if provided_end_index:
            if end_index_val > end_index:
                end_index_val = end_index
                
        x_indx_val = np.arange(start_index, end_index_val)
        
        
        # Plot mean losses
        ax.plot(x_indx_val, mean_losses[start_index:end_index_val], label=details)
        
        #Plot std 
        std_losses = np.std(current_losses, axis=0)
        ax.fill_between(x_indx_val, (mean_losses-std_losses)[start_index:end_index], (mean_losses+std_losses)[start_index:end_index], alpha=0.1)
    
        ax.set_xlabel('Weight update')
        ax.set_ylabel('Loss')
        ax.set_title(get_label(solver_method))

    # Plot legend in upper right subplot
    axes.ravel()[n_cols-1].legend()
    # handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # fig.legend(handles,labels, loc=(0.87, 0.75))
    fig.suptitle('Average training loss for different solver methods')
    fig.tight_layout()
    

def plot_timing_experiment(experiment: experiment_helper.TimingExperiment):
        
    #Create nested list of times where each method has seperate list of times
    times = []
    assigned_methods = []
    for i, result in enumerate(experiment.results):
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            times.append([])
            assigned_methods.append(method)
            
            # New method is last
            times[-1].append(experiment.times[i])
            
        else:
            method_index = assigned_methods.index(method)
            times[method_index].append(experiment.times[i])
    
    # Specific for batch size time
    color_adam = 'lightblue'
    color_als = 'lightgreen'
    labels = []
    colors = []
        
    #Largest batch size is full data set
    largest_batch_size = 0
    
    for method in assigned_methods:
        # Check wheter Adam and save batch size
        if re.match('Adam', method) is not None:
            batch_size = re.split(': ', method)[1]
            labels.append(batch_size)
            colors.append(color_adam)
            
            if int(batch_size) > largest_batch_size:
                largest_batch_size = int(batch_size)
        else:
            labels.append(method)
            colors.append(color_als)
    
    # Plot boxplot
    fig, ax = plt.subplots()
    bplot = ax.boxplot(times, patch_artist=True)
    
    #Assign colors to boxplot
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel('Method')
    ax.set_ylabel('Time (s)')
    ax.set_title('Execution time for one pass over the data for different methods')
    fig.tight_layout()
    
    # Adjust times for time per update
    adjusted_times = []
    for i, method in enumerate(assigned_methods):
        # Check wheter Adam and save batch size
        if re.match('Adam', method) is not None:
            batch_size = re.split(': ', method)[1]
            number_of_updates = int(largest_batch_size/int(batch_size))
            adjusted_times.append(np.array(times[i])/number_of_updates)
        else:
            adjusted_times.append(np.array(times[i]))
    
    # Plot boxplot
    fig, ax = plt.subplots()
    bplot = ax.boxplot(adjusted_times, patch_artist=True)
    
    #Assign colors to boxplot
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel('Method')
    ax.set_ylabel('Time (s)')
    ax.set_title('Execution time for one update for different methods')
    fig.tight_layout()
    
def plot_method_combination(experiment: experiment_helper.MethodCombiationExperiment, plot_training=False):
        
    
    # Create plots
    if plot_training:
        fig_loss, ax_loss = plt.subplots()        
        
    fig_loss_val, ax_loss_val = plt.subplots()
    
    # Get colormaps
    cmap_green = cm.get_cmap('Greens')
    cmap_red = cm.get_cmap('Reds')
    cmap_purple = cm.get_cmap('Purples')
    cmap_blue = cm.get_cmap('Blues')
    cmap_grey = cm.get_cmap('Greys')
    cmap_copper = cm.get_cmap('copper')
    cmap_summer = cm.get_cmap('summer')
    cmap_cool = cm.get_cmap('cool')
    cmap_winter = cm.get_cmap('winter')
    cmap_autumn = cm.get_cmap('autumn')
    cmap_wistia = cm.get_cmap('Wistia')
    cmap_twilight = cm.get_cmap('twilight')
    
    cmaps = [cmap_green, cmap_red, cmap_purple, cmap_blue, cmap_grey, 
             cmap_copper, cmap_summer, cmap_cool, cmap_winter, cmap_autumn,
             cmap_wistia, cmap_twilight]
    
    
    
    # Assign colormap to each method
    current_cmap = 0
    assigned_cmaps = []
    assigned_methods = []
    
    # Create proxy for legend

    
    # Plot all results
    start_index_train = 0
    start_index_val = 0
    for results in experiment.results_per_method:
        
        method  = results[0].method
        if method not in assigned_methods:
            assigned_cmaps.append(cmaps[current_cmap])
            current_cmap += 1
            assigned_methods.append(method)

        for i, result in enumerate(results):
        
            # Load result parameters
            method      = result.method
            mse_train   = result.mse_train
            loss_train  = result.loss_train
            mse_val     = result.mse_val
            loss_val    = result.loss_val
            
            # Compute indices
            if plot_training:
                end_index_train = start_index_train + len(mse_train)
                
            end_index_val = start_index_val + len(loss_val)
            
           
            # Get color
            method_index = assigned_methods.index(method)
            cmap = assigned_cmaps[method_index]
            color = cmap(0.25+0.05*i)
           
            if plot_training:  
                x_indx_train = np.arange(start_index_train, end_index_train)
                ax_loss.plot(x_indx_train, loss_train, c=color)
    
            x_indx_val = np.arange(start_index_val, end_index_val)
            ax_loss_val.plot(x_indx_val, loss_val, c=color)
        
        #Update start index
        start_index_train += len(results[0].mse_train)
        start_index_val += len(results[0].loss_val)
        
    # Create pretty figures
    proxy = []
    for cmap in assigned_cmaps:
        proxy.append(plt.Line2D((0,1),(0.5,0.5), color = cmap(0.75)))
        
    if plot_training:
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss on training data')
        ax_loss.legend(proxy, assigned_methods, loc='upper right')
    
    
    ax_loss_val.set_xlabel('Iteration')
    ax_loss_val.set_ylabel('Loss')
    ax_loss_val.set_title('Loss on validation data')
    ax_loss_val.legend(proxy, assigned_methods, loc = 'upper right')
    
    
# =============================================================================
# Table generator
# =============================================================================
#### Table generator

def table_mean_final_validation_losses(results: list[cpd_training.OptimizationResult], indices = [-1]):
        
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_val[indices])
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_val[indices])

    

    #Convert to losses
    losses = np.array(losses)
    mean_losses = np.mean(losses, axis=1)
    std_losses = np.std(losses, axis=1)
    
    # Create columns and data
    columns = []
    data = np.zeros((len(assigned_methods), len(indices)*2))
    for i, index in enumerate(indices):
        columns = columns + [f'i = {index}, mean', 'std']
        
        data[:,i*2] = mean_losses[:,i]
        data[:,i*2+1] = std_losses[:,i]
        
    

    df = pd.DataFrame(data, index=assigned_methods, columns = columns)
    df = df.sort_index()
    
    return df.to_latex()
    
    
def print_final_losses(results : list[cpd_training.OptimizationResult]):
    
    for result in results:
        print(f'{result.method}:')
        print(f'Training: {result.loss_train[-1]:.3f}')
        print(f'Validation: {result.loss_val[-1]:.3f}')

def print_mean_final_validation_losses(results: list[cpd_training.OptimizationResult], index : int = None):
    
    if type(index) is not int and index is not None:
        print('Illegal index')
        index = -1
    elif index is None:
        index = -1
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_val[index])
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_val[index])
            
    
    #Convert to losses
    losses = np.array(losses)
    mean_losses = np.mean(losses, axis=1)
    std_losses = np.std(losses, axis=1)
    
    print('Validation losses')
    
    for m, method in enumerate(assigned_methods):
        
        string = f'{method : <50}' + f'{mean_losses[m]: .3f} ({std_losses[m]:.4f})'
        print(string)
        
def print_min_mean_validation_losses_and_train_losses_at_min_val(results: list[cpd_training.OptimizationResult]):
    
    
    #Create nested list of losses where each method has seperate list of losses
    val_losses = []
    train_losses = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            val_losses.append([])
            train_losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            val_losses[-1].append(result.loss_val)
            train_losses[-1].append(result.loss_train)
            
        else:
            method_index = assigned_methods.index(method)
            val_losses[method_index].append(result.loss_val)
            train_losses[method_index].append(result.loss_train)
            
    
    print('')
    print('Min validation losses:')
    #Compute mean for each method
    
    
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_val_losses = np.array(val_losses[i])
        current_train_losses = np.array(train_losses[i])
        
        #Compute mean losses
        mean_val_losses = np.mean(current_val_losses,axis=0)      
        
        #Min mean validation losses
        min_mean_val_losses = np.amin(mean_val_losses)
        min_validation_loss_index =np.argmin(mean_val_losses)
        std_min_mean_val_losses = np.std(current_val_losses[:, min_validation_loss_index])
        
        print('Validation loss')
        print(f'{method : <50}{min_mean_val_losses}  ({std_min_mean_val_losses})')
        
        #Training losses
        min_validation_loss_index = min(min_validation_loss_index, current_train_losses.shape[1]-1) 
        mean_train_losses_at_min_val = np.mean(current_train_losses[:, min_validation_loss_index])
        std_train_losses_at_min_val = np.std(current_train_losses[:, min_validation_loss_index])
        print('Training loss at min val loss')
        print(f'{method : <50}{mean_train_losses_at_min_val}  ({std_train_losses_at_min_val})')
    

        
def print_mean_final_combo_validation_losses(results: list[cpd_training.OptimizationResult], index : int = None):
    
    
    if type(index) is not int and index is not None:
        print('Illegal index')
        index = -1
    elif index is None:
        index = -1
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    combo_losses = [[],[]]
    for result in results:
        if type(result) is cpd_training.CombinationExperimentOptimizationResult:
            first_validation_loss = result.first_result.loss_val
            second_validation_loss = result.second_result.loss_val
            combo_losses[0].append(first_validation_loss[index])
            combo_losses[1].append(second_validation_loss[index])
        else:
            method      = result.method
            if method not in assigned_methods:
                # Add empty list for this method
                losses.append([])
                assigned_methods.append(method)
                
                # New method is last
                losses[-1].append(result.loss_val[index])
                
            else:
                method_index = assigned_methods.index(method)
                losses[method_index].append(result.loss_val[index])
    
    #Convert to losses
    losses = np.array(losses)
    mean_losses = np.mean(losses, axis=1)
    std_losses = np.std(losses, axis=1)
    
    print('Validation losses')
    
    for m, method in enumerate(assigned_methods):
        
        string = f'{method : <50}' + f'{mean_losses[m]}  ({std_losses[m]})'
        print(string)
        
    # Combo losses
    mean_first_losses = np.mean(combo_losses[0])
    mean_second_losses = np.mean(combo_losses[1])
    
    std_first_losses = np.std(combo_losses[0])
    std_second_losses = np.std(combo_losses[1])
    
    first_method = 'Adam Gradient Descent'
    second_method = 'ALS'
    
    print('Combo losses:')
    print(f'First: {first_method : <50}' + f'{mean_first_losses}  ({std_first_losses})')
    print(f'Second: {second_method : <50}' + f'{mean_second_losses}  ({std_second_losses})')
    

def print_mean_final_training_losses(results: list[cpd_training.OptimizationResult], index : int = None):
    
    if type(index) is not int and index is not None:
        print('Illegal index')
        index = -1
    elif index is None:
        index = -1
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_train[index])
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_train[index])
            
    
    #Convert to losses
    losses = np.array(losses)
    mean_losses = np.mean(losses, axis=1)
    std_losses = np.std(losses, axis=1)
    
    print('Training losses')
    
    for m, method in enumerate(assigned_methods):
        
        string = f'{method : <50}' + f'{mean_losses[m] : .3f} ({std_losses[m]:.4f})'
        print(string)



def print_min_mean_training_losses(results: list[cpd_training.OptimizationResult]):
    
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_train)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_train)
            
    
    print('')
    print('Min training losses:')
    #Compute mean for each method
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_losses = np.array(losses[i])
        
        print(current_losses.shape)
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)      
        
        #Min mean losses
        min_mean_losses = np.amin(mean_losses)
        min_index = np.argmin(mean_losses)
        std_min_mean_losses = np.std(current_losses[:, min_index])
    
        print(f'{method} : {min_mean_losses}  ({std_min_mean_losses})')
        

        
def print_mean_final_combo_training_losses(results: list[cpd_training.OptimizationResult], index : int = None):
        
    if type(index) is not int and index is not None:
        print('Illegal index')
        index = -1
    elif index is None:
        index = -1
    
    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    combo_losses = [[],[]]
    for result in results:
        if type(result) is cpd_training.CombinationExperimentOptimizationResult:
            first_training_loss = result.first_result.loss_train
            second_training_loss = result.second_result.loss_train
            combo_losses[0].append(first_training_loss[index])
            combo_losses[1].append(second_training_loss[index])
        else:
            method      = result.method
            if method not in assigned_methods:
                # Add empty list for this method
                losses.append([])
                assigned_methods.append(method)
                
                # New method is last
                losses[-1].append(result.loss_train[index])
                
            else:
                method_index = assigned_methods.index(method)
                losses[method_index].append(result.loss_train[index])
    
    #Convert to losses
    losses = np.array(losses)
    mean_losses = np.mean(losses, axis=1)
    std_losses = np.std(losses, axis=1)
    
    print('Training losses')
    
    for m, method in enumerate(assigned_methods):
        
        string = f'{method : <50}' + f'{mean_losses[m]}  ({std_losses[m]})'
        print(string)
        
    # Combo losses
    mean_first_losses = np.mean(combo_losses[0])
    mean_second_losses = np.mean(combo_losses[1])
    
    std_first_losses = np.std(combo_losses[0])
    std_second_losses = np.std(combo_losses[1])
    
    first_method = 'Adam Gradient Descent'
    second_method = 'ALS'
    
    print('Combo losses:')
    print(f'First: {first_method : <50}' + f'{mean_first_losses}  ({std_first_losses})')
    print(f'Second: {second_method : <50}' + f'{mean_second_losses}  ({std_second_losses})')
      
def return_run_times(results: list[cpd_training.OptimizationResult]):
    
    #Create nested list of losses where each method has seperate list of losses
    times = []
    
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            times.append([])
            assigned_methods.append(method)
            
            # New method is last
            times[-1].append(result.optimization_details['Time'])
            
        else:
            method_index = assigned_methods.index(method)
            times[method_index].append(result.optimization_details['Time'])
            
    
    #Convert to losses
    times = np.array(times)
    mean_times = np.mean(times, axis=1)
    std_times = np.std(times, axis=1)
    
    time_results = list(zip(mean_times, std_times))
    
    return dict(zip(assigned_methods, time_results))

def print_run_times(results: list[cpd_training.OptimizationResult]):
    
    
    time_results = return_run_times(results)
        
    for i, (method, times) in enumerate(time_results.items()):
        
        mean_times = times[0]
        std_times = times[1]
        print(f'{method : <50}' + f'{mean_times}  ({std_times})')

def print_hyperparameters(experiment : experiment_helper.Experiment):
    
    print('Hyperparameters')
    print(experiment.learning_parameters[0])
    print(experiment.mappings[0])
    print(f'R = {experiment.results[0].weights.shape[2]}')

def plot_and_print_mean_training_loss_convergence(results: list[cpd_training.OptimizationResult], diff_to_converge = 0.0005, window_size = 5):

    """
    Method that is used to determine the convergence.
    """

    #Create nested list of losses where each method has seperate list of losses
    losses = []
    assigned_methods = []
    for result in results:
        method      = result.method
        if method not in assigned_methods:
            # Add empty list for this method
            losses.append([])
            assigned_methods.append(method)
            
            # New method is last
            losses[-1].append(result.loss_train)
            
        else:
            method_index = assigned_methods.index(method)
            losses[method_index].append(result.loss_train)
    
           
    print('Convergence index for ' + r'$\epsilon$' + f' = {diff_to_converge}' )
    
    #Compute mean for each method
    for i, method in enumerate(assigned_methods):
        
        #Losses corresponding to current method
        current_losses = losses[i]
        
        #Compute mean losses
        mean_losses = np.mean(current_losses,axis=0)
        
        # Compute difference
        diff_mean_losses = mean_losses[:-1] - mean_losses[1:]
        
        diff_mean_losses_mean = diff_mean_losses
        
        skip_ALS = 5
        if method == 'ALS':
            convergence_index = np.argmax(diff_mean_losses_mean[skip_ALS:] < diff_to_converge) + skip_ALS
        else:
            convergence_index = np.argmax(diff_mean_losses_mean < diff_to_converge)
        # plt.plot(diff_mean_losses/mean_losses[:-1])
        plt.plot(diff_mean_losses_mean)
        
        
        #Min mean losses
        min_mean_losses = np.amin(mean_losses)
        
        diff_min_mean_losses = mean_losses - min_mean_losses
        convergence_index = np.argmax(diff_min_mean_losses < diff_to_converge)
    
        print(f'{method} : {convergence_index}')

# =============================================================================
# Inspecting experiment results
# =============================================================================
#### Inspecting experiment results


def plot_experiment_losses(file_name: str):
        
    # Load results
    experiment = experiment_helper.load_results(file_name)
    
    # Plot results
    plot_training_and_validation_loss_and_mse(experiment.results)


    