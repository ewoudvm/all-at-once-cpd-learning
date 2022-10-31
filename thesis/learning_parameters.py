#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:49:09 2022

@author: Ewoud
"""

class LearningParameters:
    
    """
    A class to store all the learning parameters in a single container. 
    
    ...
 
    Attributes
    ----------
    lambda_reg : float
        Regularization term in cost function. 
    learning_rate : float
        Learning rate that is used for updating the weights. Must be positive.
    epochs : int 
        Number of epochs to use for training. In one epoch all the data is 
        used once for training. 
    batch_size : int
        The number of samples used in one batch. With each batch the weights
        are updated.
    val_split : float
        The portion of the training data that is used for validation
        
    """
    
    def __init__(self, lambda_reg: float, learning_rate: float, epochs : int, batch_size: int, val_split: float):
        
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        
        
    def __str__(self):
        
        return f'Lambda reg: {self.lambda_reg}' + '\n' +\
              f'Learning rate: {self.learning_rate}' + '\n' +\
              f'Epochs: {self.epochs}' + '\n' +\
              f'Batch size: {self.batch_size}' + '\n' +\
              f'Val split: {self.val_split}'