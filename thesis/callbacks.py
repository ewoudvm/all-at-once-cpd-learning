#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:26:49 2022

@author: Ewoud
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp


# Custom moduldes
import cpd_training
import cpd_weight_update
import cpd_functions
import cpd_linesearch
# =============================================================================
# Callback class that is callable
# =============================================================================
### Callback

class CallBack(ABC):
    """
    An abstract class that fixes the structure of a Callback method
    This method is a class that is callable and therefore works as a function.
    As a result, it can be jitted after it has be initiated. 
    
    A callback can be used to sotre any intermediate results during training.
    ...
    
    Methods
    -------
    __init__():
        Initialize callback by initializing attributes for storing intermediate results.
    __call__():
        Method to perform a weight update.
    set_method():
        Set method of callback. 
    """
    
    @abstractmethod
    def __init__(self, D, M, R, max_iter):
        pass
    
    @abstractmethod
    def __call__(self, weights, Zs, y, lambda_reg, iteration):
        pass
        
    @abstractmethod
    def set_method(self, method:str):
        self.method = method
    
class CallBackGradient(CallBack):
    
    """
    A callback that stores the intermediate gradients
    
    """
    
    def __init__(self, D, M, R, max_iter):
        self.D = D
        self.M = M
        self.R = R
        
        self.gradients = jnp.zeros((D, M, R, max_iter))
        
        self.num_calls = 0

    def __call__(self, weights, Zs, y, lambda_reg, iteration):
        
        # Compute gradient
        gradient, _, _ = cpd_weight_update.batch_gradient(weights, Zs, y, lambda_reg)
        
        # Store gradient
        self.gradients = self.gradients.at[:, :, :, self.num_calls].set(gradient)
        
        # Update num calls
        self.num_calls = self.num_calls + 1
        
    def set_method(self, method):
        super().set_method(method)
        

class CallBackGradientPrinted(CallBack):
    
    """
    A callback that stores the intermediate gradients and prints the largest
    and smallest absolute value of each gradient.
    
    """
    
    def __init__(self, D, M, R, max_iter):
        self.D = D
        self.M = M
        self.R = R
        
        self.gradients = jnp.zeros((D, M, R, max_iter))
        
        self.num_calls = 0

    def __call__(self, weights, Zs, y, lambda_reg, iteration):
        
        # Compute gradient
        gradient, _, _ = cpd_weight_update.batch_gradient(weights, Zs, y, lambda_reg)
        
        # Store gradient
        self.gradients = self.gradients.at[:, :, :, self.num_calls].set(gradient)
        
        print(f'Largest: {jnp.max(jnp.abs(gradient))}')
        print(f'Smallest: {jnp.min(jnp.abs(gradient))}')
        
        # Update num calls
        self.num_calls = self.num_calls + 1
        
    def set_method(self, method):
        super().set_method(method)

        

class CallBackCPDNorm(CallBack):
    
    """
    A callback that stores the intermediate CPD norms.
    
    """
    
    def __init__(self, D, M, R, max_iter):
       self.D = D
       self.M = M
       self.R = R
       
       self.norms = jnp.zeros(max_iter)
       
       self.num_calls = 0
       
    def __call__(self, weights):
       
       # Compute norm
       norm = cpd_functions.cpd_norm(weights)
       
       # Store norm
       self.norms = self.norms.at[self.num_calls].set(norm)
       
       # Update num calls
       self.num_calls = self.num_calls + 1
   
    def set_method(self, method):
       super().set_method(method)
        
        
class CallBackAdamStepSize(CallBack):
    
    """
    A callback that stores the intermediate update step computed by the Adam method.
    Furthermore, it stores the step size, where the step size is defined as the update step divided by the gradient.
    
    """
    
    def __init__(self, D, M, R, max_iter):
        self.D = D
        self.M = M
        self.R = R
        
        self.m = jnp.zeros((D, M ,R))
        self.v = jnp.zeros((D, M, R))
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        
        self.update_steps = jnp.zeros((D, M, R, max_iter))
        self.step_size = jnp.zeros((D, M, R, max_iter))
        
        self.num_calls = 0
        
    def __call__(self, weights, Zs, y, lambda_reg, iteration):
        
        # Compute gradient and losses
        gradient, mse, loss_reg = cpd_weight_update.batch_gradient(weights, Zs, y, lambda_reg)    
        
        # Perform Adam update
        m_t = self.beta_1*self.m + (1.0-self.beta_1)*gradient
        v_t = self.beta_2*self.v + (1.0-self.beta_2)*gradient*gradient
        
        mhat = m_t/(1.0-self.beta_1**(iteration+1))
        vhat = v_t/(1.0-self.beta_2**(iteration+1))
        
        update_step = mhat/(jnp.sqrt(vhat)+self.epsilon)
        
        # Store results
        self.m = m_t
        self.v = v_t
        
        self.update_steps = self.update_steps.at[:, :, :, self.num_calls].set(update_step)
        self.step_size = self.step_size.at[:, : , :, self.num_calls].set(update_step/gradient)
        
        self.num_calls = self.num_calls + 1
        
    def set_method(self, method):
        super().set_method(method)
        

class CallBackLineSearchStepSize(CallBack):
    
    """
    A callback that stores the optimal step size that is computed by the Line Search method.
    
    """
    
    def __init__(self, D, M, R, max_iter):
        self.D = D
        self.M = M
        self.R = R
        
        self.step_size = jnp.zeros(max_iter)
        
        self.num_calls = 0
        
    def __call__(self, weights, Zs, y, lambda_reg, iteration):
        
        # Compute gradient and losses
        gradient, mse, loss_reg = cpd_weight_update.batch_gradient(weights, Zs, y, lambda_reg)
        current_loss = mse + loss_reg
        
        # Compute coefficients of line search function
        # The line search function is a polynomial function in alpha
        alpha_coeffs = cpd_linesearch.h_alpha_grad_coeffs_batch(weights, -gradient, Zs, y, lambda_reg) 

        # Take derivatives of coefficients of the polynomial
        alpha_coeffs_deriv = cpd_linesearch.derivative_coefficients_of_polynomial(alpha_coeffs)
        
        # Compute roots to find optimal alphas
        sols = jnp.roots(jnp.flip(alpha_coeffs_deriv))
        
        # Save best solution
        loss_best = current_loss
        best_alpha = 0
        
        
        for alpha in sols:
            # complex roots not possible
            if not jnp.iscomplex(alpha):
                alpha = jnp.real(alpha)
                new_loss = jnp.polyval(jnp.flip(alpha_coeffs), alpha)
                if new_loss <= loss_best:
                    loss_best = new_loss
                    best_alpha = alpha


        self.step_size = self.step_size.at[self.num_calls].set(best_alpha)
        self.num_calls = self.num_calls + 1
    
    def set_method(self, method):
        super().set_method(method)  
    