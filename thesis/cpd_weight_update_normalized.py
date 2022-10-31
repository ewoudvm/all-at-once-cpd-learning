#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:34:17 2022

@author: Ewoud
"""

from abc import ABC, abstractmethod


import jax.numpy as jnp
from jax import value_and_grad, vmap

# Custom modules
import cpd_training_normalized
import cpd_linesearch_normalized

# =============================================================================
# Weight update class that are callable: Used for training
# =============================================================================
#### Weight update classes

class WeightUpdateMethod(ABC):
    """
    An abstract class that fixes the structure of any weight update.
    If a new weight update is added it must/can inherent this class to ensure
    that the correct call method is implemented.
    
    Specifically used for normalized CPD. Only the weights/factors are updated,
    so the norm vector is not.
    
    This method is a class that is callable and therefore works as a function.
    As a result, it can be jitted after it has be initiated. 
    
    The inputs of the weight update method are:
        Weights: CPD stored as DxMxR 
        mu: Array of size R containing norms of normalized CPD.
        Zs: Transformed input data stored as DxMxN tensor
        y: True output
        lambda_reg: Regularization parameter
        learning_rate: Step size
        optimizer_state: Any state that needs to be updated at each iteration
        iteration: The current iteration.
    
    When an update method does not use any of these parameters it must still be passed
    for consistency.
    
    
    The call method shoud return the following in this specific order:
        weights: new weights
        loss: loss for passed data
        optimizer_state: updated optimizer_state
    ...
    
    Methods
    -------
    __call__():
        Method to perform a weight update.
    """
    
    @abstractmethod
    def __call__(self, weights, mu, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        pass
    
        
class AD_Steepest_Gradient_Descent(WeightUpdateMethod):
    """
    A class that represents the Steepest Gradient Descent method. The update is given by
    W_{new} = W - learning_rate * gradient
    
    The gradient is computed using automatic differentiation.
    
    """
    
    def __call__(self, weights, mu, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        loss, gradient = value_and_grad(cpd_training_normalized.loss_function)(weights, mu, Zs, y, lambda_reg)
        
        new_weights = weights - learning_rate * gradient
        
        return new_weights, loss, optimizer_state


class Line_Search_Gradient_Descent(WeightUpdateMethod):
    """
    A class that represents the Line Search Descent method. The update is given by
    W_{new} = W - learning_rate * gradient
        
    The learning rate is computed using line search where the roots are evaluated in a loop.
    As a result, this function cannot be jitted.
    
    The gradient is computed using automatic differentiation.
    
    """
    
    def __call__(self, weights, mu, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        # Compute gradient and losses
        # gradient, mse, loss_reg = batch_gradient(weights, Zs, y, lambda_reg)
        # current_loss = mse + loss_reg
        
        current_loss, gradient = value_and_grad(cpd_training_normalized.loss_function)(weights, mu, Zs, y, lambda_reg)
        
        # Compute coefficients of line search function
        # The line search function is a polynomial function in alpha
        alpha_coeffs = cpd_linesearch_normalized.h_alpha_grad_coeffs_batch(weights, -gradient, mu, Zs, y, lambda_reg) 

        # Take derivatives of coefficients of the polynomial
        alpha_coeffs_deriv = cpd_linesearch_normalized.derivative_coefficients_of_polynomial(alpha_coeffs)
        
        # Compute roots to find optimal alphas
        sols = jnp.roots(jnp.flip(alpha_coeffs_deriv))
        
        # Find best alpha
        improve = False

        # Save best solution
        loss_best = current_loss
        weights_best = weights        
        
        for alpha in sols:
            # complex roots not possible
            if not jnp.iscomplex(alpha):
                alpha = jnp.real(alpha)
                new_loss = jnp.polyval(jnp.flip(alpha_coeffs), alpha)
                if new_loss <= loss_best:
                    weights_best = weights + alpha * -gradient
                    loss_best = new_loss
                    improve = True
                    if alpha < 0.:
                        print(f'Interesting, alpha < 0, alpha = {alpha}')
                        
        if not improve:
            print('No improvement')
        
        return weights_best, current_loss, optimizer_state
    
    
class Line_Search_Gradient_Descent_JAX(WeightUpdateMethod):
    
    """
    A class that represents the Line Search Gradient Descent method. The update is given by
    W_{new} = W - learning_rate * gradient
        
    The learning rate is computed using line search where the roots are evaluated 
    without a loop, such that the whole function can be jitted.
    
    The gradient is computed using automatic differentiation.
    
    """
    
    def __call__(self, weights, mu, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):

        
        current_loss, gradient = value_and_grad(cpd_training_normalized.loss_function)(weights, mu, Zs, y, lambda_reg)
        
        # Compute coefficients of line search function
        # The line search function is a polynomial function in alpha
        alpha_coeffs = cpd_linesearch_normalized.h_alpha_grad_coeffs_batch(weights, -gradient, mu, Zs, y, lambda_reg) 

        # Take derivatives of coefficients of the polynomial
        alpha_coeffs_deriv = cpd_linesearch_normalized.derivative_coefficients_of_polynomial(alpha_coeffs)
        
        # Compute roots to find optimal alphas
        sols = cpd_linesearch_normalized.roots_no_zeros(jnp.flip(alpha_coeffs_deriv))
        
        # Select only real solutions and for the rest select alpha = 0
        # Alpha = 0 is not an improvement
        # Will lead to extra computation, computing for alpha = 0, but easily jittalbe
        temp_zeros = jnp.zeros(sols.shape)
        sols_real = jnp.real(jnp.where(jnp.isreal(sols), sols, temp_zeros))
        
        # Use alpha coefficients to quickly compute the loss for each alpha
        h_alpha_value_map = vmap(jnp.polyval, in_axes=[None, 0], out_axes = 0)
        h_alpha_values = h_alpha_value_map(jnp.flip(alpha_coeffs), sols_real)
        
        # Select best alpha based on which result in the lowest value
        best_alpha = sols_real[jnp.argmin(h_alpha_values)]
        
        weights_best = weights + best_alpha*-gradient
        
        return weights_best, current_loss, optimizer_state


class Adam_Gradient_Descent(WeightUpdateMethod):
    """
    A class that represents the Adam method. The update is given by
    W_{new} = W - learning_rate * update_step
    
    The update step is compute using the Adam algorithm, see Kingma et al.
    
    This method has an optimizer state which is updated.
    
    The gradient is computed using automatic differentation.
    
    """
    def __call__(self, weights, mu, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        current_loss, gradient = value_and_grad(cpd_training_normalized.loss_function)(weights, mu, Zs, y, lambda_reg)
        
        # Get optimizer state
        m, v, beta_1, beta_2, epsilon = optimizer_state
        
        # Perform Adam update
        m_t = beta_1*m + (1.0-beta_1)*gradient
        v_t = beta_2*v + (1.0-beta_2)*gradient*gradient
        
        mhat = m_t/(1.0-beta_1**(iteration+1))
        vhat = v_t/(1.0-beta_2**(iteration+1))
        
        update_step = mhat/(jnp.sqrt(vhat)+epsilon)
        
        # Update weights
        new_weights = weights - learning_rate * update_step
        
        # Update optimizer state
        new_optimizer_state = (m_t, v_t, beta_1, beta_2, epsilon)
        
        return new_weights, current_loss, new_optimizer_state


