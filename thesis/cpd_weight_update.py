#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:34:17 2022

@author: Ewoud
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import value_and_grad, vmap, jit

# Custom modules
import cpd_training
import cpd_linesearch

# =============================================================================
# Weight update class that are callable: Used for training
# =============================================================================
#### Weight update classes

class WeightUpdateMethod(ABC):

    """
    An abstract class that fixes the structure of any weight update.
    If a new weight update is added it must/can inherent this class to ensure
    that the correct call method is implemented.
    
    This method is a class that is callable and therefore works as a function.
    As a result, it can be jitted after it has be initiated. 
    
    The inputs of the weight update method are:
        Weights: CPD stored as DxMxR tensor
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
    def __call__(self, weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        pass
    

class Steepest_Gradient_Descent(WeightUpdateMethod):
    """
    A class that represents the Steepest Gradient Descent method. The update is given by
    W_{new} = W - learning_rate * gradient
    
    The gradient is computed using the analytical expression.
    
    """
    
    
    def __call__(self, weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
    
        # Compute gradient and losses
        gradient, mse, loss_reg = batch_gradient(weights, Zs, y, lambda_reg)
        
        # Update weights with steepest gradient descent
        new_weights = weights - learning_rate * gradient
        
        # Compute total cost
        loss = mse + loss_reg
        
        return new_weights, loss, optimizer_state
    
    
class AD_Steepest_Gradient_Descent(WeightUpdateMethod):
    """
    A class that represents the Steepest Gradient Descent method. The update is given by
    W_{new} = W - learning_rate * gradient
    
    The gradient is computed using automatic differentiation.
    
    """
    
    def __call__(self, weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        loss, gradient = value_and_grad(cpd_training.loss_function)(weights, Zs, y, lambda_reg)
        
        new_weights = weights - learning_rate * gradient
        
        return new_weights, loss, optimizer_state


class Line_Search_Gradient_Descent(WeightUpdateMethod):
    """
    A class that represents the Line Search Descent method. The update is given by
    W_{new} = W - learning_rate * gradient
        
    The learning rate is computed using line search where the roots are evaluated in a loop.
    As a result, this function cannot be jitted.
    
    """
    
    def __call__(self, weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        # Compute gradient and losses
        gradient, mse, loss_reg = batch_gradient(weights, Zs, y, lambda_reg)
        current_loss = mse + loss_reg
        
        # Compute coefficients of line search function
        # The line search function is a polynomial function in alpha
        alpha_coeffs = cpd_linesearch.h_alpha_grad_coeffs_batch(weights, -gradient, Zs, y, lambda_reg) 

        # Take derivatives of coefficients of the polynomial
        alpha_coeffs_deriv = cpd_linesearch.derivative_coefficients_of_polynomial(alpha_coeffs)
        
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
    
    """
    
    def __call__(self, weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        # Compute gradient and losses
        gradient, mse, loss_reg = batch_gradient(weights, Zs, y, lambda_reg)
        current_loss = mse + loss_reg
        
        # Compute coefficients of line search function
        # The line search function is a polynomial function in alpha
        alpha_coeffs = cpd_linesearch.h_alpha_grad_coeffs_batch(weights, -gradient, Zs, y, lambda_reg) 

        # Take derivatives of coefficients of the polynomial
        alpha_coeffs_deriv = cpd_linesearch.derivative_coefficients_of_polynomial(alpha_coeffs)
        
        # Compute roots to find optimal alphas
        sols = cpd_linesearch.roots_no_zeros(jnp.flip(alpha_coeffs_deriv))
        
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
    
    """
    
    def __call__(self, weights, Zs, y, lambda_reg, learning_rate, optimizer_state, iteration):
        
        # Compute gradient and losses
        gradient, mse, loss_reg = batch_gradient(weights, Zs, y, lambda_reg)
        current_loss = mse + loss_reg
        
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

# =============================================================================
# Gradient functions 
# =============================================================================
#### Gradient functions

@jit
def batch_gradient(weights, Zs, y, lambda_reg):
    
    """
    Compute gradient for the loss function 1/N * (y- <Z(x), W>)^2 + l * <W,W>
    with respect of all factors for the CPD W. The gradient is computed for the
    whole batch {X, y} that contains N samples. The input has transformed to a
    higher dimensional feature space and is passed as Zs. 
    
    The MSE for the current weights and batch is also returned as well as the 
    value of the regularization term l * <W,W> where l is lambda_reg. 

    Parameters
    ----------
    weights : Numpy ndarray
        Current weights stored as DxMxR tensor.
    Zs : Numpy ndarray
        Transformed input data stored as DxMxN tensor.
    y : Numpy ndarray
        Batch of outputs for which to compute the gradient. With N samples.
    lambda_reg : float
        Regularization term for the weights. 

    Returns
    -------
    grad : Numpy ndarray
        The gradient is returned as a stacked matrix of size DxMxR where the gradient
        with respect to each factor W^{(d)} is of size MxR and the are stacked 
        along the first axis for d = 1, ..., D. 
    mse : float
        MSE value for current batch and weights.
    loss_reg : float
        Value of regularization term for current weights. 

    """
    
    # Regularization term
    # Compute full gamma
    weights_T = jnp.swapaxes(weights, 1, 2)
    gamma = jnp.matmul(weights_T, weights)
    gamma_full = jnp.prod(gamma, axis=0)
    
    # Compute gradient of regularization term
    gradient = 2*lambda_reg*jnp.matmul(weights, gamma_full / gamma)
    
    # Compute losses
    loss_reg = lambda_reg * jnp.sum(gamma_full)
    
    # Compute MSE and gradient of error term
    batch_size = Zs.shape[2]
    mse_grad, mse = mse_gradient(weights, Zs, y, batch_size)
    
    gradient = gradient + mse_grad
                
    return gradient, mse, loss_reg

    
@jit
def mse_gradient(weights, Zs, y, batch_size: int):
    """
    Compute the gradient of the MSE term 1/N * (y- <Z(x), W>)^2 with respect of W. 
    Z(X) is the mapping of batch X with feature map Z(.).
    The gradient is returned as a stacked matrix of size DxMxR where the gradient
    with respect to each factor W^{(d)} is of size MxR and the are stacked 
    on the first axis for d = 1, ..., D. 

    Parameters
    ----------
    weights : Numpy ndarray
        Current weights stored as DxMxR tensor.
    Zs : Numpy ndarray
        Transformed input data stored as DxMxN tensor.
    y : Numpy ndarray
        Batch of outputs for which to compute the gradient.
    batch_size : int
        The size of the total batch for which the gradient is computed. 

    Returns
    -------
    gradient : Numpy ndarray
        Gradients of MSE term stacked vertically for all the factors for the CPD
    mse : float
        MSE term for this batch and the current weights. 

    """
    
    # Compute full ZW
    Zs_T = jnp.swapaxes(Zs, 1 , 2)
    ZW = jnp.matmul(Zs_T, weights)
    batch_ZW = jnp.prod(ZW, axis=0)
    
    # Compute predicitions and errors for mini-batch
    preds = jnp.sum(batch_ZW, axis=1)
    errors = y-preds
    
    # Compute losses
    mse = jnp.dot(errors,errors)/batch_size

    # Compute gradient
    batch_ZW_all = batch_ZW / ZW
    batch_ZW_all_T = jnp.swapaxes(batch_ZW_all, 1, 2)
    Zs_outer_batch_ZW_all = Zs[:, :, jnp.newaxis, :] * batch_ZW_all_T[:, jnp.newaxis, : , :]
    
    gradient = -2.0/batch_size * jnp.matmul(Zs_outer_batch_ZW_all, errors)
        
    return gradient, mse

