#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:14:45 2022

@author: Ewoud
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class BatchFeatureMap(ABC):
        
    """
    An abstract class to represent feature maps in general and specify which
    methods a feature map shoud have. A feature map is used to transform an
    input sample to a higher dimensional feature space.
    
    ...

        
    Methods
    -------
    batch_feature_map(batch):
        Compute the feature map for a batch of samples. Return a MxDxN tensor.
    one_dimension_feature_map(X_d) :
        Compute the feature map along a single dimension for a batch of samples. 
        Returns a MxN tensor.
    order() :
        Returns order M of the mapping.
    __str__() :
        Specifies how the mapping is printed as a string.
    """
    
    @abstractmethod
    def batch_feature_map(self, batch):
        pass
 
    @abstractmethod
    def one_dimension_feature_map(self, X_d):
        pass
    
    @abstractmethod
    def order(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
    
class BatchPolynomialFeatureMap(BatchFeatureMap):
        
    """
    A class that represents the polynomial feature map. The feature map 
    includes a bias term. 
    
    ...

        
    Attributes
    ----------
    M: int
        The order of the polynomial feature map. When M is 1 it is a linear 
        map, M is 2 quadratic etc.
        
    Methods
    -------
    batch_feature_map(batch) :
        Compute the feature map for a batch of samples. Return a DxMxN tensor.
    one_dimension_feature_map(X_d) :
        Compute the feature map along a single dimension for a batch of samples. 
        Returns a MxN tensor.
    order() :
        Returns order M of the mapping.
    __str__() :
        Specifies how the mapping is printed as a string.
    """
    
    
    def __init__(self, M: int):
        
        if M < 1:
            raise ValueError("Order must be 1 or higher")
        self.M = M
        
    def batch_feature_map(self, batch):
                    
        num_dim = batch.shape[1]
        N = batch.shape[0]
        
        mats = jnp.ones((self.M, N , num_dim))
        
        
        for m in range(self.M-1):
            mats = mats.at[m+1, :, :].set(mats[m, :, :]*batch)
        
        mats_T = jnp.transpose(mats, (2,0,1))
        
        Zs = mats_T
        
        return Zs
    
    def one_dimension_feature_map(self, X_d):
        
        
        N = X_d.shape[0]
        Zs = jnp.ones((self.M, N))
        
        for m in range(self.M-1):
            Zs = Zs.at[m+1, :].set(Zs[m, :]*X_d)

        return Zs            
            
    def order(self):
        return self.M
    
    def __str__(self):
        return 'Poly: M = ' + str(self.M)
    
    
class BatchFourierFeatureMap(BatchFeatureMap):
    
    """
    A class that represents the Fourier feature map. It is a deterministic 
    Fourier feature map that approximates the Gaussian kernel with M 
    sinusoids. 
    
    ...

    Attributes
    ----------
    M : int
        The order of the feature map.
    bound : float
        The boundary condition that is used for the approximation
    l : float 
        The length scale of the Gaussian kernel that is to be approximated
        
    Methods
    -------
    batch_feature_map(batch) :
        Compute the feature map for a batch of samples. Return a DxMxN tensor.
    one_dimension_feature_map(X_d) :
        Compute the feature map along a single dimension for a batch of samples. 
        Returns a MxN tensor.
    order() :
        Returns order M of the mapping.
    __str__() :
        Specifies how the mapping is printed as a string.
    """
    
    
    def __init__(self, M: int, bound, l):
        
        if M < 1:
            raise ValueError("Order must be 1 or higher")
        
        self.M = M
        self.bound = bound
        self.l = l

    def batch_feature_map(self, batch):
        
        m = jnp.arange(1, self.M+1)
        
        # Merged everything into one equation to enable broadcasting for speed up
        mats = 1.0/jnp.sqrt(self.bound)*jnp.sqrt(jnp.sqrt(2.0*jnp.pi)*self.l*jnp.exp(--jnp.pi*m[jnp.newaxis, :,jnp.newaxis]/2.0/self.bound*-jnp.pi*m[jnp.newaxis, :,jnp.newaxis]/2.0/self.bound*self.l*self.l/2))* jnp.sin(jnp.pi*m[jnp.newaxis, :, jnp.newaxis]*(batch.T[:, jnp.newaxis, :]+self.bound)/2.0/self.bound)
        
        
        Zs = mats
        
        return Zs
    
    def one_dimension_feature_map(self, X_d):
        
        
        m = jnp.arange(1, self.M+1)
        
        Zs_d = 1.0/jnp.sqrt(self.bound)*jnp.sqrt(jnp.sqrt(2.0*jnp.pi)*self.l*jnp.exp(--jnp.pi*m[:,jnp.newaxis]/2.0/self.bound*-jnp.pi*m[:,jnp.newaxis]/2.0/self.bound*self.l*self.l/2))* jnp.sin(jnp.pi*m[:, jnp.newaxis]*(X_d+self.bound)/2.0/self.bound)

        return Zs_d
        
    def order(self):
        return self.M
    
    def __str__(self):
        return 'Fourier: M = ' + str(self.M) + ',\n+' + 'l = ' + "{:.4f}".format(self.l) + ', bound = ' + str(self.bound)
    