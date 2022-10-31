# all-at-once-cpd-learning
Implementation of all-at-once gradient descent optimization for a CPD constrained kernel machine.

This repository contains all the algorithms and experiments that were used for the master thesis 'All-at-once optimization for kernel machines with canonical polyadic decompositions' written by Ewoud van Mourik. It can be accessed via the TU Delft repository.  

## All at once CPD learning.
The thesis studies the application of all-at-once gradient descent optimization to a CPD constrained kernel machine. The CPD constrained kernel machine is useful, since its solution complexity scales both linearly in the number of samples and number of features. Three optimization methods are studied and implemented; Steepest Gradient Descent, Line search Gradient Descent and the Adam method. The framework can be used with any other all-at-once optimization update using the provided base classes and the general algorithm. Furthermore, the Alternating Least Squares optimization method is implemented, such that the all-at-once optimization methods can be compared to it. 

## Using the code
The code relies on several Python libraries. Most notably the JAX library. All the packages that were used and their versions are listed in the jax-env.txt file. This can also be used to create a new virtual environment that can be used to run the code. It is noted that adaptations may be required, for example when using GPUs of TPUs. 

The used data sets are not included in the repository, but they are open source and in the thesis it is listed where they can be accessed. 
