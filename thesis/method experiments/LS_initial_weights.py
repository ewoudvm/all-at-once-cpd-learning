#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:12:37 2022

@author: Ewoud
"""

import sys


# Custom modules
# Add cpd_training module to path to enable import in subfolder
CPD_TRAINING_PATH = '../'
if not CPD_TRAINING_PATH in sys.path:
    sys.path.append(CPD_TRAINING_PATH)
    

import experiment_helper
import plot_helper

ALS_COLOR = plot_helper.ALS_COLOR
LS_COLOR = plot_helper.LS_COLOR
ADAM_COLOR = plot_helper.ADAM_COLOR

# =============================================================================
# Airfoil experiment
# =============================================================================

airfoil_experimet_file = '/Users/Ewoud/Documents/Ewoud/Systems&Control/Thesis/thesis_tensor_networks/CPD JAX/data set experiments/experiment results/Airfoil initial value experiment including LS--2022-09-07 10:27:46.155375.pickle'

airfoil_experiment = experiment_helper.load_results(airfoil_experimet_file)

airfoil_results_without_SteGD = []

for result in airfoil_experiment.results:
    
    if result.method != 'Steepest Gradient Descent':
        airfoil_results_without_SteGD.append(result)
        

plot_helper.plot_training_multiple_experiments_averaged(airfoil_results_without_SteGD)

print('Airfoil')
plot_helper.print_mean_final_training_losses(airfoil_experiment.results)
plot_helper.print_mean_final_validation_losses(airfoil_experiment.results)