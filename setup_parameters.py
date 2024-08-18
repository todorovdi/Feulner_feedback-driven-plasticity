#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Defines experiment parameters.
"""

import numpy as np
import os,shutil
from pathlib import Path

def main(name='test',rand_seed=0):
    
    # NAME #################################
    directory = Path.cwd() / 'results'
    name = name
    savname = directory / name
    
    if not os.path.exists(savname):
        os.mkdir(savname)
    else:
        print('Simulation with that name already exists!')
        return 0
        
    # MODEL #################################
    # network
    n = 400
    tau = 0.05
    nonlin = 'relu'
    grec = 0. # initialize recurrent weights in chaotic regime?
    fb_initial = 1. # initialize feedback weights smaller or larger?
    in_initial = 1. # initialize input weights smaller or larger?
    fb_sparsity = 1. # sparseness of feedback signal
    rec_sparsity = 1. # other tested so far: 0.5, 0.8
    fb_type = 'poserrorfb'
    model_input_dim = 3
    model_output_dim = 2
    
    # signal delays
    fwd_delay = 2
    fb_delay = 10
    
    # input noise: trial-to-trial variability (not time-dependent)
    noise_stim_amp = 0
    
    # output noise (velocity dependent)
    noise_amp = 0
    noise_kernel_size = 5
    
    # VR perturbation
    rot_phi = 30/180*np.pi
    
    # regularization
    alpha1 = 1e-3 # reg inp & out & fb
    gamma1 = 1e-3 # reg rec 
    beta1 = 2e-3 # regularization on activity
    
    # clip gradients
    clipgrad = 0.2   
    
    # learning rate & batch size
    lr = 1e-3
    batch_size = 20
        
    # PROTOCOL ###############################
    protocol = [ # dataID, perturbation, training_trials
            ['random',0,100], # first phase is always with static fb weights
            ['random',0,500],
            ['random',1,500],
            ['center-out-reach',0,0],
            ['center-out-reach',2,0]
        ]
    if name=='test':
        protocol = [ # dataID, perturbation, training_trials
                ['random',0,1], # first phase is always with static fb weights
                ['random',0,1],
                ['random',1,1],
                ['center-out-reach',0,0],
                ['center-out-reach',2,0]
            ] # just for fast testing
    
    # DATA #################################
    ntrials = 2000
    tsteps = 300
    dt = 0.01
    output_dim = 4
    input_dim = 7
    vel = 10
    p_test = 0.1 # test set size
    go_to_peak = 50 
    stim_on = 20 
    
    # random reach data set
    r_output_range = [-6,6]
    r_go_range = [70,220]
    
    # center out reach data set
    cor_output_range = 5
    cor_go_range = [170,220]
    ntargets = 8
    
    # RANDOM PUSH PERTURBATION ###############
    p1_amp = 10
    p1_pratio = 0.75
    p1_halflength = 5
    p1_from = 20
    p1_upto = 190
    
    # SAVE IT ALL ##############################
    model = {
        # neuron
        'n':n,
        'tau':tau,
        'nonlin':nonlin,
        'grec':grec,
        'fb_initial':fb_initial,
        'fb_sparsity':fb_sparsity,
        'fb_type':fb_type,
        'rec_sparsity':rec_sparsity,
        'in_initial':in_initial,
        'input_dim':model_input_dim,
        'output_dim':model_output_dim,
        # signal delays
        'fwd_delay':fwd_delay,
        'fb_delay':fb_delay,
        # time and reproducability
        'dt':dt,
        'tsteps':tsteps,
        'rand_seed':rand_seed,
        # input and output noise
        'noise_stim_amp':noise_stim_amp,
        'noise_amp':noise_amp,
        'noise_kernel_size':noise_kernel_size,
        # simulation protocol
        'protocol':protocol,
        'rot_phi':rot_phi,
        # ml regularization
        'alpha1':alpha1,
        'beta1':beta1,
        'gamma1':gamma1,
        'clipgrad':clipgrad,
        # ml training
        'lr':lr,
        'batch_size':batch_size
        }
    
    data = {
        'ntrials':ntrials,
        'tsteps':tsteps,
        'dt':dt,
        'input_dim':input_dim,
        'output_dim':output_dim,    
        'vel':vel,
        'p_test':p_test,
        'go_to_peak':go_to_peak,
        'stim_on':stim_on,
        'random':{'output_range':r_output_range,'go_range':r_go_range},
        'center-out-reach':{'output_range':cor_output_range,'ntargets':ntargets,
                            'go_range':cor_go_range}
        }
    
    p1 = {
        'amp':p1_amp,
        'pratio':p1_pratio,
        'halflength':p1_halflength,
        'from':p1_from,
        'upto':p1_upto
        }
    
    params = {'model':model,'data':data,'directory':directory,'name':name,
              'p1':p1}
    np.save(savname / 'params',params)
    shutil.copy(__file__, savname / 'setup_parameters.py') 
    
if __name__ == "__main__":
    main()
     