#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Toolbox for data creation.
"""

import numpy as np

def _prepare_data(start_point,end_point,go_on,vel,tsteps,input_dim,
                  output_dim,dt,stim_range,go_to_peak,stim_on,transition=1):
    """
    Prepare data for RNN training by generating target and stimulus arrays.
    Parameters:
    start_point (np.ndarray): Starting points for each trial, shape (ntrials, 2).
    end_point (np.ndarray): Ending points for each trial, shape (ntrials, 2).
    go_on (np.ndarray): Time steps at which the transition starts for each trial, shape (ntrials,).
    vel (float): Velocity parameter for the sigmoid function.
    tsteps (int): Total number of time steps.
    input_dim (int): Dimensionality of the input data.
    output_dim (int): Dimensionality of the output data.
    dt (float): Time step duration.
    stim_range (float): Range of the stimulus.
    go_to_peak (int): Number of time steps to reach the peak.
    stim_on (int): Time step at which the stimulus is turned on.

    data dictionary according to the setup parameters :
    {'ntrials': 2000,
     'tsteps': 300,
     'dt': 0.01,
     'input_dim': 7,
     'output_dim': 4,
     'vel': 10,
     'p_test': 0.1,
     'go_to_peak': 50,
     'stim_on': 20,
     'random': {'output_range': [-6, 6], 'go_range': [70, 220]},
     'center-out-reach': {'output_range': 5,
     'ntargets': 8,
     'go_range': [170, 220]}}

    Returns:
    tuple: A tuple containing:
        - target (np.ndarray): Target data for each trial, shape (ntrials, tsteps, output_dim).
            first two dimensions are the position, last two dimensions are the velocity
            !it is target data for every time bin, not just the endpoint
        - stimulus (np.ndarray): Stimulus data for each trial, shape (ntrials, tsteps, input_dim).
            dim [5-6] = first two dimensions of target mvt (so position)
            dim [3-4] = last two dimensions of target mvt (so velolcity)
            dim [0-1] = beforfe stim_on it is zero, after stim_on it is end_point-start_point
            dim 2 = before go_on[k]-go_to_peak = 0, after = stim_range (hold signal)
    """

    ntrials = start_point.shape[0]
    def sig(x,beta):
        return 1/(1+np.exp(-x*beta))
    
    # prepare xaxis for smooth transition
    xx = np.linspace(-1,1,100,endpoint=False)
    ytemp = sig(xx,vel)
    
    # create target
    target = np.zeros((ntrials,tsteps,output_dim))
    for j in range(ntrials):
        # for each trial, last two dim until go_on is just standing at start
        target[j,:(go_on[j]+go_to_peak),:2] = start_point[j]
        # for each trial, last two dim after go_on is just standing at end point
        target[j,(go_on[j]+go_to_peak):,:2] = end_point[j]  
        # for each trial, last two dim around go_to_peak is defined by sigmoid
        if transition:
            target[j,(go_on[j]-go_to_peak):(go_on[j]+go_to_peak),:2] += \
                ytemp[:,None]*(end_point[j]-start_point[j])[None,:]
    
    # add target velocity
    target[:,:,2:] = np.gradient(target[:,:,:2],dt,axis=1)
    
    # create stimulus
    stimulus = np.zeros((ntrials,tsteps,input_dim))
    stimulus[:,:,3:5] = target[:,:,2:].copy()
    stimulus[:,:,5:]  = target[:,:,:2].copy()
    for j in range(ntrials):
        stimulus[j,stim_on:,:2] = end_point[j]-start_point[j] # visible endpoint position signal
        stimulus[j,:(go_on[j]-go_to_peak),2] = stim_range # hold signal
    return target,stimulus

def create_data_velocity_random(data_params):
    # PARAMS #################################
    ntrials = data_params['ntrials']
    tsteps = data_params['tsteps'] # 300
    dt = data_params['dt'] # 0.01
    output_dim = data_params['output_dim'] # 4
    input_dim = data_params['input_dim'] # 7
    vel = data_params['vel'] # 10
    p_test = data_params['p_test'] # 0.1
    go_to_peak = data_params['go_to_peak'] # 50
    stim_on = data_params['stim_on'] # 20
    output_range = data_params['random']['output_range'] # -6,6
    go_range = data_params['random']['go_range'] # 70,220
    ##########################################
    
    # create artifical data
    start_point = np.random.uniform(output_range[0],output_range[1],(ntrials,2))
    end_point = np.random.uniform(output_range[0],output_range[1],(ntrials,2))
    go_on = np.random.uniform(go_range[0],go_range[1],ntrials).astype(int)
    
    target,stimulus = _prepare_data(start_point, end_point, go_on, vel, tsteps,
                                    input_dim,output_dim,dt,output_range[1],
                                    go_to_peak,stim_on)

    # create testset
    test_idx = np.random.rand(ntrials)<p_test
    test_set = {'target':target[test_idx],'stimulus':stimulus[test_idx],
                 'peak_speed':go_on[test_idx]}
    train_idx = test_idx==False
    
    # save it
    data = {'params':data_params,'target':target[train_idx],
           'peak_speed':go_on[train_idx],'stimulus':stimulus[train_idx],
           'test_set':test_set}
    
    print('RANDOM REACH DATASET CONSTRUCTED!')
    return data

def create_data_velocity_centeroutreach(data, transition=1):
    # PARAMS #################################
    ntrials = data['ntrials']
    tsteps = data['tsteps']
    dt = data['dt']
    output_dim = data['output_dim']
    input_dim = data['input_dim']
    vel = data['vel']
    p_test = data['p_test']
    go_to_peak = data['go_to_peak']
    stim_on = data['stim_on']
    output_range = data['center-out-reach']['output_range']
    go_range = data['center-out-reach']['go_range']
    ntargets = data['center-out-reach']['ntargets']
    ##########################################
    
    # create artifical data
    start_point = np.zeros((ntrials,2))
    phi = np.linspace(0,2*np.pi,ntargets,endpoint=False)
    tids = np.random.choice(range(ntargets),ntrials)
    end_point = (output_range*np.array([np.cos(phi[tids]),np.sin(phi[tids])])).T
    go_on = np.random.uniform(go_range[0],go_range[1],ntrials).astype(int)
    
    target,stimulus = _prepare_data(start_point, end_point, go_on, vel, tsteps,
                                    input_dim,output_dim,dt,output_range,
                                    go_to_peak,stim_on,transition=transition)

    # create testset
    test_idx = np.random.rand(ntrials)<p_test
    test_set = {'target':target[test_idx],'stimulus':stimulus[test_idx],
                 'peak_speed':go_on[test_idx],'tids':tids[test_idx]}
    train_idx = test_idx==False
    
    # save it
    data = {'params':data,'target':target[train_idx],
           'peak_speed':go_on[train_idx],'stimulus':stimulus[train_idx],
           'test_set':test_set,'tids':tids[train_idx]}
    
    print('CENTER OUT REACH DATASET CONSTRUCTED!')
    return data

def perturb(tdat,batch_size,dpert):
    '''
    generate trajectory _velocity_ perturbations 

    dpert is a dictionary with the following keys (example)
    dpert = {'pratio':0.5 -- ratio of trajectories recieveing perturbation,
    'halflength':10, -- num of timebins of perturbation (velocity bump)
    'from':20,
    'upto':200,'amp':10}
    '''
    pratio = dpert['pratio']
    halflength = dpert['halflength']
    for i in range(batch_size):
        # run random twice to perturn first or sedond coordinate
        for coordi in [0,1]:
            if np.random.rand()<pratio:
                tmp = int(np.random.choice(range(dpert['from'],dpert['upto'],1),1)[0])
                tdat[(tmp-halflength):(tmp+halflength),i,coordi] = dpert['amp']

        # if np.random.rand()<pratio:
        #     tmp = int(np.random.choice(range(dpert['from'],dpert['upto'],1),1)[0])
        #     tdat[(tmp-halflength):(tmp+halflength),i,0] = dpert['amp']
        # if np.random.rand()<pratio:
        #     tmp = int(np.random.choice(range(dpert['from'],dpert['upto'],1),1)[0])
        #     tdat[(tmp-halflength):(tmp+halflength),i,1] = dpert['amp']
    return tdat