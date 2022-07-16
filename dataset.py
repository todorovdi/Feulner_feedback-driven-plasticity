#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Toolbox for data creation.
"""

import numpy as np

def _prepare_data(start_point,end_point,go_on,vel,tsteps,input_dim,
                  output_dim,dt,stim_range,go_to_peak,stim_on):
    ntrials = start_point.shape[0]
    def sig(x,beta):
        return 1/(1+np.exp(-x*beta))
    
    # prepare xaxis for smooth transition
    xx = np.linspace(-1,1,100,endpoint=False)
    ytemp = sig(xx,vel)
    
    # create target
    target = np.zeros((ntrials,tsteps,output_dim))
    for j in range(ntrials):
        target[j,:(go_on[j]+go_to_peak),:2] = start_point[j]
        target[j,(go_on[j]+go_to_peak):,:2] = end_point[j]  
        target[j,(go_on[j]-go_to_peak):(go_on[j]+go_to_peak),:2] += \
                ytemp[:,None]*(end_point[j]-start_point[j])[None,:]
    
    # add target velocity
    target[:,:,2:] = np.gradient(target[:,:,:2],dt,axis=1)
    
    # create stimulus
    stimulus = np.zeros((ntrials,tsteps,input_dim))
    stimulus[:,:,3:5] = target[:,:,2:]
    stimulus[:,:,5:] = target[:,:,:2]
    for j in range(ntrials):
        stimulus[j,stim_on:,:2] = end_point[j]-start_point[j]
        stimulus[j,:(go_on[j]-go_to_peak),2] = stim_range
    return target,stimulus

def create_data_velocity_random(data):
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
    output_range = data['random']['output_range']
    go_range = data['random']['go_range']
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
    data = {'params':data,'target':target[train_idx],
           'peak_speed':go_on[train_idx],'stimulus':stimulus[train_idx],
           'test_set':test_set}
    
    print('RANDOM REACH DATASET CONSTRUCTED!')
    return data

def create_data_velocity_centeroutreach(data):
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
                                    go_to_peak,stim_on)

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
    pratio = dpert['pratio']
    halflength = dpert['halflength']
    for i in range(batch_size):
        if np.random.rand()<pratio:
            tmp = int(np.random.choice(range(dpert['from'],dpert['upto'],1),1)[0])
            tdat[(tmp-halflength):(tmp+halflength),i,0] = dpert['amp']
        if np.random.rand()<pratio:
            tmp = int(np.random.choice(range(dpert['from'],dpert['upto'],1),1)[0])
            tdat[(tmp-halflength):(tmp+halflength),i,1] = dpert['amp']
    return tdat