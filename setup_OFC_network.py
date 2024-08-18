#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Runs initial training of network model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from dataset import create_data_velocity_centeroutreach, \
                    create_data_velocity_random, \
                        perturb
from modeldef import RNN,test

def main(name='test'):
    
    # LOAD PARAMETERS ###################
    directory = Path.cwd() / 'results'
    name = name
    savname = directory / name
        
    params = np.load(savname / 'params.npy',allow_pickle=True).item()
    protocol = params['model']['protocol']
    
    if directory!=params['directory'] or name!=params['name']:
        print('Naming is inconsistent!')
        return 0 
    
    
    # SETUP SIMULATION #################
    rand_seed = params['model']['rand_seed']
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    
    # GPU usage 
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        torch.set_default_device('cuda')
    else:
        dtype = torch.FloatTensor
        torch.set_default_device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    params['model'].update({'dtype':dtype,'device':device})
    
        
    # CREATE DATA ###################
    dataA = create_data_velocity_random(params['data'])
    dataB = create_data_velocity_centeroutreach(params['data'])
    data0 = {'random':dataA,'center-out-reach':dataB}
        
    
    # SETUP MODEL #################
    model = RNN(params['model']['input_dim'],
                params['model']['output_dim'],
                params['model']['n'],
                params['model']['dt']/params['model']['tau'],
                dtype,
                params['model']['dt'],
                params['model']['fwd_delay'],
                params['model']['fb_delay'],
                fb_sparsity=params['model']['fb_sparsity'],
                nonlin=params['model']['nonlin'],
                noiseout=params['model']['noise_amp'],
                noise_kernel_size=params['model']['noise_kernel_size'],
                noisein=params['model']['noise_stim_amp'],
                rec_sparsity=params['model']['rec_sparsity'])
       
    # recurrent initialization
    if params['model']['grec']!=0:
        tmp = model.state_dict()
        tmp['rnn.weight_hh_l0'] = torch.FloatTensor(
            params['model']['grec']/np.sqrt(params['model']['n']) \
            * np.random.randn(params['model']['n'],params['model']['n'])).type(dtype)
        model.load_state_dict(tmp,strict=True)
    
    # input initialization
    tmp = model.state_dict()
    tmp['rnn.weight_ih_l0'] = params['model']['in_initial']*tmp['rnn.weight_ih_l0']
    model.load_state_dict(tmp,strict=True)
    
    # SETUP OPTIMIZER #################
    criterion = nn.MSELoss(reduction='none') 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=params['model']['lr']) 
    
    # START INITIAL TRAINING #################
    for phase in range(len(protocol)):
        print('\n####### PHASE %d #######'%phase)
        if phase==0: # fix fb weights in first training phase
            tmp = model.state_dict()
            tmp['feedback.weight'] *= params['model']['fb_initial']
            model.load_state_dict(tmp,strict=True)
            model.feedback.weight.requires_grad = False
            np.save(savname / 'data',data0)
        else:
            model.feedback.weight.requires_grad = True
        data = data0[protocol[phase][0]]
        perturbation = protocol[phase][1]
        training_trials = protocol[phase][2]
        tout = data['target'][:,:,2:]
        tstim = data['stimulus']
        # rotate output matrix if perturbation type is VR (2)
        if perturbation==2:
            rot_phi = params['model']['rot_phi']
            rotmat = np.array([[np.cos(rot_phi),-np.sin(rot_phi)],
                                [np.sin(rot_phi),np.cos(rot_phi)]])
            state_dict = model.state_dict()
            state_dict['output.weight'] = dtype(rotmat) @ \
                                            state_dict['output.weight']
            model.load_state_dict(state_dict, strict=True)
        # convert to pytorch form
        target = torch.zeros(training_trials, params['model']['tsteps'], 
            params['model']['batch_size'], params['model']['output_dim']).type(dtype)
        stimulus = torch.zeros(training_trials, params['model']['tsteps'], 
            params['model']['batch_size'], tstim.shape[-1]).type(dtype)
        pert = torch.zeros(training_trials, params['model']['tsteps'], 
            params['model']['batch_size'], params['model']['output_dim']).type(dtype)
        for j in range(training_trials):
            idx = np.random.choice(range(tout.shape[0]),params['model']['batch_size'],
                                   replace=False)
            target[j] = torch.Tensor(tout[idx].transpose(1,0,2)).type(dtype)
            stimulus[j] = torch.Tensor(tstim[idx].transpose(1,0,2)).type(dtype)
            # insert perturbation if perturbation type is random push (1)
            if perturbation==1:
                pert[j] = perturb(pert[j],params['model']['batch_size'],params['p1'])
        # ACTUAL TRAINING STARTS
        lc = []
        model.train()
        for epoch in range(training_trials): 
            toprint = OrderedDict()
            optimizer.zero_grad()
            output,hidden = model(stimulus[epoch],pert[epoch])
            loss_train = criterion(output, output*0).mean()
            toprint['Loss'] = loss_train
            
            # add regularization
            # term 1: parameters
            regin = params['model']['alpha1']*model.rnn.weight_ih_l0.norm(2)
            regout = params['model']['alpha1']*model.output.weight.norm(2)
            regoutb = params['model']['alpha1']*model.output.bias.norm(2)
            regfb = params['model']['alpha1']*model.feedback.weight.norm(2)
            regfbb = params['model']['alpha1']*model.feedback.bias.norm(2)
            regrec = params['model']['gamma1']*model.rnn.weight_hh_l0.norm(2)
            toprint['In'] = regin
            toprint['Rec'] = regrec
            toprint['Out'] = regout
            toprint['OutB'] = regoutb
            toprint['Fb'] = regfb
            toprint['FbB'] = regfbb
            # term 2: rates
            regact = params['model']['beta1']*hidden.pow(2).mean() 
            toprint['Act'] = regact
         
            loss = loss_train+regin+regrec+regout+regoutb+regact+regfbb+regfb
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           params['model']['clipgrad'])
            optimizer.step()
                
            train_running_loss = [loss_train.detach().item(),regact.detach().item(),
                                  regin.detach().item(), regrec.detach().item(),
                                  regout.detach().item(), regoutb.detach().item(),
                                  regfb.detach().item(), regfbb.detach().item()]
            print(('Epoch=%d | '%(epoch)) +" | ".join("%s=%.4f"%(k, v) for \
                                                      k, v in toprint.items()))
            lc.append(train_running_loss)       
        # save this phase
        torch.save({'epoch': training_trials,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lc':np.array(lc),
                    'params':params
                    }, savname / ('phase'+str(phase)+'_training'))
        print('MODEL TRAINED!')
        # test model
        model.eval()
        test(model,data,params,str(savname / ('phase'+str(phase)+'_')),lc,
                      dopert=0 if perturbation==2 else perturbation,
                      dataC=dataB)
        print('MODEL TESTED!')

if __name__ == "__main__":
    main()
     