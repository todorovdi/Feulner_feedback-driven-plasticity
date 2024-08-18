#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Toolbox for model definition.
"""
import numpy as np
import torch
import torch.nn as nn

# random
class RNN(nn.Module):
    '''Feedback of error: online position difference.'''
    def __init__(self, n_inputs, n_outputs, n_neurons, alpha, dtype, dt,
                 fwd_delay, fb_delay=0, biolearning=False, noiseout=0, noisein=0,
                 nonlin='relu',fb_sparsity=1,noise_kernel_size=1, rec_sparsity=1):
        super(RNN, self).__init__()
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.alpha = alpha
        self.dt = dt
        
        self.rnn = nn.RNN(n_inputs, n_neurons, num_layers=2, bias=False) 
        self.output = nn.Linear(n_neurons, n_outputs)
        self.feedback = nn.Linear(n_outputs, n_neurons)
        self.dtype = dtype
        
        self.fwd_delay = fwd_delay
        self.fb_delay = fb_delay
        self.delay = fwd_delay+fb_delay
        
        self.biolearning = biolearning
        if biolearning:
            self.rnn.weight_hh_l0.requires_grad = False
        self.noisein = noisein
        self.noiseout = noiseout
        self.noise_kernel_lenk = noise_kernel_size
        
        if nonlin=='relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlin=='tanh':
            self.nonlin = torch.nn.Tanh()
        elif nonlin=='sigmoid':
            self.nonlin = torch.nn.Sigmoid()

        self.mask = nn.Linear(n_outputs, n_neurons, bias=False) 
        self.mask.weight = nn.Parameter((torch.rand(n_neurons, n_outputs) < fb_sparsity).float()).type(dtype)
        self.mask.weight.requires_grad = False

        self.mask2 = nn.Linear(n_neurons, n_neurons, bias=False) 
        self.mask2.weight = nn.Parameter((torch.rand(n_neurons, n_neurons) < rec_sparsity).float()).type(dtype)
        self.mask2.weight.requires_grad = False

    def init_hidden(self):
        return ((torch.rand(self.batch_size, self.n_neurons)-0.5)*0.2).type(self.dtype)
    
    # ONE SIMULATION STEP
    def f_step(self,xin,x1,r1,v1fb,v1,pin,popto):
        x1 = x1 + self.alpha*(-x1 + r1 @ (self.mask2.weight*self.rnn.weight_hh_l0).T 
                                  + xin[:,:3] @ self.rnn.weight_ih_l0.T
                                  + v1fb @ (self.mask.weight*self.feedback.weight).T 
                                  + self.feedback.bias.T
                                  + popto
                              )
        r1 = self.nonlin(x1)
        vt = self.output(r1) + pin
        v1 = v1 + self.dt*(xin[:,3:5]-vt)
        return x1,r1,v1
    
    # BIOLOGICALLY PLAUSIBLE LEARNING RULE
    def dW(self,pre,post,lr,inp,fb,presum):
        with torch.no_grad():
            return self.alpha*0.1*(
                + lr*self.mask2.weight*torch.outer(fb@(self.mask.weight*self.feedback.weight).T,presum)
                )

                    
    # GET VELOCITY OUTPUT (NOT ERROR)
    def get_output(self,testl1):
        if self.noiseout>0:
            return self.output(testl1)+self.output(testl1)*self.create_noise(testl1)
        else:
            return self.output(testl1)
        
    # SIMULATE MOTOR NOISE (scales with velocity output)
    def create_noise(self,testl1):
        # time varying noise
        tmp = self.noiseout*torch.randn(testl1.shape[1],testl1.shape[0],2)
        tmp = tmp.permute(0,2,1)
        lenk = self.noise_kernel_lenk
        kernel = torch.ones(2,tmp.shape[1],lenk)/lenk
        padding = lenk // 2 + lenk % 2
        noise = torch.nn.functional.conv1d(tmp, kernel, padding=padding)[:,:,:testl1.shape[0]]
        noise = noise.permute(2,0,1)*np.sqrt(lenk)  
        return noise
    
    # RUN MODEL
    def forward(self, X, Xpert, lr=1e-3, popto=None):
        self.batch_size = X.size(1)
        hidden0 = self.init_hidden()
        x1 = hidden0
        r1 = self.nonlin(x1)
        v1 = self.output(r1)
        hidden1 = []
        poserr = []
        presum = r1
        dw_acc = torch.zeros(self.rnn.weight_hh_l0.shape)

        if popto is None:
            popto = torch.zeros((X.shape[0],X.shape[1],r1.shape[1]))
        # prerun simulation (until 'delay' is reached)
        for j in range(self.fwd_delay+1):
            x1,r1,v1 = self.f_step(X[0],x1,r1,
                                         v1*0,v1,Xpert[0],
                                         popto[0]) 
            hidden1.append(r1)
            poserr.append(v1)
        # generate noise (input and output)
        X = X+ self.noisein*torch.randn(X.shape[1],X.shape[2])[None,:,:]
        if self.noiseout>0:
            noise = self.create_noise(Xpert)
        else:
            noise = torch.zeros(Xpert.shape)
        # now real time simulation
        for j in range(X.size(0)):
            if (self.fb_delay<0) | (j<=self.fb_delay):
                x1,r1,v1 = self.f_step(X[j],x1,r1,
                                       v1*0,v1,
                                       Xpert[j]+noise[j]*self.output(r1),
                                       popto[j])
            else:
                x1,r1,v1 = self.f_step(X[j],x1,r1,
                                             poserr[j-self.delay],
                                             v1,Xpert[j]+noise[j]*self.output(r1),
                                             popto[j])
                if self.biolearning and j%5:
                    dw_acc += self.dW(x1[0],r1[0],lr,X[j,0],
                                      poserr[j-self.delay][0],presum[0]).detach()
            if self.biolearning:
                presum += r1.detach()
            hidden1.append(r1)
            poserr.append(v1)
        hidden1 = torch.stack(hidden1[1:])
        poserr = torch.stack(poserr[1:])
        if self.biolearning:
            with torch.no_grad():
                self.rnn.weight_hh_l0 += dw_acc
        if self.fwd_delay==0:
            return poserr, hidden1
        elif self.fwd_delay>0:
            return poserr[(self.fwd_delay):], hidden1[:(-self.fwd_delay)]


def run_model(model,params,data,fb=True,dopert=0):
    if not fb:
        state_dict = model.state_dict()
        save = state_dict['feedback.weight'].detach().clone()
        state_dict['feedback.weight'] *= 0
        model.load_state_dict(state_dict)

    from dataset import perturb
    # SETUP ############
    testdata = data['test_set']
    tout = testdata['target'][:,:,2:]
    tstim = testdata['stimulus']
    stim = torch.Tensor(tstim.transpose(1,0,2)).type(model.dtype)
    pert = torch.zeros(tout.transpose(1,0,2).shape).type(model.dtype)
    if dopert==1: # if perturbation type is random pushes
        pert = perturb(pert,stim.shape[1],params['p1'])
    # RUN ############
    testout,testl1 = model(stim,pert)
    error = testout.cpu().detach().numpy().transpose(1,0,2)
    testout = model.get_output(testl1) + pert
    # SAVE ############
    output = testout.cpu().detach().numpy().transpose(1,0,2)
    activity = testl1.cpu().detach().numpy().transpose(1,0,2)
    pert = pert.cpu().detach().numpy().transpose(1,0,2)
    dic = {'target':tout,'stimulus':tstim,'error':error,
            'peak_speed':testdata['peak_speed'],'pert':pert,
            'activity':activity,'output':output}
    if 'tids' in testdata.keys():
        dic.update({'tid':testdata['tids']})
    
    if not fb:
        state_dict = model.state_dict()
        state_dict['feedback.weight'] = save
        model.load_state_dict(state_dict)
    return dic
    
def test(model,data,params,savname,lc,dopert,dataC):
    dic = run_model(model,params,data,dopert=dopert)
    np.save(savname+'testing',dic)
    
    # PLOTS ############
    import matplotlib.pyplot as plt
    import sklearn.metrics as met
    
    # setup
    cols = [plt.cm.magma(i) for i in np.linspace(0.1,0.9,8)]
    dt = params['model']['dt']
    
    # helper functions
    def get_pos(vel):
        pos = np.zeros(vel.shape)
        for j in range(vel.shape[1]):
            if j==0:
                pos[:,j] = dt*vel[:,j]
            else:
                pos[:,j] = pos[:,j-1] + dt*vel[:,j]
        return pos
           
    def plot_traj(datatemp,tid,xoffset=0):
        output = datatemp['output']
        target = datatemp['target']
        pos = get_pos(output)
        posT = get_pos(target)
        for j in range(pos.shape[0]):
            plt.plot(xoffset+pos[j,:,0],pos[j,:,1],color=cols[tid[j]],alpha=0.2)
        utid = np.unique(tid)
        for j in range(utid.size):
            idx = np.argwhere(utid[j]==tid)[0]
            plt.scatter(xoffset+posT[idx,-1,0],posT[idx,-1,1],facecolor='None',edgecolor='k',
                        marker='s',zorder=50)
        
    idx = 5
    result = dic
    plt.figure(figsize=(4,6),dpi=150)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    
    plt.subplot(3,2,1)
    plt.title('Example Trial %d'%idx)
    plt.plot(result['stimulus'][idx])
    plt.ylabel('Stimulus')
    
    plt.subplot(3,2,2)
    plt.plot(lc)
    plt.ylabel('Loss')
    plt.xlabel('Training epochs')
    velT = result['target']
    vel = result['output']
    posT = get_pos(velT)
    pos = get_pos(vel)
    score = met.explained_variance_score(posT.reshape(-1,2),pos.reshape(-1,2))
    plt.title('VAF=%.2f'%score)
    
    plt.subplot(3,2,3)
    plt.plot(result['target'][idx],label='Target')
    plt.plot(result['output'][idx],'--',label='Produced')
    plt.ylabel('Output')
    
    plt.subplot(3,2,4)
    model_wFB = run_model(model,params,dataC)
    model_woFB = run_model(model,params,dataC,fb=False)
    tid = model_woFB['tid']
    
    plot_traj(model_wFB,tid)
    plt.text(0,10,'wFB',ha='center')
    
    plot_traj(model_woFB,tid,xoffset=30)
    plt.text(30,10,'woFB',ha='center')
    
    plt.xlim(-8,38)
    plt.ylim(-18,18)
    plt.axis('off')
        
    plt.subplot(3,2,5)
    plt.ylabel('Neurons')
    plt.imshow(result['activity'][idx,:,:].T,aspect='auto')
    plt.xlabel('Time bins')
    
    plt.subplot(3,2,6)
    plt.hist(result['activity'].ravel(),histtype='step',color='k')
    plt.xlabel('Neural activity')
    plt.ylabel('Histogram')
    
    plt.savefig(savname+'summary.svg',bbox_inches='tight')
    plt.savefig(savname+'summary.png',dpi=300,bbox_inches='tight')
    
