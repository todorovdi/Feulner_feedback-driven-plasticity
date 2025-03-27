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
        self.n_neurons = n_neurons #400
        self.n_inputs = n_inputs #3
        self.n_outputs = n_outputs #2
        self.alpha = alpha #0.20
        self.dt = dt #0.01
        
        self.rnn = nn.RNN(n_inputs, n_neurons, num_layers=2, bias=False) #(3,400,2)
        self.output = nn.Linear(n_neurons, n_outputs) #(400,2)
        self.feedback = nn.Linear(n_outputs, n_neurons) #(2,400)
        self.dtype = dtype
        
        self.fwd_delay = fwd_delay # delay from neural activity to movement execution, 2
        self.fb_delay = fb_delay # feedback delay 10
        self.delay = fwd_delay+fb_delay # total delay, 10+2 = 12
        

        self.biolearning = biolearning #choosing biolearning 

        # if biolearning is true, then the recurrent weights (rnn.weight_hh_l0) are not directly trained by backpropagation
        if biolearning:
            self.rnn.weight_hh_l0.requires_grad = False

        
        self.noisein = noisein # amplitute of noise added to input, 0
        self.noiseout = noiseout  # amplitude of noise added to output velocity, 0 
        self.noise_kernel_lenk = noise_kernel_size # kernel size 5. Convolution used?
        
        # choosing nonlienearity to be used 
        if nonlin=='relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlin=='tanh':
            self.nonlin = torch.nn.Tanh()
        elif nonlin=='sigmoid':
            self.nonlin = torch.nn.Sigmoid()

        # mask for feedback weights, zeros or ones
        # depends on fb_sparsity, determines sparsity of feedback connections
        # it is a non trainable parameter is grad is false
        self.mask = nn.Linear(n_outputs, n_neurons, bias=False) # (2,400) 
        self.mask.weight = nn.Parameter((torch.rand(n_neurons, n_outputs) < fb_sparsity).float()).type(dtype)
        self.mask.weight.requires_grad = False

        # mask for recurrent weights, zeros or ones
        # depends on rec_sparsity, determines sparsity of recurrent connections
        # it is a non trainable parameter as grad is false
        self.mask2 = nn.Linear(n_neurons, n_neurons, bias=False) 
        self.mask2.weight = nn.Parameter((torch.rand(n_neurons, n_neurons) < rec_sparsity).float()).type(dtype)
        self.mask2.weight.requires_grad = False

    def init_hidden(self):
        # if batch size 20, neurons 400 then it gives a 
        # 20 x 400 shape tensor. Each row has 400 values 
        # between (-0.1, 0.1). Used to initialize the hidden state
        return ((torch.rand(self.batch_size, self.n_neurons)-0.5)*0.2).type(self.dtype)
    
    # ONE SIMULATION STEP
    def f_step(self,xin,x1,r1,v1fb,v1,vel_pert_ext,popto):
        '''
        simulates one time bin step (a sub-part of a trial)
        Meaning in a trial if there are 300 time steps, this function
        simulates 1 time step until it iterates over 300 time steps

        in all of the arguments shape[0] = batch_size

        xin  input   .shape torch.Size([20, 7])
        x1   current hidden net input (pre-activation)   torch.Size([20, 400])
        r1   current hidden activation   torch.Size([20, 400])
        v1fb feedback (2D coordintes of the error torch.Size([20, 2])
        v1   current velocity(?)  shape torch.Size([20, 2])
        vel_per_ext external perturbation applied to velocity shape (20,2)
        pin  pertrubation of velocity in current timebin shape = (20, 2)
        popto perturbation of the neural activity (pre-synaptic) shape torch.Size([20, 400])

        returns x1,r1,v1
            x1 shape is (20, 400)
            r1.shape torch.Size([20, 400])
            v1  -- will be saved in poserr, shape=torch.Size([20, 2])
        '''

        # xin is the input from the "stiumulus" dataset
        # xin col 0,1 = (endpoint_x - startpoint_x), (endpoint_y - startpoint_y)
        # xin col 2 = stim range flag. Only active until go_on signal occurs
        # xin col 3,4 = the "perfect" velocity_x and velocity_y to reach endpoint
        # xin col 5,6 = the "perfect" trajectory of x,y coordinate moves to reach endpoint
          
        # dim [0-1] = beforfe stim_on it is zero, after stim_on it is end_point-start_point
        # dim 2 = before go_on[k]-go_to_peak = 0, after = stim_range (hold signal)

        perceived_signals = xin[:,:3]

        # FORMULA:
        # 1.    - sparse rnn hidden weights with mask2 matrix, then transpose it
        #       - matmul the transposed hidden weight matrix with current hidden activation

        # 2.    - Transpose input hidden weight matrix
        #       - Matmul hiddden weight matrix with first 3 cols of "stimulus"

        # 3.    - Sparse the feedback weight matrix with mask1 matrix
        #       - Matmul the sparsed feedback weight matrix with current velocity error matrix

        # 4.    - (1+2+3+ bias of feedback + velocity perturbation - current hidden net input)
        # 5.    - Multiply 4 with alpha, then add current hidden net input 
        x1 = x1 + self.alpha*(-x1 + r1 @ (self.mask2.weight*self.rnn.weight_hh_l0).T
                                  + perceived_signals @ self.rnn.weight_ih_l0.T
                                  + v1fb @ (self.mask.weight*self.feedback.weight).T 
                                  + self.feedback.bias.T
                                  + popto
                              )
        
        # 6.    - Apply nonlinearity to (5)
        r1 = self.nonlin(x1)

        # 7.    - Decode the velocity using the output layer, add velocity perturbation to decoded velocity
        vt = self.output(r1) + vel_pert_ext # vt.shape torch.Size([20, 2])


        # xin[:,3:5] = last two dimensions of target mvt (so velolcity)
        # v1 is the accumulated (signed) mismatch between the target and the actual velocity

        # 8. Calculate current velocity error, multiply with time step, add to 7 ("produced" velocity)
        v1_upd = v1 + self.dt*(xin[:,3:5]-vt) # will be added to poserr 

        return x1,r1,v1_upd
    
    # BIOLOGICALLY PLAUSIBLE LEARNING RULE
    def dW(self,pre,post,lr,inp,fb,presum):
        with torch.no_grad():
            return self.alpha*0.1*(
                + lr*self.mask2.weight*torch.outer(fb@(self.mask.weight*self.feedback.weight).T,presum)
                )
                    
    # GET VELOCITY OUTPUT (NOT ERROR)
    def get_output(self,testl1):

        # is testl1 of (shape 20x400)?
        if self.noiseout>0:
            return self.output(testl1)+self.output(testl1)*self.create_noise(testl1)
        else:
            return self.output(testl1)
        
    # SIMULATE MOTOR NOISE (scales with velocity output)
    def create_noise(self,testl1):
        '''
        testl1 is a 2D tensor, it is not actually used, only its shape is used
        '''
        # time varying noise
        tmp = self.noiseout*torch.randn(testl1.shape[1],testl1.shape[0],2)
        tmp = tmp.permute(0,2,1)
        lenk = self.noise_kernel_lenk
        kernel = torch.ones(2,tmp.shape[1],lenk)/lenk
        padding = lenk // 2 + lenk % 2

        # noise smoothing with 1D convolution
        noise = torch.nn.functional.conv1d(tmp, kernel, padding=padding)[:,:,:testl1.shape[0]]
        noise = noise.permute(2,0,1)*np.sqrt(lenk)  
        return noise
    
    # RUN MODEL
    def forward(self, X, Xpert, lr=1e-3, popto=None):
        '''
        X is "stimulus" or "input", 
        Xpert is perturbation
        X.shape     = 300x20x7  = time x batch x input_dim
        Xpert.shape = 300x20x2 = velocity pertrubations 
        popto = perturbation of the neural activty (pre synaptic), used in adaptation_learning.py only

        each forward run is simulation of the entire single trial, meaning over 300 times steps 
        '''

        self.batch_size = X.size(1) # 20

        # init the hidden state with random 
        hidden0 = self.init_hidden() # shape = batch x n_neurons

        x1 = hidden0         # random values between (-0.1, 0.1), shape is (20 x 400) 
        r1 = self.nonlin(x1) # activation function, likely relu shape = batch x n_neurons (20x400)
        v1 = self.output(r1) # outputs, shape = batch x n_output (=20 x 2)

        hidden1 = []
        poserr = []
        presum = r1
        dw_acc = torch.zeros(self.rnn.weight_hh_l0.shape) # 400 x 400

        ######## SKIP FOR NOW
        if popto is None:
            popto = torch.zeros((X.shape[0],X.shape[1],r1.shape[1])) # 300 x batch x 400

        # prerun simulation (until 'delay' is reached)
        for j in range(self.fwd_delay+1):
            x1,r1,v1 = self.f_step(X[0],x1,r1,
                                    v1*0,v1,Xpert[0],
                                    popto[0]) 
            # v1 is the accumulated (signed) mismatch between the target and the actual velocity
            hidden1.append(r1)
            poserr.append(v1)

        # generate noise on the input level (input and output)
        X = X+ self.noisein*torch.randn(X.shape[1],X.shape[2])[None,:,:]

        # noise applied to the output velocity
        if self.noiseout>0:
            noise = self.create_noise(Xpert)
        else:
            noise = torch.zeros(Xpert.shape)

        # now real time simulation
        for j in range(X.size(0)):
            tpl = X[j],x1,r1
            d = dict(v1=v1, 
                 vel_pert_ext=Xpert[j]+noise[j]*self.output(r1), 
                 popto= popto[j])
            if (self.fb_delay<0) | (j<=self.fb_delay):
                x1,r1,v1 = self.f_step(*tpl,
                                    v1fb=v1*0, **d)
            else:
                feedback_val_curf = poserr[j-self.delay]
                x1,r1,v1 = self.f_step(*tpl,
                                    v1fb=feedback_val_curf, **d)

                if self.biolearning and j%5:
                    dw_acc += self.dW(x1[0],r1[0],lr,X[j,0],
                                      poserr[j-self.delay][0],presum[0]).detach()
            if self.biolearning:
                presum += r1.detach()
            # v1 is the accumulated (signed) mismatch between the target and the actual velocity
            hidden1.append(r1)
            poserr.append(v1)
        # truncate very first (probably because it was random.. but we ran warmup)
        hidden1 = torch.stack(hidden1[1:])
        poserr = torch.stack(poserr[1:])
        if self.biolearning:
            with torch.no_grad():
                self.rnn.weight_hh_l0 += dw_acc

        # we later do
        # output,hidden = model(stimulus[epoch],pert[epoch]) # runs forward
        # loss_train = criterion(output, output*0).mean() # we compare output (errors per trial) with 0


        if self.fwd_delay==0:
            return poserr, hidden1
        elif self.fwd_delay>0:
            return poserr[(self.fwd_delay):], hidden1[:(-self.fwd_delay)]


def run_model(model,params,data,fb=True,dopert=0):
    """
    Runs the specified model on the provided data with optional feedback and perturbation.
    Parameters:
    model (torch.nn.Module): The neural network model to be run.
    params (dict): Dictionary containing parameters for the model and perturbation.
    data (dict): Dictionary containing the test data with keys 'test_set', 'target', 'stimulus', and optionally 'tids'.
    fb (bool, optional): Flag to enable or disable feedback. Defaults to True.
    dopert (int, optional): Type of perturbation to apply. 0 for no perturbation, 1 for random pushes. Defaults to 0.
    Returns:
    dict: A dictionary containing the following keys:
        - 'target': The target data from the test set.
        - 'stimulus': The stimulus data from the test set.
        - 'error': The error between the model output and the target.
        - 'peak_speed': The peak speed data from the test set.
        - 'pert': The perturbation applied to the model.
        - 'activity': The activity of the model's hidden layer.
        - 'output': The output of the model.
        - 'tid' (optional): The trial IDs from the test set, if present.
    """
    '''only used in test()'''

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
    model_wFB  = run_model(model,params,dataC)
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
    
