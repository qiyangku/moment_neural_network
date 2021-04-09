# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:45:19 2020

@author: dell
"""

import matplotlib.pyplot as plt
from Mnn_Core.maf import MomentActivation
import numpy as np

#np.random.seed(1)

class SynfireChain():
    def __init__(self):
        self.N = 100 # neurons per layer
        self.M = 100 # number of layers
        self.W = self.mexi_mat(self.N)        
        self.maf = MomentActivation()
        self.K = 0.0005 #fraction of synapses relative to # of input nodes neurons
        self.r = 0.8 #I-E ratio?
    
    def synthetic_input(self, N, h = 0.15, type = 'mexi'):               
        #x = np.arange(0.0, 2*np.pi, 2*np.pi/N)        
        x = np.arange(-np.pi,np.pi,2*np.pi/N)
        input_mean =  h*np.exp( (np.cos(x)-1)/(0.1))
        input_std = 2*input_mean.copy()
        
        #input_rho = np.random.rand(N,N)
        #input_rho = 0.5*(input_rho + input_rho.T) #make symmetric
        #input_rho = (2*input_rho - 1)
        input_rho = np.zeros((N,N))
        y =  np.cos(2*x)        
        for i in range(N):
            #input_rho[i,i]=1.0
            input_rho[i,:] = np.roll(y,i)
        
        return input_mean, input_std, input_rho
    
    def mexi_mat(self, N, h = 5.4, ie_ratio = 0.5, w = 1):
        "Mexican hat weight matrix"
        
        x = np.arange(0.0, 2*np.pi, 2*np.pi/N)
        y =  h*np.exp( (np.cos(x)-1)/(0.1*w))
        y -= ie_ratio*h*np.exp( (np.cos(x)-1)/(0.5*w))        
        self.mexican_hat = y        
        W = np.zeros((N,N))
        for i in range(N):
            W[i,:] = np.roll(y,i)        
        return W
    
    def summation(self, mean_in, std_in, corr_in):
        
        scaling_factor = np.sqrt(self.K*self.N) #weight scales with the sqrt of in-degree
        
        mean_out = self.W.dot(mean_in)*(1-self.r)/scaling_factor #scale with indegree
        
        C = std_in.reshape(1,self.N)*corr_in*std_in.reshape(self.N,1) #covariance matrix
        
        v = self.W.dot(C).dot(self.W.T)*(1+self.r*self.r) #balanced network; #W is symmetric
        
        std_out = np.sqrt(np.diag(v).copy())/scaling_factor
        
        std_out = np.maximum(1e-16,std_out)
        corr_out = v/std_out.reshape(self.N,1)/std_out.reshape(1,self.N)
        
        for i in range(self.N):
            corr_out[i,i] = 1.0
        
        return mean_out, std_out, corr_out
        
    
    def forward(self, mean_in, std_in, corr_in):
        mean_out, std_out, corr_out = self.summation(mean_in, std_in, corr_in)
        
        u = self.maf.mean( mean_out, std_out)
        s,_ = self.maf.std( mean_out, std_out)
        chi = self.maf.chi( mean_out, std_out)
        rho = chi.reshape(self.N,1)*chi.reshape(1,self.N)*corr_in
        for i in range(self.N):
            rho[i,i]=1.0
        return u, s, rho
    
    def run(self):
        u, s, rho = self.synthetic_input(self.N)       
        U = np.zeros((self.N, self.M))
        S = np.zeros((self.N, self.M))
        R = np.zeros((self.N, self.N, self.M))
        for i in range(self.M):
            #print('i={}, s.shape = {}'.format(i,s.shape)  )
            U[:,i] = u
            S[:,i] = s
            R[:,:,i] = rho
            u, s, rho = self.forward(u, s, rho)
            
        return U, S, R
    
    def para_sweep(self):
        '''Do a parameter sweep over the weight space'''
        WE = np.linspace(0.0,10.0,21)
        ie_ratio = np.linspace(0.0,1.0,20)
        
        U = np.zeros( (len(WE), len(ie_ratio), self.N) )
        S = np.zeros( (len(WE), len(ie_ratio), self.N) )
        R = np.zeros( (len(WE), len(ie_ratio), self.N, self.N) )
        
        for i in range(len(WE)):
            for j in range(len(ie_ratio)):
                self.W = self.mexi_mat(self.N, h = WE[i], ie_ratio = ie_ratio[j])     
                u,s,r = self.run()
                U[i,j,:] = u[:,-1]
                S[i,j,:] = s[:,-1]
                R[i,j,:,:] = r[:,:,-1]
            print('WE={}, ie_ratio={}'.format(WE[i],ie_ratio[j]))
        return WE, ie_ratio, U, S, R
        
    
def plot_rho(model):        
    num_layers = np.arange(len(model.layers)+1)
    temp = input_rho[0].detach().numpy()
    np.fill_diagonal(temp,np.nan)    
    mean_rho = [np.nanmean(temp)]
    std_rho = [np.nanstd(temp)]
    for layer in model.layers:
        rho = layer.output[2].detach().numpy()[0,:,:]           
        #rho = np.mean(rho, axis = 0)
        #rho -= np.eye(rho.shape[0])
        np.fill_diagonal(rho,np.nan)
        mean_rho.append(np.nanmean(rho))
        std_rho.append(np.nanstd(rho))
    
    #print(mean_rho)
    #plt.plot(num_layers,mean_rho,num_layers,std_rho)
    plt.errorbar(num_layers,mean_rho, yerr = std_rho)
    plt.xlabel('Layer index')
    plt.ylabel('Correlation Coefficient')
    #print(rho.shape)
    #plt.imshow(rho)# cmap='hot', interpolation='nearest')
    plt.show()
    
def plot_mean(model):
    mu = np.zeros((100,len(model.layers)))
    i = 0
    for layer in model.layers:        
        mu[:,i] = layer.output[0].detach().numpy()[0,:]
        i += 1
    img = plt.imshow(mu,aspect='auto')
    plt.xlabel('Layer index')
    plt.ylabel('Neuron index')
    plt.colorbar(img)
    plt.show()

def plot_rho_grid(model):
    #rho = np.zeros( (len(model.layers),50) )
    i = 0
    j = 0
    fig, axes = plt.subplots(3, 5)
    for layer in model.layers:        
        #rho[i,:,:]
        rho = layer.output[2].detach().numpy()[0,:,:]
        np.fill_diagonal(rho,np.nan)
        i += 1
        if i % 10 == 0:            
            img = axes.flatten()[j].imshow(rho)#, aspect='auto')                        
            axes.flatten()[j].set_title('Layer {}'.format(i))
            axes.flatten()[j].set_xticks([])
            axes.flatten()[j].set_yticks([])
            j += 1
    plt.colorbar(img)
    print('Corr. coef. range: [{},{}]'.format(np.nanmin(rho),np.nanmax(rho)))
    plt.show()

#%% 
if __name__ == "__main__":    
    sfc = SynfireChain()
    dat = sfc.para_sweep()
#%%    
    m = dat[4].shape[0]
    n = dat[4].shape[1]
    num_neurons = dat[4].shape[3]
    
    rho_max = np.zeros((m,n))
    std_max = rho_max.copy()
    
    mean_max = np.zeros((m,n))
    
    for i in range(m):
        for j in range(n):
            rho = dat[4][i,j,:,:] - np.eye(num_neurons)
            rho_max[i,j] = np.max(np.abs(rho))
            std_max[i,j] = np.max(dat[3][i,j,:])            
            mean_max[i,j] = np.max(dat[2][i,j,:])
            
            
    fig = plt.figure()        
    
    ax1 = fig.add_subplot(311)
    pos = ax1.imshow(mean_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    ax1.set_ylabel('Excitatory weight')
    fig.colorbar(pos, ax=ax1)
    ax1.set_title('Max mean')
    
    ax3 = fig.add_subplot(312)
    pos3 = ax3.imshow(std_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    ax3.set_ylabel('Excitatory weight')
    fig.colorbar(pos3, ax=ax3)
    ax3.set_title('Max std')
    
    
    ax2 = fig.add_subplot(313)
    pos2 = ax2.imshow(rho_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    fig.colorbar(pos2, ax=ax2)
    ax2.set_ylabel('Excitatory weight')
    ax2.set_xlabel('I-E weight ratio')
    ax2.set_title('Max corr. coef. (off-diagonal)')
    
    #======================='
    
    
    # U, S, R = sfc.run()
    
    # fig = plt.figure()        

    # ax1 = fig.add_subplot(211)
    # ax1.imshow(U,extent = [0,sfc.M,0,sfc.N],origin='lower')#,interpolation="lanczos")#
    # ax1.set_xlabel('Layers')
    # ax1.set_ylabel('Neurons')
    # ax1.set_title('Mean')            
    # ax1.set_aspect('auto')
    
    # ax2 = fig.add_subplot(212)
    # ax2.imshow(S,extent = [0,sfc.M,0,sfc.N],origin='lower')#,interpolation="lanczos")#extent = [ubar[0],ubar[-1],sbar[0],sbar[-1]],
    # ax2.set_xlabel('Layers')
    # ax2.set_ylabel('Neuron')   
    # ax2.set_title('Std')
    # ax2.set_aspect('auto')
    
    # fig.subplots_adjust(hspace = 0.6)
    
    # fig2 = plt.figure()
    
    # for i in range(9):
    #     ax = fig2.add_subplot('33{}'.format(i+1))
    #     #layer_num = int(i*10)
    #     layer_num = i
    #     ax.imshow(R[:,:,layer_num],vmin=-1,vmax=1)# - np.eye(sfc.N))#,interpolation="lanczos")#extent = [ubar[0],ubar[-1],sbar[0],sbar[-1]],
    #     ax.axis('off')
    #     #ax2.set_xlabel('Layers')
    #     #ax2.set_ylabel('Neurons')   
    #     ax.set_title('Layer {}'.format(layer_num))
   
        
    