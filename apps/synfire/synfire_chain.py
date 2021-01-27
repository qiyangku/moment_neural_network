# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:45:19 2020

@author: dell
"""

import matplotlib.pyplot as plt
import time
from Mnn_Core.mnn_pytorch import *
import numpy as np
import torch
import torch.fft
from apps.synfire.conv1d_fft import Mnn_Conv1d_fft

#np.random.seed(1)

class SynfireChain():
    def __init__(self, num_neurons, weight_type = 'vonmises'):
        self.num_neurons = num_neurons # neurons per layer                
        self.weight_type = weight_type
        self.linear = Mnn_Linear_Corr(num_neurons, num_neurons)
        self.linear.weight.data = self.mexi_mat()
        self.conv1d = Mnn_Conv1d_fft(num_neurons)
        self.conv1d.weight.data = self.mexi_mat()[0,:]
        
    
    def mexi_mat(self, we = 5, wi = 1.5, de = 0.5, di = 1.5):
        "Mexican hat weight matrix"
        x = torch.arange(0.0, 2*np.pi, 2*np.pi/self.num_neurons)
        if self.weight_type == 'vonmises':            
            func = lambda x, d: torch.exp( (torch.cos(x)-1)/d/d)
        elif self.weight_type == 'gaussian':
            x =  (x + np.pi) % (2 * np.pi) - np.pi #map (0,2pi) to (0,pi)U(-pi,0)
            func = lambda x, d: torch.exp(-x*x/d/d/2 )
        else:
            pass
        
                
        y = we*func(x,de) - wi*func(x,di)        
        W = torch.zeros(self.num_neurons,self.num_neurons)
        for i in range(self.num_neurons):
            W[i,:] = y.roll(i)
        return W
    
    def forward_fft(self, u, s, rho, u_ext, s_ext, timesteps = 100):
        '''Experimental fft-based forward pass'''
        with torch.no_grad():
            U = torch.zeros(self.num_neurons ,timesteps)
            S = torch.zeros(self.num_neurons ,timesteps)
            R = torch.zeros(self.num_neurons , self.num_neurons, timesteps)
            
            #w = self.linear.weight.data[0,:]            
            #wfft = torch.fft.fft(w)                        
            #print('Shape of w: ',w.shape) #make sure the shape is the same
            #print('Shape of u: ',u.shape)
            
            
            for i in range(timesteps):
                #u, s, rho = self.linear.forward(u, s, rho)
                u, s, rho = self.conv1d.forward(u, s, rho)
                
                # #>>>> FFT-based summation layer                
                # u = torch.real(torch.fft.ifft(wfft*torch.fft.fft(u)))
                
                # C = rho*s.view(self.num_neurons,1)*s.view(1,self.num_neurons)
                
                # Cfft = torch.fft.fft( torch.fft.fft(C, dim = 0), dim = 1)
                # #Cfft = torch.fft.fftn(C, dim = (0,1)) #somehow slightly slower than 1d fft by 5%                
                # Cfft = wfft.view(self.num_neurons,1)*Cfft*wfft.view(1,self.num_neurons)                
                # C = torch.real(torch.fft.ifft(torch.fft.ifft(Cfft, dim = 0), dim =1))
                # # = torch.real(torch.fft.ifftn( Cfft , dim = (0,1)))
                
                # s = torch.sqrt(torch.diag(C))
                # rho = C/s.view(self.num_neurons,1)/s.view(1,self.num_neurons)
                
                # #<<<<
                
                u += u_ext
                s = torch.sqrt( s*s + s_ext*s_ext)
                
                u_activated = Mnn_Activate_Mean.apply(u, s)
                s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
                corr_activated = Mnn_Activate_Corr.apply( rho , u, s, u_activated, s_activated)
                
                u = u_activated.clone()
                s = s_activated.clone()
                rho = corr_activated.clone()
                
                U[:,i] = u
                S[:,i] = s
                R[:,:,i] = rho
        return U, S, R
    
    def forward(self, u, s, rho, u_ext, s_ext, timesteps = 100):
        with torch.no_grad():                        
            U = torch.zeros(self.num_neurons ,timesteps)
            S = torch.zeros(self.num_neurons ,timesteps)
            R = torch.zeros(self.num_neurons , self.num_neurons, timesteps)
            
            for i in range(timesteps):
                u, s, rho = self.linear.forward(u, s, rho)
                u += u_ext
                s = torch.sqrt( s*s + s_ext*s_ext)
                
                u_activated = Mnn_Activate_Mean.apply(u, s)
                s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
                corr_activated = Mnn_Activate_Corr.apply( rho , u, s, u_activated, s_activated)
                
                u = u_activated.clone()
                s = s_activated.clone()
                rho = corr_activated.clone()
                
                U[:,i] = u
                S[:,i] = s
                R[:,:,i] = rho
        return U, S, R
           
    def run(self, timesteps, use_fft = False):
        x = torch.arange(0.0, 2*np.pi, 2*np.pi/self.num_neurons).roll( int(self.num_neurons/2) )
        u = torch.exp( torch.cos(x)-1 ).squeeze(0)
        s = torch.exp( torch.cos(x)-1 ).squeeze(0)
        rho = torch.eye(len(x) ).squeeze(0)
        
        u_ext = 0.3
        s_ext = 2
        if use_fft:
            U, S, R = self.forward_fft(u, s, rho, u_ext, s_ext, timesteps = timesteps)
        else:
            U, S, R = self.forward(u, s, rho, u_ext, s_ext, timesteps = timesteps)
            
        return U, S, R, x
    
    def para_sweep(self, de = 0.5, di = 1.5):
        '''Do a parameter sweep over the weight space'''
        WE = np.linspace(0.0,10.0, 21)
        ie_ratio = np.linspace(0.0,1.0, 20)
        timesteps = 100
        
        U = np.zeros( (len(WE), len(ie_ratio), self.num_neurons, timesteps) )
        S = np.zeros( (len(WE), len(ie_ratio), self.num_neurons, timesteps) )
        R = np.zeros( (len(WE), len(ie_ratio), self.num_neurons, self.num_neurons, timesteps) )
        
        t0 = time.time()
        
        for i in range(len(WE)):
            for j in range(len(ie_ratio)):
                self.linear.weight.data = self.mexi_mat(we = WE[i], wi = WE[i]*ie_ratio[j], de = de, di = di)
                u,s,r,x = self.run(timesteps)
                U[i,j,:,:] = u
                S[i,j,:,:] = s
                R[i,j,:,:,:] = r
            print('WE={}, ie_ratio={}, Time Elapsed ={}'.format(WE[i],ie_ratio[j], int(time.time()-t0)))
        return WE, ie_ratio, U, S, R, x
    
    def para_sweep_extra(self):
        '''Sweep the entire parameter space: we, wi, di; fix de'''
        de = 0.5
        di = de*np.linspace(1,5,11)
        for k in range(len(di)):
            WE, ie_ratio, U, S, R, x = self.para_sweep(de=de,di=di[k])
            
            # dat = {}
            # dat['we'] = WE
            # dat['ie_ratio'] = ie_ratio
            # dat['U'] = U
            # dat['S'] = S
            # dat['R'] = R
            # dat['x'] = x
            # dat['di'] = di[k]
            
            filename = str(k).zfill(3)
            path = './data/synfire_sweep_{}/'.format(self.weight_type)
            #np.save(path+filename, dat, allow_pickle=True)
            np.savez_compressed(path+filename, we=WE, ie_ratio=ie_ratio, U=U, S=S, R=R, x=x, di=di[k], de=de)
            print('Overall progress: {}/{}'.format(k,len(di)))
        return
            
            
            
            

class SynfireVisualize():
    @staticmethod
    def plot_summary(U,S,R,x):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)  
        #ax1.imshow(U, aspect = 'auto')
        ax1.plot(x,U[:,-1])
        ax2 = fig.add_subplot(2,2,2)  
        #ax2.imshow(S, aspect = 'auto')
        ax2.plot(x,S[:,-1])
        ax3 = fig.add_subplot(2,2,3)  
        #ax3.imshow(w_fitted, cmap = 'bwr', vmax = torch.max(y_fit), vmin = -torch.max(y_fit))
        #ax3.plot(x, y_fit)
        ax4 = fig.add_subplot(2,2,4)  
        ax4.imshow(R[:,:,-1], cmap = 'bwr', vmax = 1, vmin = -1)
    
    @staticmethod    
    def plot_corr(R):
        fig = plt.figure()        
        n = 0
        M, N = 10, 10        
        for i in range(M):
            for j in range(N):
                n += 1
                ax = fig.add_subplot(M,N,n)
                ax.imshow(R[2*i,2*j,:,:,-1], cmap='bwr', vmin = -1, vmax = 1)
                ax.axis('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                #label_mu = r'$\bar{\mu}=$'+'{:.1f}'.format( dat['u'][i] )
                #label_sig = r'$\bar{\sigma}=$'+'{:.1f}'.format(dat['s'][j])
                #if i == 0:
                #    ax.set_title(label_sig)
                #if j == 0:
                #    ax.set_ylabel(label_mu)
    
    @staticmethod    
    def plot_stats(U,S,R, WE, ie_ratio):
        figsize = (9,9)
        fig_mean = plt.figure(figsize = figsize) #inches at 100 dpi
        fig_std = plt.figure(figsize = figsize)
        fig_corr = plt.figure(figsize = figsize)
        fig_cov = plt.figure(figsize = figsize)
        n = 0
        M, N = 10, 10
        
        ax = [None for i in range(4)]
        
        for i in range(M):
            for j in range(N):
                n += 1
                #mean
                ax[0] = fig_mean.add_subplot(M,N,n)
                ax[0].imshow(U[2*i,2*j,:,:], vmin = 0, vmax = 0.15)                                
                
                #std
                ax[1] = fig_std.add_subplot(M,N,n)
                ax[1].imshow(S[2*i,2*j,:,:], vmin = 0, vmax = 0.3)                
                
                #corr
                ax[2] = fig_corr.add_subplot(M,N,n)
                ax[2].imshow(R[2*i,2*j,:,:,-1], cmap='bwr', vmin = -1, vmax = 1)                
                
                ax[3] = fig_cov.add_subplot(M,N,n)
                s = S[2*i,2*j,:,-1].reshape(S.shape[2],1)             
                ax[3].imshow(R[2*i,2*j,:,:,-1]*s*s.T, cmap='bwr')                
                
                for k in range(4):
                    ax[k].set_xticks([])
                    ax[k].set_yticks([])
                    label_we = r'$w_E=$'+'{:.1f}'.format( WE[2*i] )
                    label_wi = r'$r=$'+'{:.1f}'.format( ie_ratio[2*j])
                    if i == 0:
                        ax[k].set_title(label_wi)
                    if j == 0:
                        ax[k].set_ylabel(label_we)
        
        fig_mean.tight_layout()
        fig_std.tight_layout()
        fig_corr.tight_layout()
        fig_cov.tight_layout()
        return        

    @staticmethod    
    def load_n_plot(indx, weight_type = 'vonmises'):
        filename = str(indx).zfill(3)
        path = './data/synfire_sweep_{}/'.format(weight_type)
        dat = np.load(path+filename+'.npz')
        SynfireVisualize.plot_stats(dat['U'],dat['S'],dat['R'], dat['we'], dat['ie_ratio'])
        


#%% 
if __name__ == "__main__":    
    sfc = SynfireChain(100, weight_type = 'gaussian')
    t0 = time.time()
    U, S, R, x = sfc.run(100, use_fft = True)
    print('Elapsed time for fft: ', time.time()-t0)
    U2, S2, R2, x = sfc.run(100, use_fft = False)
    print('Elapsed time for non-fft: ', time.time()-t0)
    
    plt.plot(U[:,-1])
    plt.plot(U2[:,-1],'.')
    #plt.plot((U[:,-1]-U2[:,-1])/U2[:,-1])
    
    #WE, ie_ratio, U, S, R, x = sfc.para_sweep()
    #sfc.para_sweep_extra()
    
    #plot_stuff(U,S,R, x)
    
    #runfile('./apps/synfire/synfire_chain.py', wdir='./')
    
    #dat = sfc.para_sweep()
# #%%    
#     m = dat[4].shape[0]
#     n = dat[4].shape[1]
#     num_neurons = dat[4].shape[3]
    
#     rho_max = np.zeros((m,n))
#     std_max = rho_max.copy()
    
#     mean_max = np.zeros((m,n))
    
#     for i in range(m):
#         for j in range(n):
#             rho = dat[4][i,j,:,:] - np.eye(num_neurons)
#             rho_max[i,j] = np.max(np.abs(rho))
#             std_max[i,j] = np.max(dat[3][i,j,:])            
#             mean_max[i,j] = np.max(dat[2][i,j,:])
            
            
#     fig = plt.figure()        
    
#     ax1 = fig.add_subplot(311)
#     pos = ax1.imshow(mean_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
#     ax1.set_ylabel('Excitatory weight')
#     fig.colorbar(pos, ax=ax1)
#     ax1.set_title('Max mean')
    
#     ax3 = fig.add_subplot(312)
#     pos3 = ax3.imshow(std_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
#     ax3.set_ylabel('Excitatory weight')
#     fig.colorbar(pos3, ax=ax3)
#     ax3.set_title('Max std')
    
    
    # ax2 = fig.add_subplot(313)
    # pos2 = ax2.imshow(rho_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    # fig.colorbar(pos2, ax=ax2)
    # ax2.set_ylabel('Excitatory weight')
    # ax2.set_xlabel('I-E weight ratio')
    # ax2.set_title('Max corr. coef. (off-diagonal)')
    