# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:11:47 2021

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from Mnn_Core.maf import *


class InteNFire():
    def __init__(self, T = 1e3, num_neurons = 10):
        self.L = 1/20 #ms
        self.Vth = 20
        self.Vres = 0        
        self.Tref = 5 #ms
        self.Vspk = 50
        self.dt = 1e-1 #integration time step (ms)
        self.num_neurons = num_neurons
        self.T = T
        self.synaptic_weight = 0.1 #synaptic weight
        
        #self.T = 10e3 #ms 
    
    def input_spk_time(self, spk_mean, spk_var):
        '''
            generate gamma distributed spike time
            isi_mean = shape*scale (unit: kHz)
            isi_var = shape*scale^2 (unit: kHz)
        '''
        
        isi_mean = 1/spk_mean
        isi_var = np.power(isi_mean,3)*spk_var
        
        scale = isi_var/isi_mean
        shape = isi_mean*isi_mean/isi_var
        
        num_spikes = int(self.T/isi_mean)*5
        num_samples = num_spikes*self.num_neurons
        
        isi = np.random.gamma(shape, scale, num_samples)
        isi = isi.reshape((self.num_neurons, num_spikes ))
        
        spk_time = np.cumsum(isi, axis=1)
        
        return spk_time
    
    def input_synaptic_current(self, t, spk_time):
        '''Convert input spike time to post-synaptic current'''        
        indx1 = t < spk_time
        indx2 = t-self.dt < spk_time
        num_spks = np.sum(indx1 ^ indx2, axis = 1) #^ = xor
        
        return num_spks*self.synaptic_weight
    
    
    def input_gaussian_current(self, mean, std, corr = None):
        ''' Generate gaussian input current '''
        if corr is None:
            input_current = np.random.randn(self.num_neurons)
            input_current = input_current*std*np.sqrt(self.dt) + mean*self.dt
            #input_current = input_current
        else:
            N = len(mean)
            cov = corr*std.reshape(N,1)*std.reshape(1,N)
            input_current = np.random.multivariate_normal(mean*self.dt, cov*self.dt)
            
        return input_current
    
    def run(self, input_mean = 1, input_std = 1, input_corr = None, input_type = 'gaussian', record_v = False, show_message = False):
        '''Simulate integrate and fire neurons'''
        num_timesteps = int(self.T/self.dt)
        
        tref = np.zeros(self.num_neurons) #tracker for refractory period
        v = np.random.rand(self.num_neurons)*self.Vth #initial voltage
        
        SpkTime = [[] for i in range(self.num_neurons)]
        t = np.arange(0, self.T , self.dt)
        if record_v:
            V = np.zeros( (self.num_neurons, num_timesteps) )
        else:            
            V = None
                
        mean, std, rho = input_mean, input_std, input_corr # input current stats (not firing rate, it is current)
        if input_type == 'spike':
            spk_mean = mean/self.synaptic_weight
            spk_var = (std/self.synaptic_weight)**2
            spk_time = self.input_spk_time(spk_mean, spk_var)
        
        start_time = time.time()
        for i in range(num_timesteps):
            
            #evolve forward in time
            if input_type == 'gaussian':
                input_current = self.input_gaussian_current(mean, std)
            elif input_type == 'bivariate_gaussian':
                corr = np.eye(2) + rho*(1-np.eye(2))
                input_current = self.input_gaussian_current(mean, std, corr = corr)
            elif input_type == 'spike':
                input_current = self.input_synaptic_current(i*self.dt, spk_time)
                
            v += -v*self.L*self.dt + input_current
            
            #check state
            is_ref = (tref > 0.0) & (tref < self.Tref)
            is_spike = (v > self.Vth) & ~is_ref
            is_sub = ~(is_ref | is_spike) 
            
            v[is_spike] = self.Vspk            
            v[is_ref] = self.Vres
            
            #update refractory period timer
            tref[is_sub] = 0.0
            tref[is_ref | is_spike] += self.dt
            
            
            if record_v:
                V[:,i] = v            
            
            for k in range(self.num_neurons):
                if is_spike[k]:
                    SpkTime[k].append(i*self.dt)
            
            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
                
        return SpkTime, V, t
    
    def empirical_maf(self, SpkTime):
        isi = []
        for spk_time in SpkTime:            
            spk_time = np.array(spk_time)
            spk_time = spk_time[ spk_time > 500] #remove burn-in time; unit: ms
            isi.extend( list(np.diff(spk_time)) )
        mean_isi = np.mean(isi)
        var_isi = np.var(isi)
        
        #mean firing rate
        mu = 1/mean_isi
        #std firing rate
        sig = np.sqrt( np.power(mu,3)*var_isi )
        return mu, sig

def input_output_anlaysis_2neurons():
    inf = InteNFire(T = 1e3, num_neurons = 2)
    u = np.array([1,1])
    s = np.array([1,1])
    rho = np.linspace(-1,1,11)
    output_rho = rho.copy()
    ntrials = 100
    for i in range(len(rho)):
        spk_count = np.zeros((ntrials,2))
        for j in range(ntrials):
            SpkTime, _, _ = inf.run(input_mean = u, input_std = s, input_corr = rho[i], input_type = 'bivariate_gaussian')
            spk_count[j,0] = len(SpkTime[0])
            spk_count[j,1] = len(SpkTime[1])
            print('Progress: i={}/{}, j={}/{}'.format(i,len(rho),j,ntrials))
        output_rho[i] = np.corrcoef( spk_count[:,0], spk_count[:,1] )[0,1]
    plt.plot(rho, output_rho)
    
    return rho, output_rho
            
    
    
def input_output_anlaysis(input_type):
    inf = InteNFire(T = 1e3, num_neurons = 100) #time unit: ms        
    N = 31
    u = np.linspace(1,3,N)
    s = np.ones(N)*1.5
    
    emp_u = np.zeros(N)
    emp_s = np.zeros(N)
    
    start_time = time.time()
    for i in range(N):
        SpkTime, _, _ = inf.run(input_mean = u[i], input_std = s[i],input_type = input_type, show_message = True)
        emp_u[i], emp_s[i] = inf.empirical_maf(SpkTime)    
                
        progress = (i+1)/N*100
        elapsed_time = (time.time()-start_time)/60
        print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
        
    
    maf = MomentActivation()
    maf_u = maf.mean(u,s)
    maf_s, _ = maf.std(u,s)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)    
    ax1.plot(u, maf_u)
    ax1.plot(u, emp_u, '.')
    ax1.set_xlabel('Mean Input Current')
    ax1.set_ylabel('Mean Firing Rate')
    #ax1.set_title('Mean')
    #cbar1 = fig.colorbar(img1)
    #cbar1.set_label('kHz')#, rotation=270)
    
    ax2 = fig.add_subplot(2,1,2)    
    ax2.plot(u, maf_s)
    ax2.plot(u, emp_s, '.')
    ax2.set_xlabel('Mean Input Current')
    ax2.set_ylabel('Std Firing Rate')
    
    
    return emp_u, emp_s, maf_u, maf_s
    

def simple_demo(input_type):
    inf = InteNFire(T = 1e3, num_neurons = 2) #time unit: ms
    #s = inf.input_spk_time(1,1)
    SpkTime, V, t = inf.run(input_type = input_type, record_v = True, show_message = True)
    plt.plot(t,V[0,:])
    plt.plot(SpkTime[0],[51]*len(SpkTime[0]),'.')        

def simple_demo_two_neurons():
    inf = InteNFire(T = 1e3, num_neurons = 2) #time unit: ms    
    u = np.array([1,1])
    s = np.array([1,1])
    SpkTime, V, t = inf.run(input_mean = u, input_std = s, input_corr = -0.1, input_type = 'bivariate_gaussian', record_v = True, show_message = True)
    plt.plot(t,V[0,:])
    plt.plot(t,V[1,:])
        

if __name__=='__main__':
    input_rho, output_rho = input_output_anlaysis_2neurons()
    #simple_demo_two_neurons()
    #simple_demo(input_type = 'gaussian' )
    #input_output_anlaysis(input_type = 'spike')
    #simple_demo(input_type = 'spike' )
    #runfile('./dev_tools/validate_w_spiking_neuron.py', wdir='./')
        