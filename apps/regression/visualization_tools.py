# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:26:28 2020

@author: dell
"""
import torch
import numpy as np
from apps.regression.data import prod_normal
import matplotlib.pyplot as plt

class VisualizationTools():
    @staticmethod
    def plot_grid_map(model, rho = None, with_corr = True):        
        mean = torch.linspace(0,1,10)
        model.eval()
        
        if not rho:
            rho = 0.0
        
        input_corr = (torch.eye(2) + (1-torch.eye(2))*rho).unsqueeze(0).repeat(100,1,1)
        
        fig1 = plt.figure()            
        ax1 = fig1.add_subplot(111)
        
        for i in range(len(mean)):
            input_std = torch.linspace(1,3,100)
            input_mean = torch.ones(input_std.shape)*mean[i]
            
            target_mean, target_std = prod_normal( input_mean, input_std, torch.zeros(input_mean.shape), torch.ones(input_mean.shape), torch.ones(input_mean.shape)*input_corr[0,0,1] )
            
            input_mean2 = torch.zeros(100,2)
            input_mean2[:,0] = input_mean
            input_std2 = torch.ones(100,2)
            input_std2[:,0] = input_std
            
            if with_corr:
                output_mean, output_std, output_corr = model.forward(input_mean2, input_std2, input_corr)            
            else:
                output_mean, output_std = model.forward(input_mean2, input_std2)            
            #target_mean = target_mean*target_affine[0] + target_affine[1]
            #target_std = target_std*target_affine[2] + target_affine[3]
            target_mean, target_std = model.target_transform( target_mean, target_std )         
            
            ax1.plot(target_mean, target_std,'b', alpha=0.5)
            ax1.plot(output_mean.detach().numpy(), output_std.detach().numpy(), 'b')
        
        sigma = torch.linspace(1,3,10)
        
        for i in range(len(sigma)):
            input_mean = torch.linspace(0,1,100)
            input_std = torch.ones(input_mean.shape)*sigma[i]
                        
            target_mean, target_std = prod_normal( input_mean, input_std, torch.zeros(input_mean.shape), torch.ones(input_mean.shape), torch.ones(input_mean.shape)*input_corr[0,0,1] )
            
            input_mean2 = torch.zeros(100,2)
            input_mean2[:,0] = input_mean
            input_std2 = torch.ones(100,2)
            input_std2[:,0] = input_std
            
            if with_corr:
                output_mean, output_std, output_corr = model.forward(input_mean2, input_std2, input_corr)            
            else:
                output_mean, output_std = model.forward(input_mean2, input_std2) 
            
            target_mean, target_std = model.target_transform( target_mean, target_std )  
            
            ax1.plot(target_mean, target_std,'r', alpha=0.4)
            ax1.plot(output_mean.detach().numpy(), output_std.detach().numpy(), 'r')
        
        ax1.set_xlabel('Output mean')
        ax1.set_ylabel('Output std')
        ax1.set_title(r'$\rho=${}'.format(rho))
        
        return fig1
    
    @staticmethod
    def plot_corr(model):
        fig_corr = plt.figure()    
        fig_chist = plt.figure()        
        i=0
        for L in model.layers:
            i+=1
            #if i > len(model.layers)-1:
            #    break
            ax1 = fig_corr.add_subplot(3,4,i)
            img = ax1.imshow(L.corr[0].detach().numpy(),vmax=1,vmin=-1,cmap = 'bwr')
            ax1.axis('off')
            ax1.set_title('Layer {}'.format(i))
            
            ax2 = fig_chist.add_subplot(3,4,i)
            img2 = ax2.hist(L.corr[0].detach().numpy().flatten(),np.linspace(-1,1,31))
            ax2.set_title('Layer {}'.format(i))
            
        fig_corr.colorbar(img)
        return fig_corr
                
    @staticmethod
    def plot_weight(model):
        fig_weight = plt.figure()
        fig_whist = plt.figure()
        i=0
        for L in model.layers:
            i+=1
            if i > len(model.layers)-1:
                break
            
            
            #scale weight based on bn_mean (what about bn_std?)
            scale = L.bn_mean.weight/torch.sqrt(L.bn_mean.running_var)
            w = L.linear.weight*(scale.unsqueeze(0).T)
            w = w.detach().numpy()
            
            ax1 = fig_weight.add_subplot(3,4,i)        
            img = ax1.imshow(w,vmax=50,vmin=-50, cmap = 'bwr')                
            ax1.axis('off')
            ax1.set_title('Layer {}'.format(i))
            
            ax2 = fig_whist.add_subplot(3,4,i)
            img2 = ax2.hist(w.flatten(),np.linspace(-50,50,31))
            ax2.set_title('Layer {}'.format(i))
            
        fig_weight.colorbar(img)
        return fig_weight
    
    @staticmethod
    def plot_bias(model):
        fig_bias = plt.figure()        
        i=0
        for L in model.layers:
            i+=1
            if i > len(model.layers)-1:
                break
            
            
            #scale weight based on bn_mean (what about bn_std?)
            bias = L.bn_mean.bias - L.bn_mean.running_mean*L.bn_mean.weight/torch.sqrt(L.bn_mean.running_var)
            bias = bias.detach().numpy()
            
            ax1 = fig_bias.add_subplot(3,4,i)        
            img = ax1.bar( np.arange(bias.size) , bias )                
            #ax1.axis('off')
            ax1.set_title('Layer {}'.format(i))
            ax1.set_xlabel('Neuron index')
            ax1.set_ylabel('Ext. Input Current')
        
        return fig_bias
    
    @staticmethod
    def plot_rnn(model, i = 0):
        fig = plt.figure()
        L = model.recurrent_layer
        #i = 3 #sample index
        
        ax1 = fig.add_subplot(2,2,1)    
        u = L.output[0][:,i,:]
        img1 = ax1.imshow(u, origin = 'lower', aspect = 'auto')
        ax1.set_xlabel('Neuron index')
        ax1.set_ylabel('Time steps')
        ax1.set_title('Mean')
        cbar1 = fig.colorbar(img1)
        cbar1.set_label('kHz')#, rotation=270)
        
        ax2 = fig.add_subplot(2,2,2)    
        s = L.output[1][:,i,:]
        img2 = ax2.imshow(s, origin = 'lower', aspect = 'auto')
        
        ax2.set_xlabel('Neuron index')
        ax2.set_title('Std')
        cbar2 = fig.colorbar(img2)
        cbar2.set_label('kHz')
        #plt.imshow(u, cmap = 'bwr', vmin = -1, vmax = 1)
        
        ax3 = fig.add_subplot(2,2,3)
        scale = L.bn_mean.weight/torch.sqrt(L.bn_mean.running_var)
        w = L.linear.weight*(scale.unsqueeze(0).T)
        vmax = torch.max(torch.abs(w))
        img3 = ax3.imshow(w.detach().numpy(), cmap = 'bwr', vmax = vmax, vmin = -vmax)
        ax3.set_title('Recurrent weight (BN-scaled)')
        fig.colorbar(img3)
       
        if len(L.input) > 3:
            ax4 = fig.add_subplot(2,2,4)
            with torch.no_grad():
                u_ext, s_ext = L.linear_ext.forward( L.input[3], L.input[4] ) #comment out if transforming the external input is not needed.        
                s_ext = L.bn_std_ext.forward(L.bn_mean_ext, u_ext, s_ext)
                u_ext = L.bn_mean_ext(u_ext)
            ax4.plot(u_ext[i,:])
            ax4.plot(s_ext[i,:])
            ax4.set_xlabel('Neuron index')
            ax4.set_title('External Input (after BN)')

        #ax4.plot(s_ext)
        
        fig2 = plt.figure()        
        for t in range( L.output[0].shape[0] + 1):
            if t == 0:
                corr = L.input[2][i,:,:]
            else:
                corr = L.output[2][t-1,i,:,:]
            
            ax = fig2.add_subplot(3,3,t+1)
            img = ax.imshow( corr.detach().numpy(), vmin = -1, vmax = 1, cmap = 'bwr')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('t = {}'.format(t))
        
        fig2.colorbar(img)
        
        return
        