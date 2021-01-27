# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:26:28 2020

@author: dell
"""
import torch
import os, fnmatch, json
import numpy as np
#from apps.synfire.recurrent_nn import *
from apps.synfire.recurrent_nn_w_input_current import *

from apps.synfire.synfire_data import prod_normal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class ResultInspector():
    def __init__(self, folder_name):
        self.path = './data/{}/'.format(folder_name)
        self.file_list = os.listdir(self.path)        
        return
        
    def load_result(self,indx):
        '''Load result labeled with INDX inside a folder named FOLDER_NAME'''
        
        if isinstance(indx, int):
            indx = str(indx).zfill(3)        
        
        #load config file
        config_file = fnmatch.filter(self.file_list, indx+'*_config.json')[0]        
        with open(self.path +'/'+config_file) as f:
            config = json.load(f)
        
        #load model
        result = fnmatch.filter(self.file_list, indx+'*.pt')[0]
        checkpoint = torch.load(self.path +'/'+result)
        
        #model = MoNet(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'], input_size = config['input_size'], output_size = config['output_size'])
        model = Renoir(max_time_steps = config['max_time_steps'], hidden_layer_size = config['hidden_layer_size'], input_size = config['input_size'], output_size = config['output_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() #set to evaluation mode
        
        return model, config, checkpoint
    
    def loss_vs_lr(self):
        
        with open(self.path +'/' + 'search_space.json') as f:
            x = json.load(f)['lr']
        
        #num_files = len(fnmatch.filter(self.file_list, '*.pt'))
        num_files = len(x)
        loss = np.zeros(num_files)
        x = np.zeros(num_files)
        
        fig = plt.figure()
        ax2 = fig.add_subplot(1,2,2)
        
        legend_labels = []
        for i in range(num_files):
            try:
                _, config, cp = self.load_result(i)
                loss[i] = cp['loss'][-1]
                x[i] = config['lr']
                if i % 5 ==  0:
                    #c = [i/num_files*k/256 for k in (50, 64, 164)]
                    #ax2.semilogy(cp['loss'], color = c , alpha = 0.5)
                    ax2.semilogy(cp['loss'], alpha = 0.8)
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    legend_labels.append(config['lr'])
            except Exception as e:
                loss[i] = np.nan
                x[i] = np.nan
                print('No data found for trial {}!'.format(i))
                print(e)
        
        ax2.legend(labels = legend_labels)
        
        ax1 = fig.add_subplot(1,2,1)    
        ax1.loglog(x,loss,'.')
        ax1.set_xlabel('Learning rate')
        ax1.set_ylabel('Loss')
        return loss, x
    
    def plot_all_trials(self):
        with open(self.path +'/' + 'search_space.json') as f:
            search_space = json.load(f)
        
        num_files = len(fnmatch.filter(self.file_list, '*.pt'))
        
        loss = np.zeros( (num_files, search_space['num_epoch'][0]))
        LR = np.zeros(num_files)
        
        fig = plt.figure()
        ax2 = fig.add_subplot(2,2,2)
        
        legend_labels = []
        for i in range(num_files):
            try:
                _, config, cp = self.load_result(i)
                loss[i,:] = cp['loss']
                LR[i] = config['lr']
                if i % 5 ==  0:
                    #c = [i/num_files*k/256 for k in (50, 64, 164)]
                    #ax2.semilogy(cp['loss'], color = c , alpha = 0.5)
                    ax2.semilogy(cp['loss'], alpha = 0.8)
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    legend_labels.append(config['lr'])
            except Exception as e:
                loss[i,:] = np.nan
                LR[i] = np.nan
                print('No data found for trial {}!'.format(i))
                print(e)
        
        ax2.legend(labels = legend_labels)
        
        ax1 = fig.add_subplot(2,2,1)    
        ax1.loglog(LR,loss[:,-1],'.')
        ax1.set_xlabel('Learning rate')
        ax1.set_ylabel('Loss')
        
        ax3 = fig.add_subplot(2,2,3)    
        ax3.semilogy(np.nanmean(loss[:10,:], axis = 0))
        ax3.semilogy(np.nanmean(loss[10:,:], axis = 0))
        ax3.legend(search_space['lr'])
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (trial average)')
        
        return loss, LR
            
    def validate(self, model , config):
        '''depreciated method'''
        
        validation_dataset = SynfireDataset(config)        
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 32, shuffle=False)
        
        model.eval() #set to evaluation mode
        for i_batch, sample in enumerate(validation_dataloader):
            ext_mean = torch.ones(sample['input_data'][3].shape)
            ext_std = torch.ones(sample['input_data'][4].shape)        
            
            if config['with_corr']:
                u, s, rho = model.forward(sample['input_data'][0], sample['input_data'][1], sample['input_data'][2], ext_mean , ext_std)
            else:
                u, s = model.forward(sample['input_data'][0], sample['input_data'][1])
            
            if config['loss'] == 'mse_no_corr':
                loss = loss_function_mse(u, s, sample['target_data'][0], sample['target_data'][1])
            elif config['loss'] == 'mse_covariance':
                loss = loss_mse_covariance(u, s, rho, sample['target_data'][0], sample['target_data'][1], sample['target_data'][2])
            break
        
        return model, loss.item()



class VisualizationTools():
    @staticmethod
    def trial_average(model):
        model.eval()
        L = model.recurrent_layer
        with torch.no_grad():
            scale = L.bn_mean.weight/torch.sqrt(L.bn_mean.running_var)
            w = L.linear.weight*(scale.unsqueeze(0).T)
            if len(L.input) > 3:                    
                u_ext, s_ext = L.linear_ext.forward( L.input[3][:,:,-1], L.input[4][:,:,-1] ) #comment out if transforming the external input is not needed.        
                s_ext = L.bn_std_ext.forward(L.bn_mean_ext, u_ext, s_ext)
                u_ext = L.bn_mean_ext(u_ext) + L.bn_mean.bias - scale*L.bn_mean.running_mean
                
                u_ext = torch.mean(u_ext) #average over space and minibatch
                s_ext = torch.mean(s_ext)
        print('External input mean = {}, std ={}'.format(u_ext,s_ext))
        
        w_avg = 0
        
        for i in range(w.shape[0]):
            w_avg += w[i,:].roll(-i).detach().numpy()
        
        w_avg = w_avg/w.shape[0]
                
        #plt.imshow(w,cmap = 'bwr',vmax =10,vmin=-10)
        x = np.arange(0,2*np.pi, 2*np.pi/w.shape[0])
        
        func = lambda x, we, wi, ke, ki: we*np.exp( ke*(np.cos(x)-1) ) - wi*np.exp( ki*(np.cos(x)-1) )
        popt, pcov  = curve_fit(func, x, w_avg, p0 = [15, 8, 4, 1], maxfev=10000)
        y_fit =  func(x, *popt)
        y_fit = torch.Tensor(y_fit)
        print(popt)
        #y_fit = func(x, 15, 8, 4, 1)
        
        #plt.plot(x, w_avg,'.')
        #plt.plot(x, y_fit)
        
        w_fitted = torch.zeros(w.shape)
        for i in range(w.shape[0]):
            w_fitted[i,:] = y_fit.roll(i)
        
        #plt.imshow(w_fitted)
        
        with torch.no_grad():
            x = torch.Tensor(x)
            
            u = torch.exp( torch.cos(x)-1 ).squeeze(0)
            s = torch.exp( torch.cos(x)-1 ).squeeze(0)
            rho = torch.eye(len(x) ).squeeze(0)
            # manualy run recurrent loops
            timesteps = 100
            U = torch.zeros(len(x) ,timesteps)
            S = torch.zeros(len(x) ,timesteps)
            R = torch.zeros(len(x) , len(x), timesteps)
            
            linear = Mnn_Linear_Corr(w.shape[0], w.shape[0])
            linear.weight.data = w_fitted        
            
            for i in range(timesteps):
                u, s, rho = linear.forward(u, s, rho)
                u += u_ext + 0.3
                s = torch.sqrt( s*s + s_ext*s_ext + 2)
                
                u_activated = Mnn_Activate_Mean.apply(u, s)
                s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
                corr_activated = Mnn_Activate_Corr.apply( rho , u, s, u_activated, s_activated)
                
                u = u_activated.clone()
                s = s_activated.clone()
                rho = corr_activated.clone()
                
                U[:,i] = u
                S[:,i] = s
                R[:,:,i] = rho
        
        #L.linear.weight
        #u, s, rho = L.forward(u, s, rho, u_ext, s_ext, self.max_time_steps)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)  
        #ax1.imshow(U, aspect = 'auto')
        ax1.plot(x,u)
        ax2 = fig.add_subplot(2,2,2)  
        #ax2.imshow(S, aspect = 'auto')
        ax2.plot(x,s)
        ax3 = fig.add_subplot(2,2,3)  
        #ax3.imshow(w_fitted, cmap = 'bwr', vmax = torch.max(y_fit), vmin = -torch.max(y_fit))
        ax3.plot(x, y_fit)
        ax4 = fig.add_subplot(2,2,4)  
        ax4.imshow(rho, cmap = 'bwr', vmax = 1, vmin = -1)
        
        
        return
        
        
    @staticmethod
    def plot_rnn(model, i = 0):
        fig = plt.figure()
        L = model.recurrent_layer
        #i = 3 #sample index
        
        ax1 = fig.add_subplot(2,2,1)    
        u = L.output[0][:,i,:].detach().numpy()
        u = np.roll(u, (int(u.shape[1]/2), 0))
        img1 = ax1.imshow(u, origin = 'lower', aspect = 'auto')
        ax1.set_xlabel('Neuron index')
        ax1.set_ylabel('Time steps')
        ax1.set_title('Mean')
        cbar1 = fig.colorbar(img1)
        cbar1.set_label('kHz')#, rotation=270)
        
        ax2 = fig.add_subplot(2,2,2)    
        s = L.output[1][:,i,:].detach().numpy()
        s = np.roll(s, (int(s.shape[1]/2), 0))
        img2 = ax2.imshow(s, origin = 'lower', aspect = 'auto')
        
        ax2.set_xlabel('Neuron index')
        ax2.set_title('Std')
        cbar2 = fig.colorbar(img2)
        cbar2.set_label('kHz')
        #plt.imshow(u, cmap = 'bwr', vmin = -1, vmax = 1)
        
        ax3 = fig.add_subplot(2,2,3)
        scale = L.bn_mean.weight/torch.sqrt(L.bn_mean.running_var)
        
        if L.linear.weight.data.dim() > 1:
            w = L.linear.weight*(scale.unsqueeze(0).T)
            vmax = 10#torch.max(torch.abs(w))
            img3 = ax3.imshow(w.detach().numpy(), cmap = 'bwr', vmax = vmax, vmin = -vmax)
            fig.colorbar(img3)
        else:
            w = (L.linear.weight*scale)
            w = np.roll(w.detach().numpy(), (int(u.shape[1]/2), 0))
            img3 = ax3.plot(w)
            
        ax3.set_title('Recurrent weight (BN-scaled)')
        
        
       
        try:
            if len(L.input) > 3:                
                with torch.no_grad():
                    u_ext, s_ext = L.linear_ext.forward( L.input[3][:,:,-1], L.input[4][:,:,-1] ) #comment out if transforming the external input is not needed.        
                    s_ext = L.bn_std_ext.forward(L.bn_mean_ext, u_ext, s_ext)
                    u_ext = L.bn_mean_ext(u_ext) + L.bn_mean.bias - scale*L.bn_mean.running_mean
                ax4 = fig.add_subplot(2,2,4)
                ax4.plot(u_ext[i,:])
                ax4.plot(s_ext[i,:])
                ax4.set_xlabel('Neuron index')
                ax4.set_title('External Input (after BN)')
        except:
            ax4 = fig.add_subplot(2,2,4)
            #ax4.plot(u[-1,:])
            #ax4.plot(s[-1,:])
            
            u_ext = L.input[3][0,:,-1] + L.bn_mean.bias - scale*L.bn_mean.running_mean
            u_ext = u_ext.detach().numpy()
            u_ext = np.roll(u_ext,int(u.shape[1]/2) )
            s_ext = L.input[4][0,:,-1].detach().numpy()
            s_ext = np.roll(s_ext,int(u.shape[1]/2) )
            ax4.set_title('External Input (after BN)')
            ax4.plot(u_ext)
            ax4.plot(s_ext)
            pass

        #ax4.plot(s_ext)
        
        fig2 = plt.figure()        
        for t in range( L.output[0].shape[0] + 1):
            if t == 0:
                corr = L.input[2][i,:,:].detach().numpy()
            else:
                corr = L.output[2][t-1,i,:,:].detach().numpy()
                if L.linear.weight.data.dim() == 1:
                    shift = (int(u.shape[1]/2), int(u.shape[1]/2))
                    corr = np.roll(corr, shift, (0,1) )#
            
            ax = fig2.add_subplot(4,3,t+1)
            img = ax.imshow( corr, vmin = -1, vmax = 1, cmap = 'bwr')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('t = {}'.format(t))
        
        fig2.colorbar(img)
        
        return

if __name__ == "__main__":    
    
    ri = ResultInspector('synfire')    
    model, config, checkpoint = ri.load_result('1611777007')
    model, loss = ri.validate(model, config)  
    VisualizationTools.plot_rnn(model)    
    
    #VisualizationTools.trial_average(model)
    #runfile('./apps/synfire/synfire_visualization_tools.py', wdir='./')        