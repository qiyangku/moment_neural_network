# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:06:29 2021
A helper function for running multiple runs with different config
@author: dell
"""

from apps.regression.multilayer_perceptron import *
#from apps.regression.recurrent_nn import *
import numpy as np
import os, sys, time

#For PBS:
#INPUT: search_space a dictionary of lists
#       PBS_array index
#Wrapper: nested loop over the search_space
#Output the config dictionary


def hyper_para_generator(search_space, indx):
    '''
    
    Parameters
    ----------
    search_space : DICT of LISTS
    indx: a linear index
        A helper function that produce a single hyperparameter configuration given a set of configurations.

    Returns config dictionary
    -------
    None.

    '''
    
    para_name = list(search_space) #list of parameter names
    shape = [len(search_space[k]) for k in para_name] #list of length of parameters
    if indx >= np.prod(shape):
        print('Warning: index out of range!')
        config = None
    else:
        
        subs = np.unravel_index( [indx]  , shape ) #convert linear index to subscripts
        
        config = {}
        i = 0
        for k in para_name:
            x = subs[i][0] #subscript for parameter k
            config[k] = search_space[k][x]
            i += 1    
        config['trial_id'] = i #int(time.time()) #add unique time stamp
    
    return config
    

        

if __name__ == "__main__":    
    #below is a demo
    
    search_space = {'num_batches': [2000],
              'batch_size': [32],
              'num_epoch': [30],
              'lr': list(np.logspace(-2,0,51)),
              'momentum': [0.9],
              'optimizer_name': ['Adam'],
              'num_hidden_layers': [3],
              'max_time_steps': [None],
              'input_size': [2],
              'output_size': [1],
              'hidden_layer_size': [32],
              'tensorboard': [False],
              'with_corr': [True],
              'dataset_name': ['cue_combo'],
              'log_dir': ['runs/cue_combo'],
              'loss': ['mse_no_corr'],
              'seed': [0], #set to None for random seed
              'fixed_rho': [None], #ignored if with_corr = False
              'exp_id': [sys.argv[2]] #the name given to this experiment
        }
    
    indx = int(sys.argv[1])
    config = hyper_para_generator(search_space, indx)
    model = MultilayerPerceptron.train(config)
    
    path =  './data/{}/'.format( config['exp_id'] )
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path+'search_space.json','w') as f:
            json.dump(search_space, f)
    
    file_name = str(indx).zfill(3) +'_'+str(int(time.time()))
    torch.save(model.checkpoint, path +'{}.pt'.format(file_name) ) #save result by time stamp
    with open(path +'{}_config.json'.format(file_name),'w') as f:
        json.dump(config,f)

    #runfile('./dev_tools/batch_processor.py', args = '3', wdir='./')
