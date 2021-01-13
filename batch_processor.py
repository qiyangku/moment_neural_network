# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:06:29 2021
A helper function for running multiple runs with different config
@author: dell
"""

from apps.regression.multilayer_perceptron import *
#from apps.regression.recurrent_nn import *
import numpy as np
import sys, time

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
        config['trial_id'] = int(time.time()) #add unique time stamp
    
    return config
    

        

if __name__ == "__main__":    
    #below is a demo
    
    search_space = {'num_batches': [6000],
              'batch_size': [32],
              'num_epoch': [100],
              'lr': np.logspace(-4,-1,20),
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
              'fixed_rho': [None] #ignored if with_corr = False
        }
    
    indx = int(sys.argv[1])
    config = hyper_para_generator(search_space, indx)
    
    model = MultilayerPerceptron.train(config)
    
    folder_name = str(int(  time.time()*1000 ))
    file_name = str(indx).zfill(3)+'_'+ str(config['trial_id'])
    torch.save(model.state_dict(), './data/{}/{}.pt'.format(folder_name, file_name) ) #save result by time stamp
    with open('./data/{}/{}_config.json'.format(folder_name, file_name),'w') as f:
        json.dump(config,f)
    #if search_space already saved to folder_name then skip
    #otherwise save it
    print('Run {} complete.'.format(indx))

    #runfile('./dev_tools/batch_processor.py', args = '3', wdir='./')
