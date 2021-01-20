# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:33:20 2021

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

file = './data/snn/validate_w_spk_neuron.npy'
emp_u, emp_s, maf_u, maf_s, u, s = np.load(file, allow_pickle = True)


fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)    

color = ['tab:blue', 'tab:orange']
label = [r'$\bar{\sigma} = 1$', r'$\bar{\sigma} = 2$']


for j in range(len(s)):
    ax1.plot(u, maf_u[:,j], color = color[j], alpha = 0.8, label = label[j])
    ax1.plot(u, emp_u[:,j], '.', color = color[j])
ax1.set_xlabel(r'Mean Input Current $\bar{\mu}$')
ax1.set_ylabel(r'Mean Firing Rate $\mu$')


ax2 = fig.add_subplot(1,2,2)    
for j in range(len(s)):
    ax2.plot(u, maf_s[:,j], color = color[j], alpha = 0.8, label = label[j])
    ax2.plot(u, emp_s[:,j], '.', color = color[j])
ax2.set_xlabel(r'Mean Input Current $\bar{\mu}$')
ax2.set_ylabel(r'Firing Variability $\sigma$')

ax1.legend()#['IF neuron','Moment Activation'])


#runfile('./dev_tools/validate_w_spiking_neuron_plots.py', wdir='./')