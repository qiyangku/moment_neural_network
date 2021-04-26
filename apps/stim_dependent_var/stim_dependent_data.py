import torch
import torch.utils.data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)

def test_circ_shift(dx):
    #func = lambda x: np.power(x,2)
    func = lambda x: torch.cos(2.5*x)
    x = torch.linspace(-np.pi,np.pi,101)    
    y = circ_shift_even_func(func, x, dx)
    plt.plot(x,y)
    return
    


def circ_shift_even_func(func, x, dx):
    #circular-shift any even function defined on (-pi,pi)
    # x = x-coordinates
    # dx = shift    
    dx = dx % (2*np.pi); #circular shift, always between (0,2*pi)    
    y = func(x-dx);        
    indx = (x-(-np.pi+dx)) < 0;    
    y2 = func(x+2*np.pi-dx); #this should keep consistency in dimensionality    
    y[indx] = y2[indx];
        
    return y


class StimDepDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform = None, debug = False):
        """
        Args:
            dataset_name (string): name of the dataset
            sample_size (int): sample size
            input_dim, output_dim: number of features/labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sample_size = config['sample_size']
        self.transform = transform
        self.fixed_rho = config['fixed_rho']
        self.input_dim = config['input_size']
        self.output_dim = config['output_size']
        #self.num_timesteps = config['max_time_steps']
        #self.ext_input_type = config['ext_input_type']        
        self.gen = torch.Generator()
        
        self.debug = debug
        
        # Define target function (contrast, orientation) --> (rate,FF)
        #parameters for rate
        alpha = 44.232813273588505e-3 #make sure using unit in kHz
        kappa = 1.260292217144681
        r0 = 0.12999431984111384*alpha
        Ainf = 0.9424024250939554*alpha
        b = 2.2836823176519014
        
        
        #parameters for FF
        F0 = 1.762867589808626
        k = 1.4490923936326574
        Binf = 0.09875216728919675
        d = 8.394477025250017
        bprime = 3.225479723072191
        
        self.fun_rate = lambda x,y: Ainf*x/(x+b)*np.exp(kappa*(np.cos(y)-1)) +r0
        self.fun_FF = lambda x,y: F0-Binf*x/(x+bprime)*(np.cos(k*y)+d)
        
        # Define input neuron locations (uniformly scattered on a disc)
        th = 2*np.pi*torch.rand( self.input_dim, generator = self.gen) #random angles
        r = torch.rand( self.input_dim, generator = self.gen) + torch.rand( self.input_dim, generator = self.gen) #radius
        r[r>1] = 2 - r[r>1]
        self.x_input = r*torch.cos(th)
        self.y_input = r*torch.sin(th)
        
        self.c0 = 1 #global illuminance
        self.cmax = 0.16 #maximum contrast 0<cmax<1
        self.k = 5*np.pi #spatial frequency (assume radius of grating pattern is 1)
        
        #prefered direction of output neurons
        #self.th_out = torch.linspace(0,2*np.pi,self.output_dim+1)[:-1].view(1,self.output_dim)
        self.th_out = torch.linspace(-np.pi,np.pi,self.output_dim+1)[:-1].view(1,self.output_dim)
        #BOTH periodic domain above are fine.
                
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            batch_size = len(idx)
            self.gen.manual_seed(idx[0]) #set random seed with input index
        else:
            batch_size = 1
            self.gen.manual_seed(idx) #set random seed with input index
        
        input_data, target_data = self.gen_sample(batch_size, idx = idx)
        #use indx[0] as seed and generate len(indx) samples!
        sample = {'input_data': input_data[0],
                  'target_data': target_data[0]}
        #print('input data length: ',len(input_data))
        return sample    
        
    def gen_sample(self, batch_size,  idx = None):#num_timesteps, ext_input_type,
        
        # STEP 1: generate random orientations and contrast
        if self.debug: # Debug mode
            #theta = torch.linspace(0,2*np.pi, self.sample_size+1)[:-1].unsqueeze(-1).unsqueeze(-1)
            theta = torch.linspace(-np.pi,np.pi, self.sample_size+1)[:-1].unsqueeze(-1).unsqueeze(-1)
            #NB for some reason, theta must be defined on (-pi,pi) as the circshift would fail for (pi,2*pi)! HOW?
            theta = theta[idx,:,:].unsqueeze(0)
            c = 0.1*torch.ones(batch_size,1)
        else:
            theta = torch.rand(batch_size,1,1, generator = self.gen)*2*np.pi- np.pi
            c = torch.rand(batch_size,1, generator = self.gen)*self.cmax
        
        
        
        # STEP 2: input encoding
        input_mean = torch.ones(batch_size, self.input_dim)*self.c0
        input_std = self.c0/np.sqrt(2)*c*torch.ones(1,self.input_dim)
        
        dx = self.x_input.view(1,self.input_dim,1) - self.x_input.view(1,1,self.input_dim)
        dy = self.y_input.view(1,self.input_dim,1) - self.y_input.view(1,1,self.input_dim)
        
        kx = self.k*torch.cos(theta)
        ky = self.k*torch.sin(theta)
        
        input_corr = torch.cos(  kx*dx +ky*dy  )
        
        
        # STEP 3: target encoding
        target_mean = self.fun_rate( c, theta[:,:,0] - self.th_out )
        #target_FF = self.fun_FF(c, theta[:,:,0] - self.th_out ) #this probably will break
        target_FF = circ_shift_even_func(lambda y: self.fun_FF(c,y) , theta[:,:,0], self.th_out )
        
        #target_std = torch.sqrt( target_FF*target_mean )
        target_FF = target_FF
        target_corr = torch.zeros(batch_size, self.output_dim, self.output_dim)
        
        #input_data = list(zip(input_mean, input_std, input_corr, input_mean_ext, input_std_ext))
        input_data = list(zip(input_mean, input_std, input_corr))
        target_data = list(zip(target_mean, target_FF, target_corr))
        return input_data, target_data
        
    def visualize_sample(self, sample):
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.th_out.T, sample['target_data'][0].T)
        plt.xlabel('Neuron location')
        plt.ylabel('Target mean')
        
        
        plt.subplot(2,2,2)
        plt.plot(self.th_out.T, sample['target_data'][1].T)
        plt.xlabel('Neuron location')
        plt.ylabel('Target FF')
        
        return

if __name__ == "__main__":    
    
    #test_circ_shift(0)
    
    config ={
        'sample_size': 5,        
        'fixed_rho': False,
        'input_size': 20,
        'output_size': 100,
        'max_time_steps': None,
        'ext_input_type': None
        }
    
    dataset = StimDepDataset(config, debug = False)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)
    for i_batch, sample_batched in enumerate(dataloader):
        sample = sample_batched
        dataset.visualize_sample(sample)
        break
    #runfile('./apps/regression/data.py', wdir='./')