import torch
import torch.utils.data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)


def prod_normal(mu1,s1,mu2,s2,rho):
    '''Stats of the product of two normal distributions'''
    v1 = s1*s1
    v2 = s2*s2
    cov = s1*s2*rho    
    det = v1 + v2 - 2*cov #determinant of the covariance matrix
    
    mu3 = ((v2 - cov)*mu1 + (v1-cov)*mu2)/det
    s3 = (v1*v2*(1-rho*rho))/det
    s3 = torch.pow(s3,0.5)
    
    return mu3, s3


class SynfireDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform = None):
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
        self.num_timesteps = config['max_time_steps']
        self.ext_input_type = config['ext_input_type']        
        self.gen = torch.Generator()
                
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
        
        input_data, target_data = self.gen_sample(batch_size, self.num_timesteps, self.ext_input_type)
        #use indx[0] as seed and generate len(indx) samples!
        sample = {'input_data': input_data[0],
                  'target_data': target_data[0]}
        #print('input data length: ',len(input_data))
        return sample    
        
    def gen_sample(self, batch_size, num_timesteps, ext_input_type):
        
        input_mean = torch.zeros(batch_size, self.input_dim)
        input_std = torch.zeros(batch_size, self.input_dim)
        input_corr = torch.zeros(batch_size, self.input_dim, self.input_dim)
        
        
        target_mean = torch.zeros(batch_size, self.output_dim)
        target_std = torch.zeros(batch_size, self.output_dim)
        target_corr = torch.zeros(batch_size, self.output_dim, self.output_dim)
        
        #patch = torch.zeros(self.input_dim)        
        #patch[:int(self.input_dim/4)] = 1.0        
        x = torch.arange(0,2*np.pi,  2*np.pi/self.input_dim)        
        d = 0.7
        patch = torch.exp( (torch.cos(x)-1)/d/d);  #von Mises
        
        #patch_2d = torch.eye(self.input_dim)
        #patch_2d[:int(self.input_dim/4), :int(self.input_dim/4)] = self.fixed_rho
        #patch_2d.fill_diagonal_(1.0)
        corr_tmp = torch.zeros(self.input_dim, self.input_dim)
        for i in range(self.input_dim):
            corr_tmp[i,:] = patch.roll(i)*self.fixed_rho
        corr_tmp.fill_diagonal_(1.0)
        
        
        for i in range(batch_size):
            rand_shift = int(torch.randint(self.input_dim, (1,), generator = self.gen))
            # input_mean[i,:] = patch.roll( rand_shift )*2.0 #randomly shifting the location of the patch
            # input_std[i,:] = patch.roll( rand_shift)*5.0            
            input_mean[i,:] = patch.roll( rand_shift )*0.15 #randomly shifting the location of the patch
            input_std[i,:] = patch.roll( rand_shift)*0.2  
            
            #inputs are weakly correlated without spatial structure
            temp_corr = (torch.rand( self.input_dim, self.input_dim, generator = self.gen )*2 -1)*0.2 #weakly correlated
            temp_corr.fill_diagonal_(1.0)
            input_corr[i,:,:] = temp_corr
            
            #target output has fixed correlation
            target_mean[i,:] = patch.roll( rand_shift )*0.1
            target_std[i,:] = patch.roll( rand_shift)*0.1
            #target_corr[i,:,:] = patch_2d.roll( (rand_shift,rand_shift) , (0,1))
            target_corr[i,:,:] = corr_tmp
        
        #use external input
        if ext_input_type == 'persistent':            
            input_mean_ext = input_mean.clone().unsqueeze(-1).repeat(1, 1, num_timesteps)
            input_std_ext = input_std.clone().unsqueeze(-1).repeat(1, 1, num_timesteps)
        elif ext_input_type == 'transient':
            input_mean_ext = 0.1*torch.randn(batch_size, self.input_dim, num_timesteps, generator = self.gen)  #uniform random external input
            input_std_ext = 0.2*torch.rand(batch_size, self.input_dim, num_timesteps, generator = self.gen)   
            input_mean_ext[:,:,0] = input_mean.clone()
            input_mean_ext[:,:,1] = input_std.clone()            
        elif ext_input_type == 'fadeoff':
            #use a loop for the sake of readability.
            input_mean_ext = 0.1*torch.randn(batch_size, self.input_dim, num_timesteps, generator = self.gen)  #uniform random external input
            input_std_ext = 0.2*torch.rand(batch_size, self.input_dim, num_timesteps, generator = self.gen)       
            for k in range(int(num_timesteps/2)): #fade off the input for the first half of time steps
                amp = 1 - k/int(num_timesteps/2)
                input_mean_ext[:,:,k] += input_mean.clone()*amp
                input_std_ext[:,:,k] += input_std.clone()*amp
        else:                
            input_mean_ext = 0.1*torch.randn(batch_size, self.input_dim, num_timesteps, generator = self.gen)  #uniform random external input
            input_std_ext = 0.2*torch.rand(batch_size, self.input_dim, num_timesteps, generator = self.gen)    
        
        input_data = list(zip(input_mean, input_std, input_corr, input_mean_ext, input_std_ext))
        target_data = list(zip(target_mean, target_std, target_corr))
        return input_data, target_data
        
    def butterfly(self):
        ''' Encode an image of a butterfly into the '''
        img = Image.open("./data/butterfly.png").convert('L')
        img.thumbnail((self.output_dim, self.output_dim), Image.ANTIALIAS)
        
        img = np.asarray(img)/256
        img = np.triu(img) #take the upper triangular entries
        img += img.T
        img = torch.Tensor(img)                
        #np.fill_diagonal(img, 1.0)
        #plt.imshow(img, cmap = 'bwr', vmin = -1, vmax = 1)
        #plt.imshow(img, vmin = 0, vmax = 1)
        
        input_mean = (torch.rand(self.sample_size, self.input_dim)*2-1)*1
        input_std = torch.rand(self.sample_size, self.input_dim)*5
        
        input_corr = torch.zeros(self.sample_size, self.input_dim, self.input_dim)
        
        target_mean = torch.zeros(self.sample_size, self.output_dim)
        target_std = torch.zeros(self.sample_size, self.output_dim)
        target_corr = torch.zeros(self.sample_size, self.output_dim, self.output_dim)
        
        for i in range(self.sample_size):
            temp = img + torch.randn((self.output_dim, self.output_dim))*0.2 #add noise
            temp[temp > 1] = 1.0 #clip range
            temp[temp < -1] = -1.0
            temp.fill_diagonal_(1.0)
            
            input_corr[i,:,:] = temp
            target_corr[i,:,:] = img
        
        self.input_data = list(zip(input_mean, input_std, input_corr))
        self.target_data = list(zip(target_mean, target_std, target_corr))
        
        return 

if __name__ == "__main__":    
    dataset = Dataset('cue_combo')   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        break
    #runfile('./apps/regression/data.py', wdir='./')