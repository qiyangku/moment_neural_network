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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, sample_size = 10, input_dim = 1, output_dim = 1, transform = None, with_corr = True, fixed_rho = None):
        """
        Args:
            dataset_name (string): name of the dataset
            sample_size (int): sample size
            input_dim, output_dim: number of features/labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sample_size = sample_size
        self.transform = transform
        self.fixed_rho = fixed_rho
        self.input_dim = input_dim
        self.output_dim = output_dim
        if dataset_name == 'cue_combo':          
            self.cue_combination(with_corr = with_corr)
        elif dataset_name == 'synfire':
            self.synfire()
        elif dataset_name == 'butterfly':
            self.butterfly()
        else:
            pass
                
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        sample = {'input_data': self.input_data[idx],
                  'target_data': self.target_data[idx]}            
        return sample    
        
    def cue_combination(self, with_corr = False):
        '''Generate random synthetic data points'''
        input_mean = 1 * torch.rand(self.sample_size, 2) #NB: 0<min(input mean)< output mean < max(input mean)<1/Tref = 0.2
        input_mean[:,1] = 0.0 #fix one of the distributions to unit gaussian
        input_std = 2 * torch.rand(self.sample_size, 2) + 1
        input_std[:,1] = 1.0 #fix one of the distributions to unit gaussian
        
        if with_corr:
            if self.fixed_rho:
                rho = torch.ones(self.sample_size)*self.fixed_rho
            else:
                rho = (2*torch.rand(self.sample_size) - 1)*0.5
        else:
            rho = torch.zeros(self.sample_size)
            if self.fixed_rho:
                print('Warning: correlation has been set to zero.')
                        
        input_corr = torch.zeros(self.sample_size, 2, 2)
        for i in range(self.sample_size): #not the most pythonic way of coding; but easier to read
            input_corr[i,:,:] = torch.eye(2) + (1-torch.eye(2))*rho[i]
                
        target_mean, target_std = prod_normal( input_mean[:,0], input_std[:,0], input_mean[:,1], input_std[:,1], rho)
        
        if not self.transform:            
            mean_scale = 1/(torch.max(target_mean)-torch.min(target_mean))*0.1
            mean_bias = - torch.min(target_mean)/(torch.max(target_mean)-torch.min(target_mean))*0.1+0.02
            std_scale = 1/(torch.max(target_std)-torch.min(target_std))*0.1
            std_bias = - torch.min(target_std)/(torch.max(target_std)-torch.min(target_std))*0.1+0.05
            
            self.transform = lambda x, y: (x*mean_scale + mean_bias , y*std_scale + std_bias)
            self.transform_coefs = {'mean_scale': mean_scale, 'mean_bias': mean_bias, 'std_scale': std_scale, 'std_bias': std_bias}
        
        target_mean, target_std = self.transform( target_mean, target_std )
        target_corr = torch.ones(self.sample_size,1,1)
        
        self.input_data = list(zip(input_mean, input_std, input_corr))
        self.target_data = list(zip(target_mean.unsqueeze(1), target_std.unsqueeze(1), target_corr))
        
        return
    
    def synfire(self):
        
        input_mean = torch.zeros(self.sample_size, self.input_dim)
        input_std = torch.zeros(self.sample_size, self.input_dim)
        input_corr = torch.zeros(self.sample_size, self.input_dim, self.input_dim)
        
        
        input_mean_ext = 0.1*torch.randn(self.sample_size, self.input_dim)  #uniform random external input
        input_std_ext = 0.2*torch.rand(self.sample_size, self.input_dim)
        
        target_mean = torch.zeros(self.sample_size, self.output_dim)
        target_std = torch.zeros(self.sample_size, self.output_dim)
        target_corr = torch.zeros(self.sample_size, self.output_dim, self.output_dim)
        
        patch = torch.zeros(self.input_dim)        
        patch[:int(self.input_dim/4)] = 1.0        
        patch_2d = torch.eye(self.input_dim)
        patch_2d[:int(self.input_dim/4), :int(self.input_dim/4)] = self.fixed_rho
        patch_2d.fill_diagonal_(1.0)
        
        for i in range(self.sample_size):
            rand_shift = int(torch.randint(self.input_dim, (1,)))
            # input_mean[i,:] = patch.roll( rand_shift )*2.0 #randomly shifting the location of the patch
            # input_std[i,:] = patch.roll( rand_shift)*5.0            
            input_mean[i,:] = patch.roll( rand_shift )*0.15 #randomly shifting the location of the patch
            input_std[i,:] = patch.roll( rand_shift)*0.2  
            
            #inputs are weakly correlated without spatial structure
            temp_corr = (torch.rand( self.input_dim, self.input_dim )*2 -1)*0.2 #weakly correlated
            temp_corr.fill_diagonal_(1.0)
            input_corr[i,:,:] = temp_corr
            
            #target output has fixed correlation
            target_mean[i,:] = patch.roll( rand_shift )*0.1
            target_std[i,:] = patch.roll( rand_shift)*0.1
            target_corr[i,:,:] = patch_2d.roll( (rand_shift,rand_shift) , (0,1))
        
        #input_mean_ext = input_mean.clone()
        #input_std_ext = input_std.clone()
        
        self.input_data = list(zip(input_mean, input_std, input_corr, input_mean_ext, input_std_ext))
        self.target_data = list(zip(target_mean, target_std, target_corr))
        return
        
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