import torch

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
    def __init__(self, dataset_name, sample_size = 10, input_dim = 1, output_dim = 1, transform = None):
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
        if dataset_name == 'cue_combo_no_corr':
            self.cue_combination(with_corr= False)
        elif dataset_name == 'cue_combo':
            self.cue_combination(with_corr= True)        
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
            rho = 2*torch.rand(self.sample_size) - 1
        else:
            rho = torch.zeros(self.sample_size)
                        
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
        
        target_mean, target_std = self.transform( target_mean, target_std )
        target_corr = torch.ones(self.sample_size,1,1)
        
        self.input_data = list(zip(input_mean, input_std, input_corr))
        self.target_data = list(zip(target_mean.unsqueeze(1), target_std.unsqueeze(1), target_corr))
        
        return
        

if __name__ == "__main__":    
    dataset = Dataset('cue_combo')   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        break
    #runfile('./apps/regression/data.py', wdir='./')