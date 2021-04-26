#from Mnn_Core.mnn_pytorch import *
from Mnn_Core.mnn_modules import *
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
#from ray import tune
import json
from apps.stim_dependent_var.stim_dependent_data import *
#from apps.ppc.visualization_tools import *

torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)

def loss_mse_mean_FF(pred_mean, pred_std, target_mean, target_FF):    
    pred_FF = pred_std*pred_std/(pred_mean+1e-5) # not ideal.
    loss = F.mse_loss(pred_FF, target_FF)
    loss += F.mse_loss(25*pred_mean, 25*target_mean) #scale the mean from ~0.04 to around 1
    return loss

# class MomentLayer(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MomentLayer, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.linear = Mnn_Linear_Module_with_Rho(input_size, output_size)                      
#         #cache the output
#         self.mean = 0
#         self.std = 0
#         self.corr = 0

#         return
        
#     def forward(self, u, s, rho):        
#         u_activated, s_activated, corr_activated = self.linear.forward(u, s, rho)                
#         self.mean, self.std, self.corr = u_activated, s_activated, corr_activated        
#         return u_activated, s_activated, corr_activated
    
MomentLayer = Mnn_Linear_Module_with_Rho #aren't they identical? Just create an alias


# class MomentLayer_w_FF(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MomentLayer, self).__init__()
#         '''Outputs Fano factor instead of std'''    
#         self.input_size = input_size
#         self.output_size = output_size
        
#         self.linear = Mnn_Summation_Layer_with_Rho(input_size, output_size)
#         self.bn = Mnn_BatchNorm1d_with_Rho(in_features=output_size)

#     def forward(self, u, s, rho):
#         u, s, rho = self.linear(u, s, rho)
#         u, s, rho = self.bn(u, s, rho)
#         u_activated = Mnn_Activate_Mean.apply(u, s)
#         #s_activated = Mnn_Activate_Std.apply(u, s, u_activated)
#         FF_activated = Mnn_Activate_FF.apply(u, s, u_activated)
#         #corr_activated = Mnn_Activate_Corr.apply(rho, u, s, u_activated, s_activated)
        
#         return u_activated, FF_activated
    

class MoNet(torch.nn.Module):
    def __init__(self, num_hidden_layers = 10, hidden_layer_size = 64, input_size = 2, output_size = 1):
        super(MoNet, self).__init__()
        self.layer_sizes = [input_size]+[hidden_layer_size]*num_hidden_layers+[output_size]  # input, hidden, output
        
        
        self.layers = torch.nn.ModuleList(
            [MomentLayer(self.layer_sizes[i], self.layer_sizes[i + 1], bn_ext_std= True) for i in range(len(self.layer_sizes) - 1)])
        
        for i in range(len(self.layers)): #initialization for batchnorm
            self.layers[i].bn.bn_mean.weight.data.fill_(2.0)
            self.layers[i].bn.bn_mean.bias.data.fill_(1.0)
        
        return

    def forward(self, u, s, rho):
        for i in range(len(self.layer_sizes) - 1):
            u, s, rho = self.layers[i].forward(u, s, rho)
        return u, s, rho


class MultilayerPerceptron():
    @staticmethod
    def train(config):        
        if config['seed'] is None:            
            torch.manual_seed(int(time.time())) #use current time as seed
        else:
            torch.manual_seed(config['seed'])
            
        if config['tensorboard']:
            writer = SummaryWriter(log_dir = config['log_dir'] + '_'+ str(config['trial_id']))#log_dir='D:\\mnn_py\\moment_activation\\runs2'
        
        sample_size = config['sample_size']        
        batch_size = config['batch_size'] #64
        num_batches = int(sample_size/batch_size)
        num_epoch = config['num_epoch'] #50#1000
        lr = config['lr']#0.01
        momentum = config['momentum'] #0.9
        optimizer_name = config['optimizer_name']
        input_size = config['input_size']
        output_size = config['output_size']
        
        if config['with_corr']:
            model = MoNet(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'], input_size = input_size, output_size = output_size)
        else:
            model = MoNet_no_corr(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'], input_size = input_size, output_size = output_size)        
            
        train_dataset = StimDepDataset(config)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)        
    
        model.target_transform = train_dataset.transform
        
        config_validate = config.copy()
        config_validate['sample_size'] = 32
        validation_dataset = StimDepDataset(config_validate)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 32)
            
        
        #model.target_affine = target_affine
        
        #param_container.set_ratio(0.1) # E/I balance. Mostly affect initialization. Mild value seems to improve result.
        #param_container.set_eps(0.01) # 1. Ensures sigma is positive; 2. avoid divide by 0 error. 3. Avoid overflow. Shouldn't need this to be too big. If error still occurs look for other sources.            
        #param_container.print_params()
        
        params = model.parameters()
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr = lr, amsgrad = True) #recommended lr: 0.1 (Adam requires a much smaller learning rate than SGD otherwise won't converge)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params, lr= lr, momentum= momentum) #recommended lr: 2
        else:
            optimizer = torch.optim.Adam(params, lr = lr)
            
        model.checkpoint = {
            'epoch': [],
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': []            
            }
        
        t0 = time.perf_counter()
        for epoch in range(num_epoch):            
            model.train()
            #--- iterate ver minibatches
            for i_batch, sample in enumerate(train_dataloader):
                optimizer.zero_grad()                
                if config['with_corr']:
                    u, s, rho = model.forward(sample['input_data'][0], sample['input_data'][1], sample['input_data'][2])
                else:
                    u, s = model.forward(sample['input_data'][0], sample['input_data'][1])
                
                #if config['loss'] == 'loss_mse_mean_FF':
                loss = loss_mse_mean_FF(u, s, sample['target_data'][0], sample['target_data'][1])
                loss.backward()
                optimizer.step()
            #--- 
                
            if config['tensorboard']:
                writer.add_scalar("Loss/Train", loss, epoch)
                writer.flush()
            
            #tune.report(loss = loss.item())
            
            if epoch % 1 == 0:
                print('Training epoch {}/{}'.format(epoch,num_epoch))
                with torch.no_grad():
                    model.eval()
                    for i_batch, sample in enumerate(validation_dataloader):
                        if config['with_corr']:
                            u, s, rho = model.forward(sample['input_data'][0], sample['input_data'][1], sample['input_data'][2])
                        else:
                            u, s = model.forward(sample['input_data'][0], sample['input_data'][1])
                        
                        #if config['loss'] == 'loss_mse_mean_FF':
                        loss = loss_mse_mean_FF(u, s, sample['target_data'][0], sample['target_data'][1])
                        
                        
                    if config['tensorboard']:
                        writer.add_scalar("Loss/Validation", loss, epoch)
                        writer.flush()
                
                    model.checkpoint['loss'].append(loss.item())
                    model.checkpoint['epoch'].append(epoch)
                    print(loss.item())
                    x = np.linspace(-np.pi,np.pi,101)[:-1]
                    plt.clf()
                    j = 4
                    plt.subplot(1,2,1)                                        
                    plt.plot(x,u[j,:],x,sample['target_data'][0][j,:])
                    plt.ylim(0,0.05)
                    plt.subplot(1,2,2)                    
                    plt.plot(x,s[j,:]*s[j,:]/(u[j,:]+1e-5),x,sample['target_data'][1][j,:])                    
                    #plt.ylim(1,2)
                    plt.pause(0.1)                    
                    plt.show()
            
            
        
        #print("===============================")
        print("Number of batches: ", num_batches)
        print("Batch size: ",batch_size)
        print("Learning rate: ", lr)
        print("Momentum: ", momentum)
        print("Time Elapsed: ", time.perf_counter()-t0)
        print("===============================")
        
        model.checkpoint['model_state_dict'] =  model.state_dict()
        model.checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        #writer.add_graph(model, (input_mean,input_std))
        if config['tensorboard']:
            if config['dataset_name'] == 'cue_combo':
                fig, ax1 = plt.subplots(nrows=1, ncols=1)
                l1, = ax1.plot(u.detach().numpy(),s.detach().numpy(),'+')
                l2, = ax1.plot(sample['target_data'][0].detach().numpy(), sample['target_data'][1].detach().numpy(),'.')
                ax1.set_xlabel('Mean')
                ax1.set_ylabel('Std')
                plt.legend([l1,l2],['Output','Target'])    
                
                writer.add_figure('Final output vs target',fig)            
            
                fig1 = VisualizationTools.plot_grid_map(model, with_corr = config['with_corr'], rho = config['fixed_rho'])
                writer.add_figure('Test data grid map',fig1)
            else:
                pass
            
            writer.flush()
            writer.close()
        
        return model

if __name__ == "__main__":    

    config = {'sample_size': 32*100,
              'batch_size': 64,
              'num_epoch': 50,
              'lr': 0.005,
              'momentum': 0.9,
              'optimizer_name': 'Adam',
              'num_hidden_layers': 5,
              'input_size': 200,
              'output_size': 100,
              'hidden_layer_size': 100,
              'trial_id': int(time.time()),
              'tensorboard': True,
              'log_dir': 'runs/ppc',
              'with_corr': True,
              'dataset_name': None,
              'loss':'mse_mean_FF',
              'seed': None,
              'fixed_rho': None #ignored if with_corr = False
        }
    
    model = MultilayerPerceptron.train(config)
    
    file_name = config['trial_id']
    #torch.save(model.checkpoint, './data/ppc/{}.pt'.format(file_name) ) #save result by time stamp
    #with open('./data/ppc/{}_config.json'.format(file_name),'w') as f:
    #    json.dump(config,f)
    
    #plt.plot(model.layers[-1].mean)
    #runfile('./apps/stim_dependent_var/stim_multilayer_perceptron.py', wdir='./')