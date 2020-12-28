from Mnn_Core.mnn_pytorch import *
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
#from ray import tune
import json
from apps.regression.data import *
from apps.regression.visualization_tools import *

torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)

class MomentLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MomentLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = Mnn_Linear_Corr(input_size, output_size)

        self.bn_mean = torch.nn.BatchNorm1d(output_size)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)
        #this roughly set the input mu in the range (0,5)
        
        self.bn_std = Mnn_Std_Bn1d(output_size , bias = False)
        #self.bn_std.ext_bias.data.fill_(1.0)
        
        #cache the output
        self.mean = 0
        self.std = 0
        self.corr = 0

        return
        
    def forward(self, u, s, rho):
        u, s, rho = self.linear.forward(u, s, rho)
        
        #s = mnn_std_bn1d(self.bn_mean, u, s)
        s = self.bn_std.forward(self.bn_mean, u, s)
        u = self.bn_mean(u)
                
        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
        corr_activated = Mnn_Activate_Corr.apply( rho , u, s, u_activated, s_activated)
        
        self.mean, self.std, self.corr = u_activated, s_activated, corr_activated
        
        return u_activated, s_activated, corr_activated

class MomentLayer_no_corr(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MomentLayer_no_corr, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = Mnn_Linear_without_Corr(input_size, output_size)

        self.bn_mean = torch.nn.BatchNorm1d(output_size)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)
        #this roughly set the input mu in the range (0,5)
        
        self.bn_std = Mnn_Std_Bn1d(output_size , bias = False)
        #self.bn_std.ext_bias.data.fill_(1.0)
        
        #cache the output
        self.mean = 0
        self.std = 0
        self.corr = 0

        return
        
    def forward(self, u, s):
        u, s = self.linear.forward(u, s)
        
        #s = mnn_std_bn1d(self.bn_mean, u, s)
        s = self.bn_std.forward(self.bn_mean, u, s)
        u = self.bn_mean(u)
                
        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
        
        self.mean, self.std = u_activated, s_activated
        
        return u_activated, s_activated


class MoNet(torch.nn.Module):
    def __init__(self, num_hidden_layers = 10, hidden_layer_size = 64):
        super(MoNet, self).__init__()
        self.layer_sizes = [2]+[hidden_layer_size]*num_hidden_layers+[1]  # input, hidden, output
        
        
        self.layers = torch.nn.ModuleList(
            [MomentLayer(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])
        
        return

    def forward(self, u, s, rho):
        for i in range(len(self.layer_sizes) - 1):
            u, s, rho = self.layers[i].forward(u, s, rho)
        return u, s, rho

class MoNet_no_corr(torch.nn.Module):
    def __init__(self, num_hidden_layers = 10, hidden_layer_size = 64):
        super(MoNet_no_corr, self).__init__()
        self.layer_sizes = [2]+[hidden_layer_size]*num_hidden_layers+[1]  # input, hidden, output
        
        self.layers = torch.nn.ModuleList(
            [MomentLayer_no_corr(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])
        return

    def forward(self, u, s):
        for i in range(len(self.layer_sizes) - 1):
            u, s = self.layers[i].forward(u, s)
        return u, s


class MultilayerPerceptron():
    @staticmethod
    def train(config):        
        if config['tensorboard']:
            writer = SummaryWriter(comment = str(config['trial_id']))#log_dir='D:\\mnn_py\\moment_activation\\runs2'
        
        num_batches = config['num_batches'] #500#20 need more data to train well
        batch_size = config['batch_size'] #64
        num_epoch = config['num_epoch'] #50#1000
        lr = config['lr']#0.01
        momentum = config['momentum'] #0.9
        optimizer_name = config['optimizer_name']
        
        if config['with_corr']:
            model = MoNet(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'])                    
        else:
            model = MoNet_no_corr(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'])        
            
        train_dataset = Dataset(config['dataset_name'], sample_size = num_batches*batch_size, input_dim = 2, output_dim = 1, with_corr = config['with_corr'])        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)        
    
        model.target_transform = train_dataset.transform
        
        validation_dataset = Dataset(config['dataset_name'], sample_size = 32, input_dim = 2, output_dim = 1, transform = train_dataset.transform, with_corr = config['with_corr'])        
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
                loss = loss_function_mse(u, s, sample['target_data'][0], sample['target_data'][1])
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
                        loss = loss_function_mse(u, s, sample['target_data'][0], sample['target_data'][1])
                    if config['tensorboard']:
                        writer.add_scalar("Loss/Validation", loss, epoch)
                        writer.flush()
                # ii = 0
                # for layer in model.layers:                
                #     #writer.add_scalar("Layer {} mu BN avg. weight".format(ii), layer.bn_mean.weight.data.mean() ,epoch)
                #     writer.add_scalar("Layer {} mu BN avg. bias".format(ii), layer.bn_mean.bias.data.mean() ,epoch)
                #     #writer.add_scalar("Layer {} std BN avg. weight".format(ii), layer.bn_std.weight.data.mean() ,epoch)
                #     writer.add_scalar("Layer {} std BN avg. bias".format(ii), layer.bn_std.bias.data.mean() ,epoch)                
                #     ii += 1                
                #writer.add_histogram('Batchnorm mean bias '+str(ii), layer.bn_mean.bias.data , epoch)
                
            #loss_values.append(loss.item())
            #for param in model.layers[0].bn_mean.named_parameters(): param[0] gives names, param[1] gives values
            
        
        #print("===============================")
        print("Number of batches: ", num_batches)
        print("Batch size: ",batch_size)
        print("Learning rate: ", lr)
        print("Momentum: ", momentum)
        print("Time Elapsed: ", time.perf_counter()-t0)
        print("===============================")
        
        #writer.add_graph(model, (input_mean,input_std))
        if config['tensorboard']:
            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            l1, = ax1.plot(u.detach().numpy(),s.detach().numpy(),'+')
            l2, = ax1.plot(sample['target_data'][0].detach().numpy(), sample['target_data'][1].detach().numpy(),'.')
            ax1.set_xlabel('Mean')
            ax1.set_ylabel('Std')
            plt.legend([l1,l2],['Output','Target'])    
            
            writer.add_figure('Final output vs target',fig)
            
            fig1 = VisualizationTools.plot_grid_map(model, with_corr = config['with_corr'])
            writer.add_figure('Test data grid map',fig1)
            
            writer.flush()
            writer.close()
        
        file_name = config['trial_id']
        torch.save(model.state_dict(), './data/regression/{}.pt'.format(file_name) ) #save result by time stamp
        with open('./data/regression/{}_config.json'.format(file_name),'w') as f:
            json.dump(config,f)
        
        return model

if __name__ == "__main__":    

    config = {'num_batches': 2000,
              'batch_size': 32,
              'num_epoch': 100,
              'lr': 0.01,
              'momentum': 0.9,
              'optimizer_name': 'Adam',
              'num_hidden_layers': 3,
              'hidden_layer_size': 32,
              'trial_id': int(time.time()),
              'tensorboard': True,
              'with_corr': False,
              'dataset_name': 'cue_combo'
        }
    
    model = MultilayerPerceptron.train(config)
    #runfile('./apps/regression/multilayer_perceptron.py', wdir='./')