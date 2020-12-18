from Mnn_Core.mnn_pytorch import *
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ray import tune
import json

torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)

# class MomentLayer_no_corr(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MomentLayer, self).__init__()
#         '''Ignores correlation completely'''
#         self.input_size = input_size
#         self.output_size = output_size

#         self.linear = Mnn_Linear_without_Corr(input_size, output_size)

#         self.bn_mean = torch.nn.BatchNorm1d(output_size)
#         self.bn_mean.weight.data.fill_(2.5)
#         self.bn_mean.bias.data.fill_(2.5)
#         #this roughly set the input mu in the range (0,5)

#         self.bn_std = torch.nn.BatchNorm1d(output_size)
#         self.bn_std.weight.data.fill_(2.5)
#         self.bn_std.bias.data.fill_(10.0)        
#         #intial bias for std should be large otherwise everything decays to zero.
        
#         #cache the output
#         self.mean = 0
#         self.std = 0
#         self.corr = 0

#         return

#     def forward(self, u, s):
#         u, s = self.linear.forward(u, s)
#         u = self.bn_mean(u)
#         s = self.bn_std(s)
#         u_activated = Mnn_Activate_Mean.apply(u, s)
#         s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
#         return u_activated, s_activated


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

        self.bn_std = torch.nn.BatchNorm1d(output_size)
        self.bn_std.weight.data.fill_(2.5)
        self.bn_std.bias.data.fill_(10.0)        
        #intial bias for std should be large otherwise everything decays to zero.
        
        #cache the output
        self.mean = 0
        self.std = 0
        self.corr = 0

        return
    
    def mybatchnorm(self, x, u):
        gam = self.bn_mean.weight
        
        if self.training:
            y = x/torch.std(u, 0, keepdim = True)*torch.abs(gam)
        else:
            y = x/torch.sqrt(self.bn_mean.running_var)*torch.abs(gam)
        return y
    
    def forward(self, u, s, rho):
        u, s, rho = self.linear.forward(u, s, rho)
        
        s = self.mybatchnorm(s, u)
        #s = self.bn_std(s)
        u = self.bn_mean(u)
                
        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
        corr_activated = Mnn_Activate_Corr.apply( rho , u, s, u_activated, s_activated)
        
        self.mean, self.std, self.corr = u_activated, s_activated, corr_activated
        
        return u_activated, s_activated, corr_activated

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

# class MoNet_no_corr(torch.nn.Module):
#     def __init__(self):
#         super(MoNet, self).__init__()
#         self.layer_sizes = [2]+[128]*10+[1]  # input, hidden, output
#         self.layers = torch.nn.ModuleList(
#             [MomentLayer(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])

#         return

#     def forward(self, u, s):
#         for i in range(len(self.layer_sizes) - 1):
#             u, s = self.layers[i].forward(u, s)
#         return u, s

class ProbInference():
    @staticmethod
    def synthetic_data(num_batches = 100, batch_size = 16 ):
        
        input_mean = 1 * torch.rand(batch_size, 2, num_batches) #NB: 0<min(input mean)< output mean < max(input mean)<1/Tref = 0.2
        input_mean[:,1,:] = 0.0 #fix one of the distributions to unit gaussian
        input_std = 2 * torch.rand(batch_size, 2, num_batches) + 1
        input_std[:,1,:] = 1.0 #fix one of the distributions to unit gaussian    
        
        fun1 = lambda mu1, mu2, v1, v2: (v2 * mu1 + v1 * mu2) / (v1 + v2)
        fun2 = lambda v1, v2: v1 * v2 / (v1 + v2)
    
        v = input_std ** 2
    
        target_mean = fun1(input_mean[:,0,:], input_mean[:,1,:], v[:,0,:], v[:,1,:]).view(batch_size, 1, num_batches)
        target_std = fun2(v[:, 0], v[:, 1]).pow(0.5).view(batch_size, 1, num_batches)
        
        #transduction_factor = 0.1
        #target_mean *= transduction_factor #scale to biological range
        #target_std *= transduction_factor #scale to biological range
            
        mean_scale = 1/(torch.max(target_mean)-torch.min(target_mean))*0.1
        mean_bias = - torch.min(target_mean)/(torch.max(target_mean)-torch.min(target_mean))*0.1+0.02
        std_scale = 1/(torch.max(target_std)-torch.min(target_std))*0.1
        std_bias = - torch.min(target_std)/(torch.max(target_std)-torch.min(target_std))*0.1+0.05
        
        target_mean = target_mean*mean_scale + mean_bias
        target_std = target_std*std_scale + std_bias
        
        input_corr = (torch.eye(2) + (1-torch.eye(2))*0.1).unsqueeze(0).repeat(batch_size,1,1) #NB must be symmetric and positive definite
        
        target_corr = [] #to be added
            
        return (input_mean, input_std, input_corr), (target_mean, target_std, target_corr), (mean_scale, mean_bias, std_scale, std_bias)
    
    @staticmethod
    def test_data(model):
        
        target_affine = model.target_affine
        
        fun1 = lambda mu1, mu2, v1, v2: (v2 * mu1 + v1 * mu2) / (v1 + v2)
        fun2 = lambda v1, v2: v1 * v2 / (v1 + v2)
        
        mean = torch.linspace(0,1,10)
        model.eval()
        
        input_corr = (torch.eye(2) + (1-torch.eye(2))*0.1).unsqueeze(0).repeat(100,1,1)
        
        fig1 = plt.figure()            
        ax1 = fig1.add_subplot(111)
        
        for i in range(len(mean)):
            input_std = torch.linspace(1,3,100)
            input_mean = torch.ones(input_std.shape)*mean[i]
            
            target_mean = fun1(input_mean, 0.0, input_std.pow(2), 1.0)
            target_std = fun2(input_std.pow(2), 1.0).pow(0.5)
            
            input_mean2 = torch.zeros(100,2)
            input_mean2[:,0] = input_mean
            input_std2 = torch.ones(100,2)
            input_std2[:,0] = input_std
            
            output_mean, output_std, output_corr = model.forward(input_mean2, input_std2, input_corr)
            
            
            target_mean = target_mean*target_affine[0] + target_affine[1]
            target_std = target_std*target_affine[2] + target_affine[3]
            
            ax1.plot(target_mean, target_std,'b', alpha=0.5)
            ax1.plot(output_mean.detach().numpy(), output_std.detach().numpy(), 'b')
        
        sigma = torch.linspace(1,3,10)
        
        for i in range(len(sigma)):
            input_mean = torch.linspace(0,1,100)
            input_std = torch.ones(input_mean.shape)*sigma[i]
            
            target_mean = fun1(input_mean, 0.0, input_std.pow(2), 1.0)
            target_std = fun2(input_std.pow(2), 1.0).pow(0.5)
            
            input_mean2 = torch.zeros(100,2)
            input_mean2[:,0] = input_mean
            input_std2 = torch.ones(100,2)
            input_std2[:,0] = input_std
            
            
            output_mean, output_std, output_corr = model.forward(input_mean2, input_std2, input_corr)        
            
            target_mean = target_mean*target_affine[0] + target_affine[1]
            target_std = target_std*target_affine[2] + target_affine[3]
            
            ax1.plot(target_mean, target_std,'r', alpha=0.4)
            ax1.plot(output_mean.detach().numpy(), output_std.detach().numpy(), 'r')
        
        ax1.set_xlabel('Output mean')
        ax1.set_ylabel('Output std')
        
        
        return fig1
    
    @staticmethod
    def plot_corr(model):
        fig_corr = plt.figure()            
        i=0
        for L in model.layers:
            i+=1
            if i > len(model.layers)-1:
                break
            ax1 = fig_corr.add_subplot(3,4,i)
            img = ax1.imshow(L.corr[0].detach().numpy(),vmax=1,vmin=-1,cmap = 'bwr')
            ax1.axis('off')
            ax1.set_title('Layer {}'.format(i))
        fig_corr.colorbar(img)
        return fig_corr
                
    @staticmethod
    def plot_weight(model):
        fig_weight = plt.figure()
        fig_whist = plt.figure()
        i=0
        for L in model.layers:
            i+=1
            if i > len(model.layers)-1:
                break
            
            
            #scale weight based on bn_mean (what about bn_std?)
            scale = L.bn_mean.weight/torch.sqrt(L.bn_mean.running_var)
            w = L.linear.weight*(scale.unsqueeze(0).T)
            w = w.detach().numpy()
            # print(bn.bias)
            # print(bn.running_mean)
            # print(bn.running_var)
            
            ax1 = fig_weight.add_subplot(3,4,i)        
            img = ax1.imshow(w,vmax=10,vmin=-10, cmap = 'bwr')                
            ax1.axis('off')
            ax1.set_title('Layer {}'.format(i))
            
            ax2 = fig_whist.add_subplot(3,4,i)
            img2 = ax2.hist(w.flatten(),np.linspace(-10,10,31))
            ax2.set_title('Layer {}'.format(i))
            
        fig_weight.colorbar(img)
        return fig_weight

    @staticmethod
    def train(config):
        
        writer = SummaryWriter(comment = str(config['trial_id']))#log_dir='D:\\mnn_py\\moment_activation\\runs2'
        
        num_batches = config['num_batches'] #500#20 need more data to train well
        batch_size = config['batch_size'] #64
        num_epoch = config['num_epoch'] #50#1000
        lr = config['lr']#0.01
        momentum = config['momentum'] #0.9
        optimizer_name = config['optimizer_name']
        
        model = MoNet(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'])
        
        train_input, train_target, target_affine = ProbInference.synthetic_data(num_batches = num_batches, batch_size = batch_size)
        model.target_affine = target_affine
        
        #param_container.set_ratio(0.1) # E/I balance. Mostly affect initialization. Mild value seems to improve result.
        #param_container.set_eps(0.01) # 1. Ensures sigma is positive; 2. avoid divide by 0 error. 3. Avoid overflow. Shouldn't need this to be too big. If error still occurs look for other sources.            
        #param_container.print_params()
        
        params = model.parameters()
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr = lr) #recommended lr: 0.1 (Adam requires a much smaller learning rate than SGD otherwise won't converge)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params, lr= lr, momentum= momentum) #recommended lr: 2
        else:
            optimizer = torch.optim.Adam(params, lr = lr)
            
        
        t0 = time.perf_counter()
        for epoch in range(num_epoch):
            for j in range(num_batches):
                optimizer.zero_grad()
                u, s, rho = model.forward(train_input[0][:,:,j], train_input[1][:,:,j], train_input[2])
                loss = loss_function_mse(u, s, train_target[0][:,:,j], train_target[1][:,:,j])
                loss.backward()
                optimizer.step()
        
            writer.add_scalar("Loss/Train", loss, epoch)
            writer.flush()
            
            #tune.report(loss = loss.item())
            
            if epoch % 1 == 0:
                print('Training epoch {}/{}'.format(epoch,num_epoch))
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
    
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        l1, = ax1.plot(u.detach().numpy(),s.detach().numpy(),'+')
        l2, = ax1.plot(train_target[0][:,:,-1].detach().numpy(),train_target[1][:,:,-1].detach().numpy(),'.')
        ax1.set_xlabel('Mean')
        ax1.set_ylabel('Std')
        plt.legend([l1,l2],['Output','Target'])    
        
        writer.add_figure('Final output vs target',fig)
        
        fig1 = ProbInference.test_data(model)
        writer.add_figure('Test data grid map',fig1)
        
        writer.flush()
        writer.close()
        
        file_name = config['trial_id']
        torch.save(model.state_dict(), './data/regression/{}.pt'.format(file_name) ) #save result by time stamp
        with open('./data/regression/{}_config.json'.format(file_name),'w') as f:
            json.dump(config,f)
        
        return model

if __name__ == "__main__":    

    config = {'num_batches': 100,
              'batch_size': 64,
              'num_epoch': 30,
              'lr': 0.01,
              'momentum': 0.9,
              'optimizer_name': 'Adam',
              'num_hidden_layers': 5,
              'hidden_layer_size': 64,
              'trial_id': int(time.time())
        }
    
    model = ProbInference.train(config)
    #runfile('./apps/regression/probability_inference.py', wdir='./')