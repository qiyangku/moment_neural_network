from Mnn_Core.mnn_pytorch import *
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)


class MomentLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MomentLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = Mnn_Linear_without_Corr(input_size, output_size)

        self.bn_mean = torch.nn.BatchNorm1d(output_size)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)
        #this roughly set the input mu in the range (0,5)

        self.bn_std = torch.nn.BatchNorm1d(output_size)
        self.bn_std.weight.data.fill_(2.5)
        self.bn_std.bias.data.fill_(10.0)        
        #intial bias for std should be large otherwise everything decays to zero.

        return

    def forward(self, u, s):
        u, s = self.linear.forward(u, s)
        u = self.bn_mean(u)
        s = self.bn_std(s)
        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)
        return u_activated, s_activated


class MoNet(torch.nn.Module):
    def __init__(self):
        super(MoNet, self).__init__()
        self.layer_sizes = [2]+[64]*30+[1]  # input, hidden, output
        self.layers = torch.nn.ModuleList(
            [MomentLayer(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])

        return

    def forward(self, u, s):
        for i in range(len(self.layer_sizes) - 1):
            u, s = self.layers[i].forward(u, s)
        return u, s


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
        
    return input_mean, input_std, target_mean, target_std, (mean_scale, mean_bias, std_scale, std_bias)

def test_data(model, target_affine):
    
    
    fun1 = lambda mu1, mu2, v1, v2: (v2 * mu1 + v1 * mu2) / (v1 + v2)
    fun2 = lambda v1, v2: v1 * v2 / (v1 + v2)
    
    mean = torch.linspace(0,1,10)
    model.eval()
    
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
        
        output_mean, output_std = model.forward(input_mean2,input_std2)
        
        
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
        
        
        output_mean, output_std = model.forward(input_mean2,input_std2)
        
        
        target_mean = target_mean*target_affine[0] + target_affine[1]
        target_std = target_std*target_affine[2] + target_affine[3]
        
        ax1.plot(target_mean, target_std,'r', alpha=0.5)
        ax1.plot(output_mean.detach().numpy(), output_std.detach().numpy(), 'r')
    
    return fig1

if __name__ == "__main__":    
    model = MoNet()
    writer = SummaryWriter()#log_dir='D:\\mnn_py\\moment_activation\\runs2'
    
    
    num_batches = 20#20 need more data to train well
    batch_size = 64#64
    num_epoch = 100#1000
    
    input_mean, input_std, target_mean, target_std, target_affine = synthetic_data(num_batches = num_batches, batch_size = batch_size)
    #param_container.set_ratio(0.1) # E/I balance. Mostly affect initialization. Mild value seems to improve result.
    #param_container.set_eps(0.01) # 1. Ensures sigma is positive; 2. avoid divide by 0 error. 3. Avoid overflow. Shouldn't need this to be too big. If error still occurs look for other sources.
    
    lr = 0.01
    momentum = 0.9
        
    #param_container.print_params()
    params = model.parameters()
    #optimizer = torch.optim.SGD(params, lr= 2, momentum= momentum) #recommended lr: 2
    optimizer = torch.optim.Adam(params, lr = lr) #recommended lr: 0.1 (Adam requires a much smaller learning rate than SGD otherwise won't converge)
    
    #loss_values = []
    
    t0 = time.perf_counter()
    for epoch in range(num_epoch):
        for j in range(num_batches):
            optimizer.zero_grad()
            u, s = model.forward(input_mean[:,:,j],input_std[:,:,j])
            loss = loss_function_mse(u, s, target_mean[:,:,j], target_std[:,:,j])
            loss.backward()
            optimizer.step()
    
        writer.add_scalar("Loss/Train", loss, epoch)
        writer.flush()
        
        if epoch % 10 == 0:
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
        
    
    print("===============================")
    print("Number of batches: ", num_batches)
    print("Batch size: ",batch_size)
    print("Learning rate: ", lr)
    print("Momentum: ", momentum)
    print("Time Elapsed: ", time.perf_counter()-t0)
    print("===============================")
    
    #writer.add_graph(model, (input_mean,input_std))

    #        print("===============================")
    #        print("Weight of mnn_linear1:", mnn_linear1.weight)
    #        print("===============================")
    #plt.clf('all')
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # ax1.semilogy(loss_values)
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # l1, = ax2.plot(u.detach().numpy(),s.detach().numpy(),'.')
    # l2, = ax2.plot(target_mean.detach().numpy(),target_std.detach().numpy(),'.')
    # ax2.set_xlabel('Mean')
    # ax2.set_ylabel('Std')
    # plt.legend([l1,l2],['Output','Target'])    
    #plt.show()
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    l1, = ax1.plot(u.detach().numpy(),s.detach().numpy(),'+')
    l2, = ax1.plot(target_mean[:,:,-1].detach().numpy(),target_std[:,:,-1].detach().numpy(),'.')
    ax1.set_xlabel('Mean')
    ax1.set_ylabel('Std')
    plt.legend([l1,l2],['Output','Target'])    
    
    writer.add_figure('Final output vs target',fig)
    
    fig1 = test_data(model, target_affine)
    writer.add_figure('Test data grid map',fig1)
    
    writer.flush()
    writer.close()
    
    #runfile('./apps/regression/probability_inference.py', wdir='./')