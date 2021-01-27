# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 01:05:10 2021

@author: dell
"""
import torch
from Mnn_Core.mnn_utils import *
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import torch.fft

class Mnn_Conv1d_fft(torch.nn.Module):
    __constants__ = ['in_features']
    in_features: int    
    weight: Tensor

    def __init__(self, in_features: int, bias: bool = False) -> None:
        super(Mnn_Conv1d_fft, self).__init__()
        self.in_features = in_features
        #self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #init.kaiming_uniform_(self.weight, a=np.sqrt(5)) #not applicable for 1d weight
        init.uniform_(self.weight, -5, 5)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, mean_in: Tensor, std_in, corr_in: Tensor):
        assert mean_in.size() == std_in.size()
        
        # Use corr_in and std to compute the covariance matrix
        wfft = torch.fft.fft(self.weight)
        if std_in.dim() == 1:                  
            mean_in_fft = torch.fft.fft(mean_in)
            mean_out = torch.real(torch.fft.ifft( wfft*mean_in_fft))            
            cov_in = corr_in*std_in.view(self.in_features,1)*std_in.view(1,self.in_features)
            Cfft = torch.fft.fft( torch.fft.fft(cov_in, dim = 0), dim = 1)        
            Cfft = wfft.view(self.in_features,1)*Cfft*wfft.view(1,self.in_features)                
            cov_out = torch.real(torch.fft.ifft(torch.fft.ifft(Cfft, dim = 0), dim =1))
        else:
            #>>>> FFT-based summation layer            
            mean_in_fft = torch.fft.fft(mean_in, dim = 1)
            mean_out = torch.real(torch.fft.ifft( wfft.unsqueeze(0)*mean_in_fft , dim = 1 ))            
            cov_in = corr_in*std_in.unsqueeze(1)*std_in.unsqueeze(2)       
            Cfft = torch.fft.fft( torch.fft.fft(cov_in, dim = 1), dim = 2)        
            Cfft = wfft.unsqueeze(0).unsqueeze(1)*Cfft*wfft.unsqueeze(0).unsqueeze(2)
            cov_out = torch.real(torch.fft.ifft(torch.fft.ifft(Cfft, dim = 1), dim =2))
        
        if self.bias is not None:
            bias = self.bias.view(1, -1)
            bias = torch.mm(bias.transpose(1, 0), bias)
            cov_out += bias
        if cov_out.dim() == 2:  # one sample case
            var_out = torch.diagonal(cov_out)
            # prevent negative value
            std_out = torch.sqrt(torch.abs(var_out))
            temp_std_out = std_out.view(1, -1)
            temp_std_out = torch.mm(temp_std_out.transpose(1, 0), temp_std_out)
            corr_out = torch.div(cov_out, temp_std_out)
        else:
            var_out = torch.diagonal(cov_out, dim1=-2, dim2=-1)
            std_out = torch.sqrt(var_out) #removed the abs for debug purpose
            corr_out = cov_out/std_out.unsqueeze(1)/std_out.unsqueeze(2)

        return mean_out, std_out, corr_out

    def extra_repr(self) -> str:
        return 'in_features={}, bias={}'.format(
            self.in_features, self.bias is not None
        )