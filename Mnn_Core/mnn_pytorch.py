# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:49:43 2020

@author: zzc14
"""

from Mnn_Core.mnn_utils import *
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init

torch.set_default_tensor_type(torch.DoubleTensor)
mnn_core_func = Mnn_Core_Func()


class Mnn_Linear_without_Corr(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False, ext_bias: bool = False) -> None:
        super(Mnn_Linear_without_Corr, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if ext_bias:
            self.ext_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("ext_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(15))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.ext_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.ext_bias, 0, bound)

    def forward(self, input1: Tensor, input2: Tensor):
        assert input1.size() == input2.size()
        out1 = F.linear(input1, self.weight, self.bias)
        if self.ext_bias is not None:
            out2 = F.linear(torch.pow(input2, 2), torch.pow(self.weight, 2), torch.pow(self.ext_bias, 2))
        else:
            out2 = F.linear(torch.pow(input2, 2), torch.pow(self.weight, 2), self.ext_bias)
        out2 = torch.sqrt(out2)
        return out1, out2

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Mnn_Linear_Corr(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False, ext_std: bool = False) -> None:
        super(Mnn_Linear_Corr, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.ratio = mnn_core_func.get_ratio()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if ext_std:
            self.ext_std = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("ext_std", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.ext_std is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.ext_std, 0, bound)

    def forward(self, mean_in: Tensor, std_in, corr_in: Tensor):
        assert mean_in.size() == std_in.size()
        # ratio not used for std and corr
        mean_out = F.linear(mean_in, self.weight, self.bias) * (1 - self.ratio)
        # Use corr_in and std to compute the covariance matrix
        cov_in = get_cov_matrix(std_in, corr_in)
        # cov_out = W C W^T
        cov_out = torch.matmul(self.weight, torch.matmul(cov_in, self.weight.transpose(1, 0)))
        std_out, corr_out = update_correlation(cov_out, self.ext_bias)

        return mean_out, std_out, corr_out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Mnn_Activate_Mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_in, std_in):
        clone_mean = mean_in.detach().numpy()
        clone_std = std_in.detach().numpy()
        shape = clone_mean.shape

        # Todo Should remove flatten op to save time
        clone_mean = clone_mean.flatten()
        clone_std = clone_std.flatten()

        mean_out = mnn_core_func.forward_fast_mean(clone_mean, clone_std)

        # Todo Should remove flatten op to save time
        mean_out = torch.from_numpy(mean_out.reshape(shape))

        ctx.save_for_backward(mean_in, std_in, mean_out)
        return mean_out

    @staticmethod
    def backward(ctx, grad_output):
        mean_in, std_in, mean_out = ctx.saved_tensors
        clone_mean_in = mean_in.detach().numpy()
        clone_std_in = std_in.detach().numpy()
        clone_mean_out = mean_out.detach().numpy()

        # Todo Should remove flatten op to save time
        shape = clone_std_in.shape
        clone_mean_in = clone_mean_in.flatten()
        clone_std_in = clone_std_in.flatten()
        clone_mean_out = clone_mean_out.flatten()

        grad_mean, grad_std = mnn_core_func.backward_fast_mean(clone_mean_in, clone_std_in, clone_mean_out)
        # Todo Should remove flatten op to save time
        grad_mean = torch.from_numpy(grad_mean.reshape(shape))
        grad_std = torch.from_numpy(grad_std.reshape(shape))
        grad_mean = torch.mul(grad_output, grad_mean)
        grad_std = torch.mul(grad_output, grad_std)
        return grad_mean, grad_std


class Mnn_Activate_Std(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_in, std_in, mean_out):
        clone_mean = mean_in.detach().numpy()
        clone_std = std_in.detach().numpy()
        clone_mean_out = mean_out.detach().numpy()
        shape = clone_mean.shape

        # Todo Should remove flatten op to save time
        clone_mean = clone_mean.flatten()
        clone_std = clone_std.flatten()
        clone_mean_out = clone_mean_out.flatten()

        std_out = mnn_core_func.forward_fast_std(clone_mean, clone_std, clone_mean_out)
        # Todo Should remove flatten op to save time
        std_out = torch.from_numpy(std_out.reshape(shape))
        ctx.save_for_backward(mean_in, std_in, mean_out, std_out)
        return std_out

    @staticmethod
    def backward(ctx, grad_output):
        mean_in, std_in, mean_out, std_out = ctx.saved_tensors
        clone_mean_in = mean_in.detach().numpy()
        clone_std_in = std_in.detach().numpy()
        clone_mean_out = mean_out.detach().numpy()
        clone_std_out = std_out.detach().numpy()
        # Todo Should remove flatten op to save time
        shape = clone_std_in.shape
        clone_mean_in = clone_mean_in.flatten()
        clone_std_in = clone_std_in.flatten()
        clone_mean_out = clone_mean_out.flatten()
        clone_std_out = clone_std_out.flatten()

        std_grad_mean, std_grad_std = mnn_core_func.backward_fast_std(clone_mean_in, clone_std_in, clone_mean_out,
                                                                      clone_std_out)
        # Todo Should remove flatten op to save time
        std_grad_mean = torch.from_numpy(std_grad_mean.reshape(shape))
        std_grad_std = torch.from_numpy(std_grad_std.reshape(shape))
        std_grad_mean = torch.mul(grad_output, std_grad_mean)
        std_grad_std = torch.mul(grad_output, std_grad_std)

        grad_mean_out = torch.zeros_like(std_grad_mean)
        return std_grad_mean, std_grad_std, grad_mean_out


class Mnn_Activate_Corr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, corr_in, mean_in, std_in, mean_out, std_out):
        """
        corr_in: The covariance matrix that passed the Mnn_Linear_Cov layer
        mean_bn_in: the mean vector that passed the batch normalization layer
        std_bn_in: the std vector that passed the batch normalization layer

        The following variable should pass by using clone().detach() function (require no gradient)
        mean_out : the mean vector that is activated by Mnn_Activate_Mean
        std_out : the std vector that is activated by Mnn_Activate_Std
        """

        # Compute the chi function
        clone_mean_in = mean_in.detach().numpy()
        clone_std_in = std_in.detach().numpy()
        clone_mean_out = mean_out.detach().numpy()
        clone_std_out = std_out.detach().numpy()
        shape = clone_mean_in.shape
        clone_mean_in = clone_mean_in.flatten()
        clone_mean_out = clone_mean_out.flatten()
        clone_std_in = clone_std_in.flatten()
        clone_std_out = clone_std_out.flatten()

        func_chi = mnn_core_func.forward_fast_chi(clone_mean_in, clone_std_in, clone_mean_out, clone_std_out)
        # func_chi = np.nan_to_num(func_chi)
        func_chi = torch.from_numpy(func_chi.reshape(shape))

        # Compute the Cov of next layer
        # One sample case
        if func_chi.dim() == 1:
            temp_func_chi = func_chi.view(1, -1)
            temp_func_chi = torch.mm(temp_func_chi.transpose(1, 0), temp_func_chi)
        # Multi sample case
        else:
            temp_func_chi = func_chi.view(func_chi.size()[0], 1, func_chi.size()[1])
            temp_func_chi = torch.bmm(temp_func_chi.transpose(-1, -2), temp_func_chi)
        corr_out = torch.mul(corr_in, temp_func_chi)

        # replace the diagonal elements with 1
        if corr_out.dim() == 2:
            corr_out = corr_out.fill_diagonal_(1.0)

        else:
            torch.diagonal(corr_out, dim1=1, dim2=2).data.fill_(1.0)
        ctx.save_for_backward(corr_in, mean_in, std_in, mean_out, func_chi)
        return corr_out

    # require  the gradient of corr_in, mean_bn_in,  std_bn_in
    @staticmethod
    def backward(ctx, grad_out):
        corr_in, mean_in, std_in, mean_out, func_chi = ctx.saved_tensors
        clone_mean_in = mean_in.detach().numpy()
        clone_std_in = std_in.detach().numpy()
        clone_mean_out = mean_out.detach().numpy()
        clone_func_chi = func_chi.detach().numpy()
        shape = clone_std_in.shape

        # Todo unnecessary flatten operation, need to be optimised
        clone_mean_in = clone_mean_in.flatten()
        clone_std_in = clone_std_in.flatten()
        clone_mean_out = clone_mean_out.flatten()
        clone_func_chi = clone_func_chi.flatten()

        chi_grad_mean, chi_grad_std = mnn_core_func.backward_fast_chi(clone_mean_in, clone_std_in,
                                                                      clone_mean_out, clone_func_chi)
        chi_grad_mean = torch.from_numpy(chi_grad_mean.reshape(shape))
        chi_grad_std = torch.from_numpy(chi_grad_std.reshape(shape))

        if corr_in.dim() != 2:
            torch.diagonal(corr_in, dim1=1, dim2=2).data.fill_(0.0)
        else:
            corr_in = corr_in.fill_diagonal_(0.0)

        temp_corr_grad = torch.mul(grad_out, corr_in)

        if temp_corr_grad.dim() == 2:  # one sample case
            temp_corr_grad = torch.mm(func_chi.view(1, -1), temp_corr_grad)
        else:
            temp_corr_grad = torch.bmm(func_chi.view(func_chi.size()[0], 1, -1), temp_corr_grad)
        # reshape the size from (batch, 1, feature) to (batch, feature)
        temp_corr_grad = 2 * temp_corr_grad.view(temp_corr_grad.size()[0], -1)

        corr_grad_mean = chi_grad_mean * temp_corr_grad
        corr_grad_std = chi_grad_std * temp_corr_grad

        if func_chi.dim() == 1:
            temp_func_chi = func_chi.view(1, -1)
            chi_matrix = torch.mm(temp_func_chi.transpose(1, 0), temp_func_chi)
        else:
            temp_func_chi = func_chi.view(func_chi.size()[0], 1, -1)
            chi_matrix = torch.bmm(temp_func_chi.transpose(-2, -1), temp_func_chi)

        corr_grad_corr = torch.mul(chi_matrix, grad_out)
        # set the diagonal element of corr_grad_corr to 0
        if corr_grad_corr.dim() != 2:
            torch.diagonal(corr_grad_corr, dim1=1, dim2=2).data.fill_(0.0)
        else:
            corr_grad_corr = corr_grad_corr.fill_diagonal_(0.0)

        grad_mean_out = torch.zeros_like(mean_out)
        grad_std_out = torch.zeros_like(mean_out)
        return corr_grad_corr, corr_grad_mean, corr_grad_std, grad_mean_out, grad_std_out


class Mnn_Std_Bn1d(torch.nn.Module):
    def __init__(self, features: int, ext_bias=True):
        super(Mnn_Std_Bn1d, self).__init__()
        self.features = features
        if ext_bias:
            self.ext_bias = Parameter(torch.Tensor(features))
        else:
            self.register_parameter('ext_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.ext_bias is not None:
            init.uniform_(self.ext_bias, 2, 10)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def _mnn_std_bn1d(self, module, mean, std):
        assert type(module).__name__ == "BatchNorm1d"
        self._check_input_dim(mean)
        assert mean.size() == std.size()
        if module.training:
            std = torch.pow(std, 2) * torch.pow(module.weight, 2) / (torch.var(mean, dim=0, keepdim=True) + module.eps)
            if self.ext_bias is not None:
                std += torch.pow(self.ext_bias, 2)
        else:
            if module.track_running_stats is True:
                std = torch.pow(std, 2) * torch.pow(module.weight, 2) / (module.running_var + module.eps)
            else:
                std = torch.pow(std, 2) * torch.pow(module.weight, 2) / (
                        torch.var(mean, dim=0, keepdim=True) + module.eps)
            if self.ext_bias is not None:
                std += torch.pow(self.ext_bias, 2)
        std = torch.sqrt(std)
        return std

    def forward(self, module, mean, std):
        return self._mnn_std_bn1d(module, mean, std)


class Mnn_Layer_without_Rho(torch.nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super(Mnn_Layer_without_Rho, self).__init__()
        self.fc = Mnn_Linear_without_Corr(d_in, d_out, bias=bias)
        self.bn_mean = torch.nn.BatchNorm1d(d_out)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)

        self.bn_std = Mnn_Std_Bn1d(d_out)
        self.a1 = Mnn_Activate_Mean.apply
        self.a2 = Mnn_Activate_Std.apply

    def forward(self, ubar, sbar):
        assert ubar.size() == sbar.size()
        ubar, sbar = self.fc(ubar, sbar)
        uhat = self.bn_mean(ubar)
        shat = self.bn_std(self.bn_mean, ubar, sbar)
        u = self.a1(uhat, shat)
        s = self.a2(uhat, shat, u)
        return u, s

