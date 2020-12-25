# -*- coding: utf-8 -*-
# Copyright 2020 Zhu Zhichao, ISTBI, Fudan University China

from Mnn_Core.mnn_pytorch import *
import itertools


def _ntuple(n):
    def parse(x):
        if isinstance(x, torch._six.container_abcs.Iterable):
            return x
        return tuple(itertools.repeat(x, n))
    return parse


_pair = _ntuple(2)


class Mnn_Std_Conv2d_without_Rho(torch.nn.Module):
    def __init__(self, out_channels, ext_bias=True):
        super(Mnn_Std_Conv2d_without_Rho, self).__init__()
        if ext_bias:
            self.ext_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("ext_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.ext_bias is not None:
            init.uniform_(self.ext_bias, 0.1, 3)
    
    def _std_conv_forward(self, module, inp):
        assert type(module).__name__ == "Conv2d"
        if module.padding_mode != "zeros":
            padding_repeat = tuple(x for x in reversed(module.padding) for _ in range(2))
            if self.ext_bias is not None:
                out = F.conv2d(F.pad(inp, padding_repeat, mode=module.padding_mode), torch.pow(module.weight, 2),
                               torch.pow(self.ext_bias, 2), module.stride, _pair(0), module.dilation, module.groups)
            else:
                out = F.conv2d(F.pad(inp, padding_repeat, mode=module.padding_mode), torch.pow(module.weight, 2),
                               self.ext_bias, module.stride, _pair(0), module.dilation, module.groups)

        else:
            if self.ext_bias is not None:
                out = F.conv2d(inp, torch.pow(module.weight, 2), torch.pow(self.ext_bias, 2), module.stride,
                               module.padding, module.dilation, module.groups)
            else:
                out = F.conv2d(inp, torch.pow(module.weight, 2), self.ext_bias, module.stride, module.padding,
                               module.dilation, module.groups)

        out = torch.sqrt(out)
        return out

    def forward(self, module, inp):
        return self._std_conv_forward(module, inp)


class Mnn_Std_Bn2d(torch.nn.Module):
    def __init__(self, features: int, ext_bias=True):
        super(Mnn_Std_Bn2d, self).__init__()
        self.features = features
        if ext_bias:
            self.ext_bias = Parameter(torch.Tensor(features))
        else:
            self.register_parameter('ext_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.ext_bias is not None:
            init.uniform_(self.ext_bias, 2, 10)

    def _mnn_std_bn2d(self, module, mean, std):
        assert type(module).__name__ == "BatchNorm2d"
        if module.training:
            running_var = torch.zeros(self.features)
            for i in range(self.features):
                feature_map = mean[:, i, :, :]
                running_var[i] = torch.var(feature_map, unbiased=False)

        else:
            if module.track_running_stats is True:
                running_var = module.running_var
            else:
                running_var = torch.zeros(self.features)
                for i in range(self.features):
                    feature_map = mean[:, i, :, :]
                    running_var[i] = torch.var(feature_map, unbiased=False)
        
        out = torch.zeros_like(std)
        for i in range(self.features):
            x = std[:, i, :, :]
            x = torch.pow(x, 2)
            out[:, i, :, :] = x / (running_var[i] + module.eps) * torch.pow(module.weight[i], 2)
            if self.ext_bias is not None:
                out[:, i, :, :] += torch.pow(self.ext_bias[i], 2)

        return torch.sqrt(out)

    def forward(self, module, mean, std):
        return self._mnn_std_bn2d(module, mean, std)



