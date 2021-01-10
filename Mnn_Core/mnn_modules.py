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

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def _mnn_std_bn2d(self, module, mean, std):
        assert type(module).__name__ == "BatchNorm2d"
        self._check_input_dim(mean)
        self._check_input_dim(std)
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


class Mnn_Dropout(torch.nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(Mnn_Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

    def forward(self, mean, std):
        assert mean.size() == std.size()
        if self.training:
            temp = torch.zeros_like(mean)
            mean = F.dropout(mean, self.p, self.training, self.inplace)
            temp1 = torch.zeros_like(mean) + 1
            temp_map = torch.where(mean.clone().detach() != 0, temp1, temp)
            std = torch.mul(std, temp_map) / (1 - self.p)

        return mean, std


class Mnn_AvgPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(Mnn_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, mean: Tensor, std: Tensor):
        assert mean.dim() == 4
        assert mean.size() == std.size()
        mean = F.avg_pool2d(mean, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad, self.divisor_override)
        std = F.avg_pool2d(torch.pow(std, 2), self.kernel_size, self.stride, self.padding,
                           self.ceil_mode, self.count_include_pad, self.divisor_override)
        std = torch.sqrt(std / (self.kernel_size ** 2))
        return mean, std


class Mnn_AvgPool1d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(Mnn_AvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, mean: Tensor, std: Tensor):
        assert mean.size() == std.size()
        mean = F.avg_pool1d(mean, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)
        std = F.avg_pool1d(torch.pow(std, 2), self.kernel_size, self.stride, self.padding,
                           self.ceil_mode, self.count_include_pad)
        std = torch.sqrt(std / self.kernel_size)
        return mean, std


class Mnn_MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(Mnn_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = True
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, mean: Tensor, std: Tensor):
        assert mean.dim() == 4
        assert mean.size() == std.size()
        mean, indices = F.max_pool2d(mean, self.kernel_size, self.stride,
                                     self.padding, self.dilation, self.ceil_mode,
                                     self.return_indices)
        N, C, H, W = indices.size()
        temp_std = torch.zeros_like(mean)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        p = indices[n, c, h, w].item()
                        temp_std[n, c, h, w] = std[n, c, p//std.size()[2], p % std.size()[3]]
        return mean, temp_std


