# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 21:01:40 2020
Simple gradcheck for corr map with base numpy version
@author: dell
"""

from Mnn_Core.maf import *
from Mnn_Core.mnn_pytorch import *

maf = MomentActivation()


def gradcheck_numpy(inputs):
    r = inputs[2]  # np.random.rand(N)*2-1 #scalar random corr

    u = inputs[0]  # np.random.randn(N,2) #mean
    s = inputs[1]  # np.random.rand(N,2)+1 #std

    # base forward
    _ = maf.mean(u, s)
    _ = maf.std(u, s)
    chi = maf.chi(u, s)

    y = chi[:, 0] * chi[:, 1] * r

    # perturbed forward
    u_new = u.copy()
    du = 1e-5
    u_new[:, 0] += du
    _ = maf.mean(u_new, s)
    _ = maf.std(u_new, s)
    chi_new = maf.chi(u_new, s)
    y_new = chi_new[:, 0] * chi_new[:, 1] * r

    # numeric_grad
    numeric_grad = 2 * (y_new - y) / du

    # analytic grad
    _ = maf.mean(u, s)
    _ = maf.std(u, s)
    _ = maf.chi(u, s)
    _ = maf.grad_mean(u, s)

    chi_grad_u, chi_grad_s = maf.grad_chi(u, s)

    analytic_grad = 2 * chi_grad_u[:, 0] * chi[:, 1] * r

    return numeric_grad, analytic_grad


def gradcheck_mnn_corr(inputs):
    # repeat the same test with maf from mnn_corr to make sure maf is correctly implemented

    r = inputs[2]  # np.random.rand(N)*2-1 #scalar random corr

    u = inputs[0]  # np.random.randn(N,2) #mean
    s = inputs[1]  # np.random.rand(N,2)+1 #std

    # base forward
    mean_out = mnn_core_func.forward_fast_mean(u, s)
    std_out = mnn_core_func.forward_fast_std(u, s, mean_out)
    chi_out = mnn_core_func.forward_fast_chi(u, s, mean_out, std_out)

    y = chi_out[:, 0] * chi_out[:, 1] * r

    # perturbed forward
    u_new = u.copy()
    du = 1e-8
    u_new[:, 0] += du
    mean_out_new = mnn_core_func.forward_fast_mean(u_new, s)
    std_out_new = mnn_core_func.forward_fast_std(u_new, s, mean_out_new)
    chi_out_new = mnn_core_func.forward_fast_chi(u_new, s, mean_out_new, std_out_new)
    y_new = chi_out_new[:, 0] * chi_out_new[:, 1] * r

    # numeric_grad
    numeric_grad = 2 * (y_new - y) / du

    # analytic grad
    chi_grad_mean, chi_grad_std = mnn_core_func.backward_fast_chi(u, s, mean_out, chi_out)

    analytic_grad = 2 * chi_grad_mean[:, 0] * chi_out[:, 1] * r

    return numeric_grad, analytic_grad


def check_corr_grad_pytorch(u: Tensor, s: Tensor, r: Tensor, pos=0):
    u_new = u.clone().detach()
    s_new = s.clone().detach()
    r_new = r.clone().detach()

    a1 = Mnn_Activate_Mean.apply
    a2 = Mnn_Activate_Std.apply
    a3 = Mnn_Activate_Corr.apply

    u.requires_grad = True
    s.requires_grad = True
    r.requires_grad = True

    mean_out = a1(u, s)
    std_out = a2(u, s, mean_out)
    corr_out = a3(r, u, s, mean_out, std_out)
    loss = F.mse_loss(corr_out, corr_out)
    loss.backward()

    eps = 1e-10
    u_new[pos] += eps
    mean_out_new = a1(u_new, s_new)
    std_out_new = a2(u_new, s_new, mean_out_new)
    corr_out_new = a3(r_new, u_new, s_new, mean_out_new, std_out_new)

    numeric_grad = torch.sum(2*(corr_out_new - corr_out)/eps, dim=0)
    numeric_grad, analytic_grad = numeric_grad[pos].item(), u.grad[pos].item()
    print('Numeric grad using mnn_corr_func:\n', numeric_grad)
    print('Analytic grad using mnn_corr_func:\n', analytic_grad)


if __name__ == '__main__':
    neuron = 10
    torch.set_printoptions(precision=8)
    u = torch.randn(neuron) * 2
    s = torch.rand(neuron) + 1
    r = (torch.rand(neuron, neuron) - 0.5)
    r = r + r.transpose(1, 0)
    for i in range(neuron):
        r[i, i] = 1.0
    check_corr_grad_pytorch(u, s, r)



