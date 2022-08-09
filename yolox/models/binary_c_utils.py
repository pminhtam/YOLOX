"""
This file contains code for implementing bi-real net architectures.
Credit to Mr.Daquexian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal_50/birealnet.py
class Leakyclip(nn.Module):
    def __init__(self):
        super(Leakyclip, self).__init__()

    def forward(self, x):
         out = x
         mask1 = x < -1
         mask2 = x > 1
         out1 = (0.1 * x - 0.9) * mask1.type(torch.float32) + x * (1 - mask1.type(torch.float32))
         out2 = (0.1 * out1 + 0.9) * mask2.type(torch.float32) + out1 * (1 - mask2.type(torch.float32))
         out = out2
         return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,dilation=1,groups=1, bias=True):
        super(BinarizeConv2d, self).__init__(
                        in_chn, out_chn, kernel_size, stride, padding,dilation,groups, bias)
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weight = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weight)
        cliped_weights = torch.clamp(real_weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding,groups=self.groups,bias=self.bias)
        return y

