import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from itertools import repeat

import torch.nn as nn
from torch.autograd import Function


class WeightBinarizerFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx is a context object that can be used to stash information
        # for backward computation
        return x.sign() * x.abs().mean()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output


class WeightBinarizer(nn.Module):
    def forward(self, input):
        return WeightBinarizerFunction.apply(input)

class ActivationBinarizerFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.input.clamp(-1, 1)
        return (2 - 2 * x * x.sign()) * grad_output


class ActivationBinarizer(nn.Module):
    def forward(self, input):
        return ActivationBinarizerFunction.apply(input)

class TernarizerFunction(Function):  # from Trained Ternary Quantization
    @staticmethod
    def forward(ctx, input, pos_value, neg_value, threshold):
        ctx.save_for_backward(input, pos_value, neg_value)
        ctx.threshold = threshold
        input[input.abs() < threshold] = 0
        input[input > threshold] = pos_value
        input[input < -threshold] = -neg_value
        return input

    @staticmethod
    def backward(ctx, grad_output):
        x, pos_value, neg_value = ctx.saved_tensors
        grad_pos = grad_output[x > ctx.threshold].sum()
        grad_neg = grad_output[x < -ctx.threshold].sum()
        grad_output[x > ctx.threshold] *= pos_value
        grad_output[x < -ctx.threshold] *= neg_value
        return grad_output, grad_pos, grad_neg, None


class Ternarizer(nn.Module):
    def __init__(self, threshold=None):
        super(Ternarizer, self).__init__()
        self.threshold = threshold
        self.pos_weight = nn.Parameter(torch.Tensor([1]))
        self.neg_weight = nn.Parameter(torch.Tensor([1]))

    def forward(self, input, threshold=None):
        if threshold is None:
            if self.threshold is not None:
                threshold = self.threshold
            else:
                raise TypeError('Threshold parameter is required')
        return TernarizerFunction.apply(input, self.pos_weight, self.neg_weight, threshold)

    def extra_repr(self):
        return 'positive weight={:.3f}, negative weight={:.3f}, threshold={:.3f}'.format(
            self.pos_weight, self.neg_weight, self.threshold)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Sign(nn.Module):
    def forward(self, input):
        x = input.clone()
        x[x >= 0] = 1
        x[x < 0] = -1
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
def binarize(input): # Simplest possible binarization
    return input.sign()

# Binary Conv2d is taken from:
# https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 activation_binarizer=ActivationBinarizer(), weight_binarizer=WeightBinarizer()):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.binarize_act = activation_binarizer
        self.binarize_w = weight_binarizer

    def forward(self, input):
        if input.size(1) != 3:
            input = self.binarize_act(input)
        if not hasattr(self.weight, 'original'):
            self.weight.original = self.weight.data.clone()
        self.weight.data = self.binarize_w(self.weight.original)
        #self.weight.data = binarize(self.weight.data)
        out = F.conv2d(input, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)

        if self.bias is not None:
            # self.bias.original = self.bias.data.clone() # do we need to save bias copy if it's not quantized?
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


class InferenceBinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(InferenceBinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # print(self.weight)
        self.binarize_act = Sign()

    def forward(self, input):
        if input.size(1) != 3:
            input = self.binarize_act(input)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# https://github.com/mzemlyanikin/binary-nets/blob/master/modules.py
# def binary_conv3x3(in_planes, out_planes, stride=1, groups=1, freeze=False, **kwargs):
def BinarizeConv2d_func(in_planes, out_planes,kernel_size=3, stride=1,padding=1,dilation=1, groups=1, bias=True, freeze=True, **kwargs):
    """3x3 convolution with padding"""
    if not freeze:
        return BinaryConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups,
                            padding=padding, bias=bias, **kwargs)
    else:
        return InferenceBinaryConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups,
                                     padding=padding, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


def binary_conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return BinaryConv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)