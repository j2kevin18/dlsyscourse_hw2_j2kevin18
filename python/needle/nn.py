"""The module.
"""
from cmath import log
from os import device_encoding
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            # print(params, v)
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), dtype=dtype)
        if bias == True:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape((1, out_features)), dtype=dtype)
        else:
            self.bias = Parameter(init.zeros(out_features, 1).reshape((1, out_features)), dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        X.grad = X
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x * Tensor((x.numpy() > 0).astype(np.float))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot = Tensor((np.arange(logits.shape[1])==y.numpy()[:, None]).astype(np.float32))
        return ops.summation(ops.logsumexp(logits, axes=1)-ops.summation(logits*one_hot, axes=1)) / y.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), dtype=dtype)
        self.bias = Parameter(init.zeros(dim), dtype=dtype)
        self.running_mean = init.zeros(dim, dtype=dtype)
        self.running_var = init.ones(dim, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = init.zeros(*x.shape)
        if self.training == True:
            mean_pre = (x.sum(axes=0) / x.shape[0])
            mean = mean_pre.reshape((1, x.shape[1])).broadcast_to(x.shape)
            var_pre = (((x-mean)**2).sum(axes=0) / x.shape[0])
            var = var_pre.reshape((1, x.shape[1])).broadcast_to(x.shape)
            # print(mean, var)
            w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
            b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
            y = w * (x - mean) / ((var + self.eps) ** 0.5) + b
            
            self.running_mean = self.running_mean.data * (1 - self.momentum) + mean_pre * self.momentum
            self.running_var = self.running_var.data * (1 - self.momentum) + var_pre * self.momentum
        else:
            x_normalize = (x - self.running_mean.data) / ((self.running_var.data + self.eps) ** 0.5) 
            w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
            b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
            y = w * x_normalize + b
        return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), dtype=dtype)
        self.bias = Parameter(init.zeros(dim), dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = np.zeros(x.shape)
        mean = (x.sum(axes=1) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x-mean)**2).sum(axes=1) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        y = w * (x - mean) / ((var + self.eps) ** 0.5) + b
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == True:
            mask = Tensor(init.randb(*x.shape, p=1-self.p)) / (1 - self.p)
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



