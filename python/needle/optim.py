"""Optimization module"""
from cmath import nan
from turtle import window_height
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for theta in self.params:
            grad = theta.grad.data + self.weight_decay * theta.data
            if theta not in self.u:
                self.u[theta] = (1. - self.momentum) * grad
            else:
                self.u[theta] = self.u[theta] * self.momentum + (1. - self.momentum) * grad

            theta_val = theta.data - self.lr * self.u[theta]
            theta.data = ndl.Tensor(theta_val.numpy().astype(np.float32))
            # print(theta.dtype)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        
        for theta in self.params:
            grad = theta.grad.data + self.weight_decay * theta.data
            if theta not in self.m:
                self.m[theta] = (1. - self.beta1) * grad
            else:
                self.m[theta] = self.m[theta] * self.beta1 + (1 - self.beta1) * grad
            
            if theta not in self.v:
                self.v[theta] = (1. - self.beta2) * (grad ** 2)
            else:
                self.v[theta] = self.v[theta] * self.beta2 + (1 - self.beta2) * (grad ** 2)

            m_next_hat = self.m[theta] / (1 - self.beta1 ** self.t)
            v_next_hat = self.v[theta] / (1 - self.beta2 ** self.t)
            theta_val = theta.data - self.lr * m_next_hat / ((v_next_hat ** 0.5) + self.eps)
            theta.data = ndl.Tensor(theta_val.numpy().astype(np.float32))
            # print(theta)

        ### END YOUR SOLUTION
