import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
import utils

print('通过实际例子查看梯度下降的问题')
eta = 0.4


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


print('竖直方向波动较大')
utils.show_trace_2d(f_2d, utils.train_2d(gd_2d))
print('在竖直方向越过最优解并发散')
eta = 0.6
utils.show_trace_2d(f_2d, utils.train_2d(gd_2d))

print('使用动量法进行改进')
eta, gamma = 0.4, 0.5


def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


utils.show_trace_2d(f_2d, utils.train_2d(momentum_2d))
print('用更大的学习率，也不会发散')
eta = 0.6
utils.show_trace_2d(f_2d, utils.train_2d(momentum_2d))

print('自行实现动量法优化')
features, labels = utils.get_nasa_data()


def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data


print('momentum = 0.02, lr = 0.02')
utils.train_opt(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.5}, features, labels)
print('momentum = 0.9, lr = 0.02')
utils.train_opt(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.9}, features, labels)
print('momentum = 0.9, lr = 0.004')
utils.train_opt(sgd_momentum, init_momentum_states(), {'lr': 0.004, 'momentum': 0.9}, features, labels)

print('框架实现')
utils.train_opt_pytorch(optim.SGD, {'lr': 0.004, 'momentum': 0.9}, features, labels)
