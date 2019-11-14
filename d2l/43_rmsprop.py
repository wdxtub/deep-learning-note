import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
import utils
import math


print('定义 rmsprop')
eta = 0.4
gamma = 0.9


def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


print('lr= 0.4 的轨迹')
utils.show_trace_2d(f_2d, utils.train_2d(rmsprop_2d))

print('自行实现 RMSprop')

features, labels = utils.get_nasa_data()


def init_rmsprop_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def rmsprop(params, states, hyperparams):
    eps = 1e-6
    gamma = hyperparams['gamma']
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1-gamma) * (p.grad.data)**2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


print('RMSProp 进行优化')
utils.train_opt(rmsprop, init_rmsprop_states(), {'lr': 0.1, 'gamma': 0.9}, features, labels)

print('简洁实现')
utils.train_opt_pytorch(optim.RMSprop, {'lr': 0.1, 'alpha': 0.9}, features, labels)
