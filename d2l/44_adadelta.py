import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
import utils
import math

print('载入数据')
features, labels = utils.get_nasa_data()


print('定义 AdaDelta')


def init_adadelta_states():
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))


def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * (p.grad.data**2)
        g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta[:] = rho * delta + (1 - rho) * g * g


print('用超参数 0.9 来训练')
utils.train_opt(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)

print('简洁实现')
utils.train_opt_pytorch(optim.Adadelta, {'rho': 0.9}, features, labels)
