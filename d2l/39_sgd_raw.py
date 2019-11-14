import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import utils


def get_nasa_data():
    data = np.genfromtxt('data/NASA/airfoil_self_noise.dat')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), torch.tensor(data[:1500, -1], dtype=torch.float32)


print('读取数据')
features, labels = get_nasa_data()
print(features.shape)

print('随机梯度下降函数')


def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


def train_sgd(lr, batch_size, num_epochs=2):
    utils.train_opt(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


print('梯度下降，学习率为 1')
train_sgd(1, 1500, 6)
print('随机梯度下降，学习率为 0.005')
train_sgd(0.005, 1)
print('小批量梯度下降')
train_sgd(0.05, 10)


