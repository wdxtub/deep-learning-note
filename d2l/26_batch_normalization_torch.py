import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('给 LeNet 加上 BatchNorm')
print('对于全连接层用 BatchNorm1d，对于卷积层，用 BatchNorm2d')
net = nn.Sequential(
    nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2), #kernel_size, stride
    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    utils.FlattenLayer(),
    nn.Linear(16*4*4, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
print(net)

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)

print('训练模型')
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


'''
training on cpu
epoch 1, loss 0.0038, train acc 0.791, test acc 0.821, time 20.5 sec
epoch 2, loss 0.0018, train acc 0.866, test acc 0.854, time 21.8 sec
epoch 3, loss 0.0014, train acc 0.881, test acc 0.872, time 22.8 sec
epoch 4, loss 0.0013, train acc 0.889, test acc 0.787, time 22.2 sec
epoch 5, loss 0.0012, train acc 0.895, test acc 0.861, time 22.2 sec
'''