import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import numpy as np
import utils

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

print('定义模型')
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    utils.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

print('训练模型')
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 10
utils.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

'''
训练模型
epoch 1, loss 0.0032, train acc 0.694, test acc 0.804
epoch 2, loss 0.0019, train acc 0.823, test acc 0.805
epoch 3, loss 0.0017, train acc 0.840, test acc 0.832
epoch 4, loss 0.0015, train acc 0.856, test acc 0.848
epoch 5, loss 0.0014, train acc 0.866, test acc 0.847
epoch 6, loss 0.0014, train acc 0.871, test acc 0.867
epoch 7, loss 0.0013, train acc 0.875, test acc 0.853
epoch 8, loss 0.0013, train acc 0.881, test acc 0.844
epoch 9, loss 0.0012, train acc 0.883, test acc 0.858
epoch 10, loss 0.0012, train acc 0.888, test acc 0.860
'''