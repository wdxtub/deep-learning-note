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
num_worker = 4
mnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transforms.ToTensor())
train_iter = Data.DataLoader(mnist_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_worker)
test_iter = Data.DataLoader(mnist_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_worker)

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
