import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
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

print('定义和初始化模型')
num_inputs = 784
num_outputs = 10


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
print('softmax 和交叉熵损失函数')
loss = nn.CrossEntropyLoss()
print('定义优化算法')
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
print('训练模型')
num_epochs = 10
utils.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)