import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('定义改良的 BatchNorm + Activation + 卷积')


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk


print('定义 DenseBlock')


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 在通道维上连接
        return X

print('卷积块的通道数控制了增长，也称为增长率')
blk = DenseBlock(2, 3, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

print('定义 过渡层')


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk


print('对前面的稠密块输出使用')
blk = transition_block(23, 10)
print(blk(Y).shape)


print('构造 DenseNet')
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

num_channels, growth_rate = 64, 32
num_convs_in_dense_block = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_block):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module('DenseBlock_%d' % i, DB)
    # 上一个稠密块的输出通道数
    num_channels = DB.out_channels
    # 在稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_block) - 1:
        net.add_module('transition_block_%d' % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 加入全局平均池化和全连接
net.add_module('BN', nn.BatchNorm2d(num_channels))
net.add_module('relu', nn.ReLU())
net.add_module('global_avg_pool', utils.GlobalAvgPool2d())
net.add_module('fc', nn.Sequential(utils.FlattenLayer(), nn.Linear(num_channels, 10)))

print('确保网络无误')
X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    print(name, 'output shape:\t', X.shape)

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=96)

print('训练模型')
lr, num_epochs = 0.001, 2
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

'''
training on cpu
epoch 1, loss 0.0018, train acc 0.835, test acc 0.825, time 2152.4 sec
epoch 2, loss 0.0011, train acc 0.899, test acc 0.873, time 2131.0 sec
'''

