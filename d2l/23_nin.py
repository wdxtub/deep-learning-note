import time
import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch import nn, optim
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('定义 NiN Block')


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk


print('定义网络')
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    # 全局平均池化层可通过将窗口形状设置成输入的高和宽实现
    nn.AvgPool2d(kernel_size=5),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    utils.FlattenLayer())


print('查看网络输出')
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

print('训练模型，轮数为 1')
lr, num_epochs = 0.002, 1
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

'''
训练模型，轮数为 1
training on cpu
epoch 1, loss 0.0056, train acc 0.483, test acc 0.727, time 4930.7 sec
'''