import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('定义残差层')


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


print('检查输入与输出的尺寸')
blk = Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape)
print('增加输出通道数，减半高和宽')
blk = Residual(3, 6, use_1x1conv=True, stride=2)
print(blk(X).shape)

print('定义残差块')


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数与输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


print('构造 ResNet')
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
net.add_module('resnet_block2', resnet_block(64, 128, 2))
net.add_module('resnet_block3', resnet_block(128, 256, 2))
net.add_module('resnet_block4', resnet_block(256, 512, 2))
# 加入全局平均池化和全连接
net.add_module('global_avg_pool', utils.GlobalAvgPool2d())
net.add_module('fc', nn.Sequential(utils.FlattenLayer(), nn.Linear(512, 10)))

print('查看不同模块的尺寸')
X = torch.rand((1, 1, 224, 224))
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

'''