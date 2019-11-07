import time
import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch import nn, optim
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('定义 VGG Block')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


print('构造 VGG-11')
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
print('经过 5 个 vgg_block，宽高会减半 5 次，224/2^5 = 7')
fc_features = 512 * 7 * 7 # channel * w * h
fc_hidden_units = 4096 # 都行


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个 vgg_block 宽高减半
        net.add_module('vgg_block_' + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接部分
    net.add_module('fc', nn.Sequential(
        utils.FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net


print('观察下每一层的输出形状')
net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape:', X.shape)


print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

print('构造一个更小的网络，方便训练，因为量计算量更大了')
ratio = 16
small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio),
                   (2, 128 // ratio, 256 // ratio), (2, 256 // ratio, 512 // ratio),
                   (2, 512 // ratio, 512 // ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)
print('进行训练，只训练 1 轮')
lr, num_epochs = 0.001, 1
optimizer = optim.Adam(net.parameters(), lr=lr)
utils.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

'''
进行训练，只训练 1 轮
training on cpu
epoch 1, loss 0.0041, train acc 0.600, test acc 0.831, time 876.0 sec
'''
