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
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

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

'''
训练模型
epoch 1, loss 0.0031, train acc 0.748, test acc 0.783
epoch 2, loss 0.0022, train acc 0.813, test acc 0.799
epoch 3, loss 0.0021, train acc 0.826, test acc 0.807
epoch 4, loss 0.0020, train acc 0.833, test acc 0.820
epoch 5, loss 0.0019, train acc 0.836, test acc 0.824
epoch 6, loss 0.0019, train acc 0.840, test acc 0.828
epoch 7, loss 0.0018, train acc 0.843, test acc 0.772
epoch 8, loss 0.0018, train acc 0.844, test acc 0.831
epoch 9, loss 0.0018, train acc 0.847, test acc 0.830
epoch 10, loss 0.0018, train acc 0.848, test acc 0.836
'''
