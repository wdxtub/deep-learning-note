import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('定义 BatchNorm 函数')


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 训练模式和预测模式逻辑不同
    if not is_training:
        # 预测模式下，直接使用传入的移动平均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层，二维数组，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用卷积层，三维数组
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) *  var
    Y = gamma * X_hat + beta # 拉伸和偏移
    return Y, moving_mean, moving_var


print('定义 BatchNorm 层')


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            # 全连接
            shape = (1, num_features)
        else:
            # 卷积
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y


print('给 LeNet 加上 BatchNorm')

net = nn.Sequential(
    nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2), #kernel_size, stride
    nn.Conv2d(6, 16, 5),
    BatchNorm(16, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    utils.FlattenLayer(),
    nn.Linear(16*4*4, 120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),
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

print('查看拉伸参数 gamma 和 beta')
print('gamma', net[1].gamma.view(-1, ))
print('beta', net[1].beta.view(-1, ))

'''
训练模型
training on cpu
epoch 1, loss 0.0040, train acc 0.778, test acc 0.830, time 19.3 sec
epoch 2, loss 0.0018, train acc 0.861, test acc 0.829, time 21.3 sec
epoch 3, loss 0.0015, train acc 0.877, test acc 0.824, time 21.8 sec
epoch 4, loss 0.0013, train acc 0.884, test acc 0.836, time 21.7 sec
epoch 5, loss 0.0012, train acc 0.891, test acc 0.827, time 21.4 sec
查看拉伸参数 gamma 和 beta
gamma tensor([1.2641, 1.1758, 1.1133, 1.0465, 0.9625, 0.8872],
       grad_fn=<ViewBackward>)
beta tensor([-0.0531, -0.8111,  0.0695,  0.3452, -0.4088,  0.1445],
       grad_fn=<ViewBackward>)
'''
