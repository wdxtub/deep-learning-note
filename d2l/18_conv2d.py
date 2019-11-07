import torch
from torch import nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] -h+1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y


print('验证一下二维互相关')
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))
print('定义二维卷积')


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


print('定义一张带变化的图像')
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)
print('边缘检测，边缘变成非 0')
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)
print('前面的 K 是我们定义的，现在我们学习出 K')
conv2d = Conv2D(kernel_size=(1, 2))
step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i+1, l.item()))
print('查看学习到的卷积核')
print('weight:', conv2d.weight.data)
print('bias:', conv2d.bias.data)
print('可以看到和我们之前定义的 K 很接近')