import torch
from torch import nn


print('定义 MLP 类')


class MLP(nn.Module):
    # 声明俩全连接层
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 前向计算
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
    # 注：无需定义反向传播，系统会自动生成 backward 函数


print('测试 MLP 类')
X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))


print('定义 Fancy MLP 类')


class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        # 不训练的参数
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        # 复用全连接层
        x = self.linear(x)
        # 对 x 值进行处理
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10

        return x.sum()


print('测试 Fancy MLP 类')
X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))