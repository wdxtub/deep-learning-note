import torch
from torch import nn


print('定义不含模型参数的 Layer')


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


print('实例化并做前向计算')
layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))
print('构造更复杂的模型')
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
print(y.mean().item())
print('定义包含模型参数的 Layer')


class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyListDense()
print(net)


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        # 新增一个
        self.params.update({
            'linear3': nn.Parameter(torch.randn(4, 2))
        })

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


net = MyDictDense()
print(net)
print('可以传入不同的 key 进行不同的前向传播')
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))
print('也可以把我们的自定义层，和其他层组合')
net = nn.Sequential(
    MyDictDense(),
    MyListDense(),
)
print(net)
print(net(x))
