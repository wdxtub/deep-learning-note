import torch
from torch import nn
from torch.nn import init


print('定义网络，默认会进行初始化')
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

print(net)
X = torch.rand(2, 4)
print(X)
Y = net(X).sum()
print(Y)

print('访问模型参数')
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

print('查看第一层的参数')
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))

print('初始化模型参数')
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


print('自定义初始化方法')


def init_weight_(tensor):
    with torch.no_grad(): # 在这里的不会计算梯度，因为我们不需要！
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)


print('共享模型参数')
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

print('在内存中他们的地址也是一样的')
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))
print('共享参数梯度累加，单次 3，两次就是 6')
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)

