import torch
from torch import nn

print('保存一个 Tensor')
x = torch.ones(3)
x_path = 'data/x.pt'
print(x)
torch.save(x, x_path)
print('再读取回来')
new_x = torch.load(x_path)
print(new_x)
print('保存一个 Tensor List')
y = torch.zeros(4)
xy_path = 'data/xy.pt'
print([x, y])
torch.save([x, y], xy_path)
print('再读取回来')
xy_list = torch.load(xy_path)
print(xy_list)
print('保存一个 Tensor Dict')
xy_dict_path = 'data/xy_dict.pt'
torch.save({'x': x, 'y': y}, xy_dict_path)
xy_dict = torch.load(xy_dict_path)
print(xy_dict)

print('state_dict 包含网络的参数')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
print(net.state_dict())
print('只有可学习参数的层才有 state_dict，优化器也有')
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())
print('保存和加载模型参数（推荐）')
model_param_path = 'data/model_param.pt'
torch.save(net.state_dict(), model_param_path)
new_model = MLP()
new_model.load_state_dict(torch.load(model_param_path))
print(new_model)
print('保存和加载整个模型')
model_path = 'data/model.pt'
torch.save(net, model_path)
another_model = torch.load(model_path)
print('验证一下是不是能够得到一样的结果')
X = torch.randn(2, 3)
Y = net(X)
print('Origin Y:', Y)
Y1 = new_model(X)
print('Y1:', Y1)
Y2 = another_model(X)
print('Y2:', Y2)
print('这些 Y 应该相同')
