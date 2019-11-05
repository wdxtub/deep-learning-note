import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.nn import init



true_w = [2, -3.4]
true_b = 4.2
num_inputs = 2
batch_size = 10
num_epochs = 3


# 生成数据集
def generate_dataset(true_w, true_b):
    num_examples = 1000

    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    # 真实 label
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 添加噪声
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    # 展示下分布
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    plt.show()

    return features, labels


print('生成数据集')
features, labels = generate_dataset(true_w, true_b)
print('组合特征与标签')
dataset = Data.TensorDataset(features, labels)
print('随机读取小批量')
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
print('打印第一个 batch')
for X, y in data_iter:
    print(X, y)
    break
print('定义模型')


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)
print('我们也可以直接用 Sequential 来搭建，就不用定义类这么麻烦')
new_net_1 = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
print(new_net_1)

new_net_2 = nn.Sequential()
new_net_2.add_module('linear', nn.Linear(num_inputs, 1))
print(new_net_2)

new_net_3 = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
]))
print(new_net_3)

print('查看可学习的参数')
for param in new_net_1.parameters():
    print(param)
# 后面我们用 new_net_1 来操作
print('初始化模型参数')
init.normal(new_net_1[0].weight, mean=0, std=0.01)
init.constant_(new_net_1[0].bias, val=0) # 或 new_net_1[0].bias.data.fill_(0)
print('定义损失函数')
loss = nn.MSELoss()
print('定义优化算法')
optimizer = optim.SGD(new_net_1.parameters(), lr=0.03)
print(optimizer)
print('训练模型')
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = new_net_1(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
print('比较学习到的参数')
dense = new_net_1[0]
print(true_w, dense.weight)
print(true_b, dense.bias)


