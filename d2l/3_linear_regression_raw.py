import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


true_w = [2, -3.4]
true_b = 4.2
num_inputs = 2


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


# batch 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个 batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 模型定义
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


print('生成数据集')
features, labels = generate_dataset(true_w, true_b)
print('尝试读取第一个 batch')
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
print('初始化模型参数')
num_inputs = 2
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
b = torch.zeros(1, dtype=torch.float)
# w 和 b 需要梯度进行更新，需要设置下
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
print('模型训练')
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        # 注意梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print('比较 w 和 b')
print(true_w, '\n', w)
print(true_b, 'vs', b.item())
print('可以看到，经过 3 轮训练，就已经非常接近了')