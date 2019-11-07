import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import time
import sys
import utils


# 获取类别标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle', 'boot']
    return [text_labels[int(i)] for i in labels]


# 展示多张图片
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 定义 softmax
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 广播机制


print('初始化模型参数')
batch_size = 256
num_worker = 4
num_inputs = 784
num_outputs = 10
num_epochs = 10
lr = 0.1

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
W.requires_grad_(requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float)
b.requires_grad_(requires_grad=True)


print('定义模型')
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


print('定义损失函数')
def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


print('计算分类准确率')
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

print('下载数据集')
mnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transforms.ToTensor())
print('数据集大小')
print(len(mnist_train), len(mnist_test))
print('访问第一个样本')
feature, label = mnist_train[0]
print(feature.shape, label)
print('查看前 9 张图片')
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
print('读取数据')
train_iter = Data.DataLoader(mnist_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_worker)
test_iter = Data.DataLoader(mnist_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_worker)
print('完整读取一次训练数据所需要的时间')
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
print('模型训练')


def train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                utils.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_softmax(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
print('预测并输出结果')
X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])

'''
完整读取一次训练数据所需要的时间
1.35 sec
模型训练
epoch 1, loss 0.7861, train acc 0.747, test acc 0.796
epoch 2, loss 0.5704, train acc 0.814, test acc 0.813
epoch 3, loss 0.5244, train acc 0.827, test acc 0.821
epoch 4, loss 0.5013, train acc 0.832, test acc 0.825
epoch 5, loss 0.4858, train acc 0.836, test acc 0.826
epoch 6, loss 0.4746, train acc 0.840, test acc 0.826
epoch 7, loss 0.4649, train acc 0.843, test acc 0.832
epoch 8, loss 0.4582, train acc 0.844, test acc 0.833
epoch 9, loss 0.4519, train acc 0.846, test acc 0.833
epoch 10, loss 0.4468, train acc 0.848, test acc 0.836
预测并输出结果
'''