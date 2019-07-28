# 这里的两层网络，指除了输入层还有两层，就是只有一个隐层
import sys, os
import numpy as np
from common.functions import *
from common.gradient import *
from dataset import load_mnist


# 只有输入和输出层
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # 用高斯分布初始化
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)

        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


print('测试一下')
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print('param-W1-shape', net.params['W1'].shape)
print('param-b1-shape', net.params['b1'].shape)
print('param-W2-shape', net.params['W2'].shape)
print('param-b2-shape', net.params['b2'].shape)
print('输入伪数据，这里的梯度计算比较低效，建议不执行')
x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)
print('grads-W1-shape', grads['W1'].shape)
print('grads-b1-shape', grads['b1'].shape)
print('grads-W2-shape', grads['W2'].shape)
print('grads-b2-shape', grads['b2'].shape)

print('Mini Batch 的实现')
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 平均每个 epoch 的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 获取 mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 计算每个 epoch 的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'train acc, test acc | {str(train_acc)}, {str(test_acc)}')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 绘制图像
epoch_list = [i for i in range(iters_num)]
plt.plot(epoch_list, test_acc_list, label='test acc')
plt.plot(epoch_list, train_acc_list, label='train acc', linestyle='--')
plt.ylim(-0.0, 1.0)
plt.title('train/test accuracy')
plt.legend()
plt.show()
