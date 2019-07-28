import sys, os
import numpy as np
from common import softmax, cross_entropy_error, numerical_gradient


# 只有输入和输出层
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 用高斯分布初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


print('测试一下')
net = SimpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
