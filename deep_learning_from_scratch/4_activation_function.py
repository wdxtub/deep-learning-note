import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    return 0


# 支持 Numpy 数组的实现
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# 简单测试一下
x = np.array([-1.0, 1.0, 2.0])
print(x)
print(step_function(x))

# 绘制图形
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
title = 'step function'

# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

# 广播
t = np.array([1.0, 2.0, 3.0])
print(1+t)
print(1/t)
# 绘制图形
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
title1 = 'sigmoid'
title2 = 'relu'
y2 = relu(x)
plt.plot(x, y1, label=title1)
plt.plot(x, y, label=title, linestyle='--')
plt.plot(x, y2, label=title2)
plt.ylim(-0.1, 5.1)
plt.title('activation functions')
plt.legend()
plt.show()
