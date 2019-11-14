import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


print('目标函数 f(x) = x*x')


def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x # 导数
        results.append(x)
    print('epoch 10, x:', x)
    return results


print('进行一维梯度下降')
res = gd(0.2)
print(res)
print('绘制迭代轨迹')


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    plt.plot(f_line, [x * x for x in f_line])
    plt.plot(res, [x * x for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


show_trace(res)

print('尝试不同学习率，0.05')
show_trace(gd(0.05))

print('尝试不同学习率，1.1')
show_trace(gd(1.1))

print('进行二维梯度下降')


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


eta = 0.1


def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)


show_trace_2d(f_2d, train_2d(gd_2d))

print('进行随机梯度下降')


def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)


show_trace_2d(f_2d, train_2d(sgd_2d))
