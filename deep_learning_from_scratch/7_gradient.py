import numpy as np


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 交叉熵误差
# 支持单个和 batch
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# 假设 第三个位置是正确的
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print('第三个位置的概率最高情况')
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print('均方误差', mean_squared_error(np.array(y), np.array(t)))
print('交叉熵误差', cross_entropy_error(np.array(y), np.array(t)))

print('第八个位置的概率最高的情况')
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print('均方误差', mean_squared_error(np.array(y), np.array(t)))
print('交叉熵误差', cross_entropy_error(np.array(y), np.array(t)))


# 数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)


# 梯度计算
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h) 的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function_2(x):
    return x[0]**2 + x[1]**2


print('用梯度下降计算函数 function_2 的最小值')
init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(result)

