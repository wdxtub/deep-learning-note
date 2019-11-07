import torch
from torch import nn
import utils


print('定义多通道互相关输入')


def corr2d_multi_in(X, K):
    # 分别计算再相加
    res = utils.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += utils.corr2d(X[i, :, :], K[i, :, :])
    return res


print('验证一下')
X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])
print(corr2d_multi_in(X, K))

print('定义多通道互相关输出')


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


print('验证一下')
K = torch.stack([K, K+1, K+2])
print(K.shape)
print(corr2d_multi_in_out(X, K))
print('第一个通道输出应该和前面的一样')

print('定义 1x1 卷积')


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h*w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X) # 全连接矩阵乘法
    return Y.view(c_o, h, w)


print('验证一下，与 corr2d_multi_in_out 一致')
X = torch.rand(3, 3, 3)
print(X)
K = torch.rand(2, 3, 1, 1)
print(K)
Y1 = corr2d_multi_in_out_1x1(X, K)
print(Y1)
Y2 = corr2d_multi_in_out(X, K)
print((Y1-Y2).norm().item())