import torch
import numpy as np

print('创建空 Tensor')
x = torch.empty(5, 3)
print(x)

print('创建随机初始化的 Tensor')
x = torch.rand(5, 3)
print(x)

print('从数组创建 Tensor')
x = torch.tensor([5.5, 3])
print(x)

print('创建全 0 的 Tensor，类型为 long')
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

print('通过 shape 或者 size() 获取 Tensor 形状')
print(x.size())
print(x.shape)

print('几种不同的加法形式')
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# inplace
y.add_(x)
print(y)

print('索引，共享内存')
y = x[0, :]
y += 1
print(y)
print(x[0, :])

print('用 view 改变 Tensor 形状')
y = x.view(15)
z = x.view(-1, 5) # -1 表示根据其他维度值来推算出
print(x.size(), y.size(), z.size())

print('深拷贝')
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

print('item 转换')
x = torch.randn(1)
print(x)
print(x.item())

print('广播机制')
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

print('是否开辟新内存的例子')
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)
print('不修改内存地址')
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)
print('另一种不修改的方法')
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before)

print('Tensor 转 Numpy')
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

print('Numpy 转 Tensor')
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

print('复制拷贝')
c = torch.tensor(a)
a += 1
print(a, c)
