import numpy as np
import time

print('1. 创建大小为 10 的空向量')
a = np.zeros(10)
print(a)

print('2. 查看矩阵占据的内存大小')
print('用元素个数乘以每个元素的大小')
print(f'占据 {a.size * a.itemsize} 字节')

print('3. 创建一个向量，值从 10 到 49')
a = np.arange(10, 50)
print(a)

print('4. 翻转一个向量')
a = a[::-1]
print(a)

print('5. 创建一个 3x3 的矩阵，值从 0 到 8')
a = np.arange(9).reshape(3,3)
print(a)

print('6. 从 [1, 2, 0, 0, 4, 0] 中寻找非零元素索引')
nz = np.nonzero([1, 2, 0, 0, 4, 0])
print(nz)

print('7. 创建 3x3 单位矩阵（对角线元素为 1 的方阵）')
a = np.eye(3)
print(a)

print('8. 创建一个 3x3x3 的随机矩阵')
a = np.random.random((3, 3, 3))
print(a)

print('9. 创建一个 10x10 的矩阵并寻找最大最小值')
a = np.random.random((10, 10))
a_min, a_max = a.min(), a.max()
print('min', a_min, ', max', a_max)

print('10. 创建一个长度为 30 的向量，并求均值')
a = np.random.random(30)
print('mean', a.mean())

print('11. 创建一个边界为 1 其他为 0 的二维矩阵')
a = np.ones((10, 10))
a[1:-1,1:-1] = 0
print(a)

print('12. 为已经存在的矩阵填充 0 的边界')
a = np.ones((5, 5))
print(a)
a = np.pad(a, pad_width=1, mode='constant', constant_values=0)
print(a)

print('13. 给出下列计算的结果')
print('0 * np.nan =', 0 * np.nan)
print('np.nan == np.nan =', np.nan == np.nan)
print('np.inf > np.nan =', np.inf > np.nan)
print('np.nan - np.nan =', np.nan - np.nan)
print('np.nan in set([np.nan]) =', np.nan in set([np.nan]))
print('0.3 == 3 * 0.1 =', 0.3 == 3 * 0.1)

print('14. 创建一个 5x5 的矩阵，对角线下的数值为 1 2 3 4')
a = np.diag(1 + np.arange(4), k=-1)
print(a)

print('15. 创建一个 8x8 矩阵，其中 0 和 1 间隔分布')
a = np.zeros((8, 8), dtype=int)
a[1::2, ::2] = 1
a[::2, 1::2] = 1
print(a)

print('16. 使用 tile 函数创建一个 8x8 矩阵，其中 0 和 1 间隔分布')
a = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(a)

print('17. 假设有一个 (6, 7, 8) 大小的矩阵，那么第 100 个元素的索引是多少')
print(np.unravel_index(100, (6, 7, 8)))

print('18. 归一化一个随机 5x5 矩阵')
a = np.random.random((5, 5))
a = (a - np.mean(a)) / np.std(a)
print(a)

print('19. 点乘一个 5x3 和 3x2 的矩阵')
a = np.dot(np.ones((5, 3)), np.ones((3, 2)))
print(a)

print('20. 给定一个一维数组，不新增空间，把 3~8 之间的数字变成负数')
a = np.arange(10)
a[(3 < a) & (a <= 8)] *= -1
print(a)

print('21. 两个数组求交集')
a1 = np.random.randint(0, 10, 10)
a2 = np.random.randint(0, 10, 10)
print(np.intersect1d(a1, a2))

print('22. 获取 2020 年 6 月的所有日期')
a = np.arange('2020-06', '2020-07', dtype='datetime64[D]')
print(a)

print('23. 用 5 种方法去掉小数部分')
a = np.random.uniform(0, 10, 10)
print('a', a)
print('1:', a - a%1)
print('2:', np.floor(a))
print('3:', np.ceil(a) - 1)
print('4:', a.astype(int))
print('5:', np.trunc(a))

print('24. 创建一个 5x5 的矩阵，每一行都是从 0 到 4')
a = np.zeros((5, 5))
a += np.arange(5)
print(a)

print('25. 创建一个大小为 10，值从 0 到 1 的向量(不包括 0 和 1)')
a = np.linspace(0, 1, 11, endpoint=False)[1:]
print(a)

print('26. 创建一个大小为 10 的随机向量并排序')
a = np.random.random(10)
a.sort()
print(a)

print('27. 如何用比 np.sum 更快的方法对一个小数组求和')
a = np.arange(10)
print('a', a)
start = time.time()
print('add.reduct', np.add.reduce(a))
end = time.time()
print('add.reduce time:', end-start)
start = time.time()
print('np.sum', np.sum(a))
end = time.time()
print('np.sum time:', end - start)

print('28. 比较两个数组是否相等')
a = np.random.randint(0, 10, 10)
b = np.random.randint(0, 10, 10)
print(np.allclose(a, b))
print(np.array_equal(a, b))

print('29. 将一个 10x2 的笛卡尔坐标系的点转成极坐标')
a = np.random.random((10, 2))
x, y = a[:, 0], a[:, 1]
r = np.sqrt(x**2 + y**2)
t = np.arctan2(y, x)
print(r)
print(t)

print('30. 创建一个大小为 10 的随机向量，并将最大的替换成 0')
a = np.random.random(10)
print('before', a)
a[a.argmax()] = 0
print('after', a)

print('31. 不用额外空间将 float 矩阵变成 int 矩阵')
a = np.arange(10, dtype=np.float32)
a = a.astype(np.int32, copy=False)
print(a)

print('32. 在一个 2 维矩阵中随机放 p 个元素')
n, p = 10, 3
a = np.zeros((n, n))
np.put(a, np.random.choice(range(n*n), p, replace=False), 1)
print(a)

print('33. 矩阵的每行减去每行的均值')
a = np.random.randint(0, 10, (5, 10))
print('before', a)
b = a - a.mean(axis=1, keepdims=True)
print('after', b)

print('34. 根据第 i 列给矩阵排序')
a = np.random.randint(0, 10, (3, 3))
print('before', a)
print('after', a[a[:, 1].argsort()])

print('35. 交换矩阵的两行')
a = np.arange(25).reshape(5, 5)
a[[0,1]] = a[[1, 0]]
print(a)

print('36. 如何计算一个数组的滑动窗口')
def moving_averate(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n
a = np.arange(20)
print(moving_averate(a, n=4))

print('37. 如何找到数组中出现次数最多的元素')
a = np.random.randint(0, 10, 50)
print(np.bincount(a).argmax())

print('38. 如何获取数组中最大的 n 个数')
a = np.arange(1000)
np.random.shuffle(a)
n = 5
start = time.time()
print('slow', a[np.argsort(a)[-n:]])
end = time.time()
print('slow time', end - start)
start = time.time()
print('fast', a[np.argpartition(-a, n)[:n]])
end = time.time()
print('fast time', end - start)